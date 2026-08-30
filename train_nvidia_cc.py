"""Pretrain the ~0.95B text-only Gemma-4 model on nvidia/Nemotron-CC-v2 via HF Trainer.

Model surgery / warm-start lives in ``model_prep.py``; the streaming data pipeline
and the benchmark eval corpora live in ``data_prep.py``. This file is just the
training loop, the WSD schedule, the token-accounting callback, and the in-loop
benchmark callback.

Run from inside gemma/:
    python train_nvidia_cc.py            # fresh run
    python train_nvidia_cc.py --resume   # continue from ./checkpoints-fineweb

Needs HF_TOKEN (gated dataset + tokenizer/weights) and, optionally, WANDB_API_KEY
in ../.env or .env.
"""

import argparse
import math
import os

import torch
import wandb
from dotenv import load_dotenv
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.integrations import WandbCallback

from model_prep import CONTEXT_LEN, NUM_LAYERS, REF_MODEL, WARM_START, build_model, load_tokenizer
from data_prep import DATASET, DATASET_CONFIG, SEED, build_eval_corpora, build_train_dataset

load_dotenv()  # WANDB_API_KEY for logging (HF_TOKEN is read by model_prep / data_prep)

# --------------------------------------------------------------------------- #
# Knobs
# --------------------------------------------------------------------------- #
OUTPUT_DIR     = "./checkpoints-fineweb"

MICRO_BATCH    = 10                         # per-device
GRAD_ACCUM     = 4                          # -> effective 10*4*2048 = 82k tokens/step (1 GPU)
MAX_STEPS      = 20_000
WARMUP_STEPS   = 500
PEAK_LR        = 3e-4

# WSD (warmup-stable-decay) LR schedule, matching the Qwen pretrain scripts:
# linear warmup -> hold at PEAK_LR -> cosine decay to MIN_LR_RATIO * PEAK_LR over
# the final DECAY_STEPS. warmup + stable + decay must sum to MAX_STEPS
# (stable = 20_000 - 500 - 3_000 = 16_500).
MIN_LR_RATIO   = 0.1
DECAY_STEPS    = 3_000

WANDB_PROJECT  = "pretrain-gemma-4-0.95b"

# Benchmark suite: run every EVAL_STEPS optimizer steps (blocking, rank 0 only).
#   * WikiText + Nemotron held-out  -> loss / perplexity  (_lm_loss)
#   * HellaSwag, PIQA, ARC-Challenge, WinoGrande, MMLU -> accuracy  (_mc_acc)
# The corpora / example sets are built in data_prep.build_eval_corpora (with the
# per-task size knobs there); these knobs only control the in-loop scoring.
EVAL_STEPS = 1_000
EVAL_BATCH = 8                              # seqs per forward for the LM-loss evals
MC_BATCH   = 32                             # (example, choice) rows per forward for the MC evals

# multiple-choice tasks that also get an acc_norm (length-normalised) metric;
# WinoGrande / MMLU score a fixed-length continuation so acc_norm == acc there.
MC_NORM_TASKS = {"hellaswag", "piqa", "arc_challenge"}


class TokensSeenCallback(TrainerCallback):
    """Add train/tokens_seen to every log record, like the Qwen pretrain scripts.

    Derived from the optimizer-step count (global_step * effective batch * seq_len),
    so it costs nothing -- unlike TrainingArguments.include_num_input_tokens_seen,
    which does an all-gather + .item() sync every step. Inserted just ahead of
    Trainer's WandbCallback so the key rides the normal wandb report and gets the
    `train/` prefix from rewrite_logs().
    """

    def __init__(self, seq_len):
        self.seq_len = seq_len

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logs["tokens_seen"] = (
                state.global_step
                * args.per_device_train_batch_size
                * args.gradient_accumulation_steps
                * args.world_size
                * self.seq_len
            )


# --------------------------------------------------------------------------- #
# Benchmark scoring
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _lm_loss(model, packed_ds, device, batch_size):
    """Mean token cross-entropy over a packed LM dataset (teacher forcing)."""
    tot_loss, tot_tok = 0.0, 0
    for i in range(0, len(packed_ds), batch_size):
        ids = torch.tensor(packed_ds[i : i + batch_size]["input_ids"], device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            loss = model(input_ids=ids, labels=ids).loss
        n = ids.numel() - ids.shape[0]          # shifted-target count
        tot_loss += loss.item() * n
        tot_tok += n
    return tot_loss / max(1, tot_tok)


@torch.no_grad()
def _mc_acc(model, tokenizer, examples, device, batch_size):
    """Multiple-choice scoring by continuation loglikelihood -> (acc, acc_norm).

    Each example is ``{"pairs": [(context, continuation), ...], "gold": int}``.
    The continuation tokens are scored given their context; the prediction is the
    argmax over summed LL (``acc``) and per-token mean LL (``acc_norm``). Choice
    counts may vary between examples (ARC) -- unused slots stay at -inf."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    n_choices = max(len(ex["pairs"]) for ex in examples)
    rows = []                                    # (ex_idx, choice_idx, ctx_len, full_ids)
    for ei, ex in enumerate(examples):
        for k, (ctx, cont) in enumerate(ex["pairs"]):
            ctx_ids = tokenizer(ctx)["input_ids"]
            full_ids = tokenizer(ctx + cont)["input_ids"]
            clen = min(len(ctx_ids), len(full_ids) - 1)   # keep >=1 target token
            rows.append((ei, k, clen, full_ids))

    ll = torch.full((len(examples), n_choices), float("-inf"))
    ll_norm = torch.full((len(examples), n_choices), float("-inf"))
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        maxlen = max(len(f) for *_, f in chunk)
        inp = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for j, (*_, f) in enumerate(chunk):
            inp[j, : len(f)] = torch.tensor(f)
            attn[j, : len(f)] = 1
        inp, attn = inp.to(device), attn.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logp = torch.log_softmax(model(input_ids=inp, attention_mask=attn).logits.float(), dim=-1)
        for j, (ei, k, clen, f) in enumerate(chunk):
            tgt = torch.tensor(f[clen:], device=device)
            lp = logp[j, clen - 1 : len(f) - 1, :].gather(-1, tgt[:, None]).squeeze(-1)
            ll[ei, k] = lp.sum().item()
            ll_norm[ei, k] = lp.mean().item()
    gold = torch.tensor([ex["gold"] for ex in examples])
    return (
        (ll.argmax(-1) == gold).float().mean().item(),
        (ll_norm.argmax(-1) == gold).float().mean().item(),
    )


class BenchmarkCallback(TrainerCallback):
    """Every EVAL_STEPS steps: WikiText + Nemotron perplexity and the
    multiple-choice accuracy benchmarks (HellaSwag, PIQA, ARC-Challenge,
    WinoGrande, MMLU), logged to wandb against train/global_step (rank 0,
    blocking)."""

    def __init__(self, tokenizer, wikitext_ds, nemotron_ds, mc_tasks):
        self.tok = tokenizer
        self.wikitext_ds = wikitext_ds
        self.nemotron_ds = nemotron_ds
        self.mc_tasks = mc_tasks

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % EVAL_STEPS != 0:
            return
        if not state.is_world_process_zero:
            return
        was_training = model.training
        model.eval()
        m = {}
        for name, ds in (("wikitext", self.wikitext_ds), ("nemotron", self.nemotron_ds)):
            loss = _lm_loss(model, ds, args.device, EVAL_BATCH)
            m[f"eval/{name}_loss"] = loss
            m[f"eval/{name}_ppl"] = math.exp(loss)
        for task, examples in self.mc_tasks.items():
            acc, acc_norm = _mc_acc(model, self.tok, examples, args.device, MC_BATCH)
            m[f"eval/{task}_acc"] = acc
            if task in MC_NORM_TASKS:
                m[f"eval/{task}_acc_norm"] = acc_norm
        if was_training:
            model.train()
        wandb.log({**m, "train/global_step": state.global_step}, step=state.global_step)
        print(
            f"[bench @ {state.global_step}] "
            + " | ".join(f"{k.split('/')[1]} {v:.4f}" for k, v in m.items())
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    set_seed(SEED)
    tokenizer = load_tokenizer()

    model = build_model(warm_start=WARM_START)

    train_ds = build_train_dataset(tokenizer)
    wikitext_ds, nemotron_ds, mc_tasks = build_eval_corpora(tokenizer)

    # wandb: one resumable run per OUTPUT_DIR. The Qwen scripts stash wandb_run_id
    # in the checkpoint; HF Trainer checkpoints don't carry custom keys, so persist
    # it next to them instead. Init before the Trainer so its WandbCallback reuses
    # this run (and just merges TrainingArguments + model config into it).
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    id_path = os.path.join(OUTPUT_DIR, "wandb_run_id.txt")
    resuming_run = args.resume and os.path.exists(id_path)
    if resuming_run:
        wandb_run_id = open(id_path).read().strip()
    else:
        wandb_run_id = wandb.util.generate_id()
        with open(id_path, "w") as f:
            f.write(wandb_run_id)

    run_hparams = {
        "ref_model": REF_MODEL,
        "dataset": DATASET,
        "dataset_config": DATASET_CONFIG,
        "context_len": CONTEXT_LEN,
        "num_layers": NUM_LAYERS,
        "warm_start": WARM_START,
        "micro_batch": MICRO_BATCH,
        "grad_accum": GRAD_ACCUM,
        "effective_tokens_per_step": MICRO_BATCH * GRAD_ACCUM * CONTEXT_LEN,
        "max_steps": MAX_STEPS,
        "peak_lr": PEAK_LR,
        "lr_schedule": "wsd",
        "warmup_steps": WARMUP_STEPS,
        "decay_steps": DECAY_STEPS,
        "min_lr_ratio": MIN_LR_RATIO,
    }
    wandb.init(
        project=WANDB_PROJECT,
        config=run_hparams,
        id=wandb_run_id,
        resume="must" if resuming_run else "allow",
        name=f"gemma4-{NUM_LAYERS}L-{MAX_STEPS}steps",
    )

    # plot every EVAL_STEPS benchmark metric against the optimizer step
    wandb.define_metric("train/global_step")
    wandb.define_metric("eval/*", step_metric="train/global_step")

    targs = TrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,                      # required: streaming dataset has no length
        per_device_train_batch_size=MICRO_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=PEAK_LR,
        lr_scheduler_type="warmup_stable_decay",  # WSD: warmup -> hold at PEAK_LR -> cosine decay
        lr_scheduler_kwargs={
            "num_decay_steps": DECAY_STEPS,        # final phase; stable phase fills the gap to MAX_STEPS
            "min_lr_ratio": MIN_LR_RATIO,         # decay floor = MIN_LR_RATIO * PEAK_LR
        },
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,                                # set False if training on CPU
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=500,
        save_total_limit=10,
        dataloader_num_workers=0,                # >0 duplicates the whole stream per worker
        dataloader_pin_memory=False,             # page-locked host RAM, no gain for tiny streaming batches
        report_to=["wandb"],                     # matches the Qwen pretrain scripts
        run_name=f"gemma4-{NUM_LAYERS}L-{MAX_STEPS}steps",
        ignore_data_skip=True,                    # streaming: don't replay batches on resume
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=default_data_collator,      # labels already set in pack()
    )

    # slot TokensSeenCallback right before WandbCallback so it can inject
    # tokens_seen into `logs` before wandb reads it (user callbacks are otherwise
    # appended after the integration callbacks).
    cbs = trainer.callback_handler.callbacks
    w_idx = next(i for i, c in enumerate(cbs) if isinstance(c, WandbCallback))
    cbs.insert(w_idx, TokensSeenCallback(CONTEXT_LEN))

    trainer.add_callback(BenchmarkCallback(tokenizer, wikitext_ds, nemotron_ds, mc_tasks))

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    wandb.finish()


if __name__ == "__main__":
    main()
