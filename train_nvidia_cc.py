"""Pretrain the ~0.95B text-only Gemma-4 model (see train_hf.py) on nvidia/Nemotron-CC-v2
via HF Trainer.

Run from inside gemma/:
    python train_fineweb.py            # fresh run
    python train_fineweb.py --resume   # continue from ./checkpoints-fineweb

Needs HF_TOKEN (with access to the gated nvidia/Nemotron-CC-v2 dataset) in ../.env or .env.
"""

import argparse
import gc
import math
import os
from itertools import chain, islice

import torch
import wandb
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from safetensors import safe_open

load_dotenv()  # HF_TOKEN for the gated dataset + tokenizer/weights; WANDB_API_KEY for logging
HF_TOKEN = os.environ.get("HF_TOKEN")
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.integrations import WandbCallback

# --------------------------------------------------------------------------- #
# Knobs
# --------------------------------------------------------------------------- #
REF_MODEL      = "google/gemma-4-E2B"       # tokenizer + warm-start weights
DATASET        = "nvidia/Nemotron-CC-v2"    # gated; needs HF_TOKEN access
DATASET_CONFIG = "High-Quality"             # or High-Quality-Synthetic / Medium-High-Quality
                                            # / Medium-Quality / Diverse-QA / Translated-Diverse-QA
CONTEXT_LEN    = 2048
SEED           = 1234

OUTPUT_DIR     = "./checkpoints-fineweb"
WARM_START     = True                       # init from E2B text weights (100% at 15 layers)
NUM_LAYERS     = 15

MICRO_BATCH    = 10                          # per-device
GRAD_ACCUM     = 4                         # -> effective 8*16*2048 = 262k tokens/step (1 GPU)
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
#   1. WikiText perplexity        (held-out public corpus; the wikitext-2 val set
#      is only ~139 packed seqs, so we probe a larger slice of the
#      wikitext-103-raw-v1 *train* split instead -- still disjoint from the
#      Nemotron-CC training data)
#   2. Nemotron-CC-v2 val loss/ppl (VAL_DOCS docs skipped from the head of the
#      train stream, so train and val are disjoint)
#   3. HellaSwag acc / acc_norm    (length-normalised loglikelihood; HELLASWAG_N
#      examples of the val split, or None for the full 10 042-example split)
EVAL_STEPS     = 1_000
EVAL_MAX_SEQS  = 200                        # packed CONTEXT_LEN-token seqs, Nemotron held-out
WIKITEXT_MAX_SEQS = 600                     # packed CONTEXT_LEN-token seqs, WikiText probe (~1.2M tok)
EVAL_BATCH     = 8                          # seqs per forward for the LM-loss evals
VAL_DOCS       = 2_000
HELLASWAG_N    = None                       # None -> full validation split (10 042 examples)
HELLASWAG_BATCH = 32                        # (example, ending) rows per forward


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
# Benchmark eval
# --------------------------------------------------------------------------- #
def _pack_texts(tokenizer, texts, max_seqs):
    """Concatenate `texts`, tokenize, and cut into non-overlapping CONTEXT_LEN
    blocks (same packing as the training loader). Returns a Dataset of input_ids."""
    ids = []
    for t in texts:
        if not t or not t.strip():
            continue
        ids.extend(tokenizer(t, truncation=True, max_length=2 * CONTEXT_LEN)["input_ids"])
        if len(ids) >= (max_seqs + 1) * CONTEXT_LEN:
            break
    n = (len(ids) // CONTEXT_LEN) * CONTEXT_LEN
    chunks = [ids[i : i + CONTEXT_LEN] for i in range(0, n, CONTEXT_LEN)][:max_seqs]
    return Dataset.from_dict({"input_ids": chunks})


def build_eval_corpora(tokenizer):
    """WikiText + held-out Nemotron packed LM sets, and the HellaSwag examples."""
    wt = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
    )
    wikitext_ds = _pack_texts(
        tokenizer, (ex["text"] for ex in wt), WIKITEXT_MAX_SEQS
    )

    raw = load_dataset(
        DATASET, name=DATASET_CONFIG, split="train", streaming=True, token=HF_TOKEN
    )
    held = [ex["text"] for ex in islice(raw, VAL_DOCS)]
    nemotron_ds = _pack_texts(tokenizer, held, EVAL_MAX_SEQS)

    hs = load_dataset("Rowan/hellaswag", split="validation")
    if HELLASWAG_N:
        hs = hs.select(range(min(HELLASWAG_N, len(hs))))
    hellaswag = [
        {"ctx": e["ctx"], "endings": e["endings"], "label": int(e["label"])}
        for e in hs
        if e["label"] != ""
    ]
    print(
        f"eval corpora: wikitext {len(wikitext_ds)} seqs | "
        f"nemotron-heldout {len(nemotron_ds)} seqs | hellaswag {len(hellaswag)} ex"
    )
    return wikitext_ds, nemotron_ds, hellaswag


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
def _hellaswag_acc(model, tokenizer, examples, device, batch_size):
    """Length-normalised loglikelihood scoring -> (acc, acc_norm)."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    rows = []                                    # (ex_idx, ending_idx, ctx_len, full_ids)
    for ei, ex in enumerate(examples):
        ctx_ids = tokenizer(ex["ctx"])["input_ids"]
        for k, end in enumerate(ex["endings"]):
            full_ids = tokenizer(ex["ctx"] + " " + end)["input_ids"]
            rows.append((ei, k, len(ctx_ids), full_ids))

    ll = torch.full((len(examples), 4), float("-inf"))
    ll_norm = torch.full((len(examples), 4), float("-inf"))
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
    labels = torch.tensor([ex["label"] for ex in examples])
    return (
        (ll.argmax(-1) == labels).float().mean().item(),
        (ll_norm.argmax(-1) == labels).float().mean().item(),
    )


class BenchmarkCallback(TrainerCallback):
    """Every EVAL_STEPS steps: WikiText + Nemotron perplexity and HellaSwag
    accuracy, logged to wandb against train/global_step (rank 0, blocking)."""

    def __init__(self, tokenizer, wikitext_ds, nemotron_ds, hellaswag):
        self.tok = tokenizer
        self.wikitext_ds = wikitext_ds
        self.nemotron_ds = nemotron_ds
        self.hellaswag = hellaswag

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
        acc, acc_norm = _hellaswag_acc(model, self.tok, self.hellaswag, args.device, HELLASWAG_BATCH)
        m["eval/hellaswag_acc"] = acc
        m["eval/hellaswag_acc_norm"] = acc_norm
        if was_training:
            model.train()
        wandb.log({**m, "train/global_step": state.global_step}, step=state.global_step)
        print(
            f"[bench @ {state.global_step}] "
            + " | ".join(f"{k.split('/')[1]} {v:.4f}" for k, v in m.items())
        )


def build_config():
    d = {
        "model_type": "gemma4_text",
        "vocab_size": 262144,
        "hidden_size": 1536,
        "intermediate_size": 6144,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "hidden_size_per_layer_input": 0,          # no Per-Layer Embeddings
        "max_position_embeddings": max(CONTEXT_LEN, 8192),
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "attention_dropout": 0.0,
        "layer_types": [
            "full_attention" if (i + 1) % 5 == 0 else "sliding_attention"
            for i in range(NUM_LAYERS)
        ],
    }
    return AutoConfig.for_model(**d)


def warm_start_from_e2b(model):
    own = model.state_dict()
    loadable = {}
    path = hf_hub_download(REF_MODEL, "model.safetensors", token=HF_TOKEN)
    with safe_open(path, framework="pt") as f:
        for src_key in f.keys():
            if "language_model" not in src_key:
                continue
            key = src_key.replace("model.language_model.", "model.", 1)
            if key in own and tuple(f.get_slice(src_key).get_shape()) == tuple(own[key].shape):
                loadable[key] = f.get_tensor(src_key).to(own[key].dtype)
    model.load_state_dict(loadable, strict=False)
    model.tie_weights()
    n = len(loadable)
    del loadable
    gc.collect()
    return n


def build_dataset(tokenizer):
    ds = load_dataset(
        DATASET, name=DATASET_CONFIG, split="train", streaming=True, token=HF_TOKEN
    )
    ds = ds.skip(VAL_DOCS)                         # reserve the head for the Nemotron held-out eval
    ds = ds.shuffle(seed=SEED, buffer_size=100)   # prefilled into RAM before the first batch; keep small
    cols = list(ds.features)

    def tokenize(batch):
        # adds BOS per doc; cap long docs so the packing buffers below stay small
        return {"input_ids": tokenizer(batch["text"], truncation=True, max_length=2 * CONTEXT_LEN)["input_ids"]}

    def pack(batch):
        ids = list(chain.from_iterable(batch["input_ids"]))
        n = (len(ids) // CONTEXT_LEN) * CONTEXT_LEN
        chunks = [ids[i : i + CONTEXT_LEN] for i in range(0, n, CONTEXT_LEN)]
        return {"input_ids": chunks, "labels": [c[:] for c in chunks]}

    # small map batches: default 1000 makes pack() build ~16M-int Python lists on
    # the main process and thrashes host RAM. 128 keeps the transient spike tiny.
    ds = ds.map(tokenize, batched=True, batch_size=128, remove_columns=cols)
    ds = ds.map(pack, batched=True, batch_size=128, remove_columns=["input_ids"])
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(REF_MODEL, token=HF_TOKEN)

    model = AutoModelForCausalLM.from_config(build_config())
    if WARM_START:
        print(f"warm-started {warm_start_from_e2b(model)} tensors from {REF_MODEL}")
    model.config.use_cache = False               # needed for gradient checkpointing

    train_ds = build_dataset(tokenizer)
    wikitext_ds, nemotron_ds, hellaswag = build_eval_corpora(tokenizer)

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

    trainer.add_callback(BenchmarkCallback(tokenizer, wikitext_ds, nemotron_ds, hellaswag))

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    wandb.finish()


if __name__ == "__main__":
    main()
