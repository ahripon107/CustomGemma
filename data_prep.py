"""Dataset preparation for the nvidia/Nemotron-CC-v2 pretraining run.

Provides two builders the trainer imports:

  * ``build_train_dataset(tokenizer)`` -> streaming ``IterableDataset`` of packed,
    non-overlapping ``CONTEXT_LEN``-token blocks (one BOS per doc, ``labels`` set).
    The first ``VAL_DOCS`` docs are ``.skip()``-ped so train / val stay disjoint.

  * ``build_eval_corpora(tokenizer)`` -> ``(wikitext_ds, nemotron_ds, hellaswag)``
    for the in-loop benchmark suite:
      - WikiText  : the wikitext-2 val set is only ~139 packed blocks, so we probe
        a larger slice of the wikitext-103-raw-v1 *train* split instead
        (``WIKITEXT_MAX_SEQS`` blocks, ~1.2M tokens), still disjoint from Nemotron.
      - Nemotron  : the ``VAL_DOCS`` skipped docs, packed to ``EVAL_MAX_SEQS`` blocks
        (a true in-distribution val signal).
      - HellaSwag : ``HELLASWAG_N`` val examples, or None for the full 10 042.

Standalone, to sanity-check the pipeline and optionally cache the (finite) eval
corpora to disk:
    python data_prep.py --peek
    python data_prep.py --eval-cache ./eval-corpora

Needs HF_TOKEN (with access to the gated nvidia/Nemotron-CC-v2 dataset) in
../.env or .env.
"""

import argparse
import os
from itertools import chain, islice

from datasets import Dataset, load_dataset
from dotenv import load_dotenv

from model_prep import CONTEXT_LEN, load_tokenizer

load_dotenv()  # HF_TOKEN for the gated dataset
HF_TOKEN = os.environ.get("HF_TOKEN")

# --------------------------------------------------------------------------- #
# Knobs
# --------------------------------------------------------------------------- #
DATASET        = "nvidia/Nemotron-CC-v2"    # gated; needs HF_TOKEN access
DATASET_CONFIG = "High-Quality"             # or High-Quality-Synthetic / Medium-High-Quality
                                            # / Medium-Quality / Diverse-QA / Translated-Diverse-QA
SEED           = 1234
VAL_DOCS       = 2_000                      # head of the train stream reserved for the Nemotron eval

# LM-perplexity corpora sizes (packed CONTEXT_LEN-token seqs)
EVAL_MAX_SEQS     = 200                     # Nemotron held-out
WIKITEXT_MAX_SEQS = 600                     # WikiText probe (~1.2M tok)

# multiple-choice benchmark sizes -- None means the whole split. MMLU is the big
# one (14 042 examples x 4 choices); set MMLU_N to subsample (shuffled, so every
# subject is represented) if the in-loop eval gets too slow.
HELLASWAG_N      = None                     # validation, 10 042 ex
PIQA_N           = None                     # validation, 1 838 ex
ARC_N            = None                     # ARC-Challenge test, 1 172 ex
WINOGRANDE_N     = None                     # winogrande_xl validation, 1 267 ex
MMLU_N           = None                     # test, 14 042 ex


# --------------------------------------------------------------------------- #
# Training stream
# --------------------------------------------------------------------------- #
def build_train_dataset(tokenizer):
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


# --------------------------------------------------------------------------- #
# Benchmark eval corpora
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


# --------------------------------------------------------------------------- #
# Multiple-choice benchmarks
#
# Each builder returns a list of {"pairs": [(context, continuation), ...],
# "gold": int}. The trainer scores the continuation tokens of every pair (summed
# and length-normalised loglikelihood) and takes the argmax -- see _mc_acc in
# train_nvidia_cc.py. Prompt formatting follows lm-eval-harness (0-shot).
# --------------------------------------------------------------------------- #
def _subsample(ds, n):
    if n and n < len(ds):
        ds = ds.shuffle(seed=SEED).select(range(n))
    return ds


def _hellaswag_examples():
    ds = _subsample(load_dataset("Rowan/hellaswag", split="validation"), HELLASWAG_N)
    return [
        {"pairs": [(e["ctx"], " " + end) for end in e["endings"]], "gold": int(e["label"])}
        for e in ds
        if e["label"] != ""
    ]


def _piqa_examples():
    ds = _subsample(
        load_dataset("ybisk/piqa", split="validation", revision="refs/convert/parquet"),
        PIQA_N,
    )
    out = []
    for e in ds:
        if e["label"] not in (0, 1):
            continue
        ctx = f"Question: {e['goal']}\nAnswer:"
        out.append({"pairs": [(ctx, " " + e["sol1"]), (ctx, " " + e["sol2"])], "gold": e["label"]})
    return out


def _arc_examples():
    ds = _subsample(
        load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"), ARC_N
    )
    out = []
    for e in ds:
        labels, texts = e["choices"]["label"], e["choices"]["text"]
        if e["answerKey"] not in labels:
            continue
        ctx = f"Question: {e['question']}\nAnswer:"
        out.append({"pairs": [(ctx, " " + t) for t in texts], "gold": labels.index(e["answerKey"])})
    return out


def _winogrande_examples():
    # official allenai/winogrande is script-only (unsupported on datasets 5.x) and
    # its parquet export merges every size config; this mirror is the plain xl set.
    ds = _subsample(
        load_dataset("coref-data/winogrande_raw", "winogrande_xl", split="validation"),
        WINOGRANDE_N,
    )
    out = []
    for e in ds:
        idx = e["sentence"].index("_")
        prefix, suffix = e["sentence"][:idx], e["sentence"][idx + 1 :].strip()
        pairs = [(prefix + e["option1"], " " + suffix), (prefix + e["option2"], " " + suffix)]
        out.append({"pairs": pairs, "gold": int(e["answer"]) - 1})
    return out


def _mmlu_examples():
    ds = _subsample(load_dataset("cais/mmlu", "all", split="test"), MMLU_N)
    letters = ["A", "B", "C", "D"]
    out = []
    for e in ds:
        subject = e["subject"].replace("_", " ")
        opts = "\n".join(f"{l}. {c}" for l, c in zip(letters, e["choices"]))
        ctx = (
            f"The following are multiple choice questions (with answers) about {subject}.\n\n"
            f"{e['question'].strip()}\n{opts}\nAnswer:"
        )
        out.append({"pairs": [(ctx, f" {l}") for l in letters], "gold": int(e["answer"])})
    return out


MC_BUILDERS = {
    "hellaswag": _hellaswag_examples,
    "piqa": _piqa_examples,
    "arc_challenge": _arc_examples,
    "winogrande": _winogrande_examples,
    "mmlu": _mmlu_examples,
}


def build_eval_corpora(tokenizer):
    """Returns (wikitext_ds, nemotron_ds, mc_tasks).

    wikitext_ds / nemotron_ds are packed LM sets for perplexity; mc_tasks is
    {task_name: [example, ...]} for the multiple-choice accuracy benchmarks
    (HellaSwag, PIQA, ARC-Challenge, WinoGrande, MMLU).
    """
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

    mc_tasks = {name: build() for name, build in MC_BUILDERS.items()}

    print(
        f"eval corpora: wikitext {len(wikitext_ds)} seqs | "
        f"nemotron-heldout {len(nemotron_ds)} seqs | "
        + " | ".join(f"{k} {len(v)} ex" for k, v in mc_tasks.items())
    )
    return wikitext_ds, nemotron_ds, mc_tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval-cache", default=None,
        help="dir to save_to_disk the wikitext + nemotron packed eval sets",
    )
    ap.add_argument(
        "--peek", action="store_true",
        help="pull one packed training example and print its length",
    )
    args = ap.parse_args()

    tokenizer = load_tokenizer()
    wikitext_ds, nemotron_ds, _ = build_eval_corpora(tokenizer)

    if args.eval_cache:
        wikitext_ds.save_to_disk(os.path.join(args.eval_cache, "wikitext"))
        nemotron_ds.save_to_disk(os.path.join(args.eval_cache, "nemotron"))
        print(f"cached eval corpora to {args.eval_cache}")

    if args.peek:
        ex = next(iter(build_train_dataset(tokenizer)))
        print(f"sample packed training example: {len(ex['input_ids'])} tokens")


if __name__ == "__main__":
    main()
