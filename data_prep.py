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

# eval-corpora sizes (packed CONTEXT_LEN-token seqs, except HELLASWAG_N)
EVAL_MAX_SEQS     = 200                     # Nemotron held-out
WIKITEXT_MAX_SEQS = 600                     # WikiText probe (~1.2M tok)
HELLASWAG_N       = None                    # None -> full validation split (10 042 examples)


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
