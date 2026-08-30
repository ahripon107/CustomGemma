"""Model preparation for the ~0.95B text-only Gemma-4.

Architectural surgery on ``google/gemma-4-E2B``: declare a 15-layer, PLE-free,
text-only ``gemma4_text`` config and warm-start every surviving tensor from the
E2B text tower (token embedding, final norm, and layers 0-14 transfer verbatim;
the Per-Layer-Embedding table and layers 15-34 are dropped).

Imported by the trainer:
    from model_prep import build_model, load_tokenizer
    model = build_model()                 # config + warm-start, ready for Trainer

Standalone, to materialise an init checkpoint for inspection:
    python model_prep.py --save-dir ./gemma4-15L-init

Needs HF_TOKEN (with access to google/gemma-4-E2B) in ../.env or .env.
"""

import argparse
import gc
import os

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

load_dotenv()  # HF_TOKEN for the gated tokenizer/weights
HF_TOKEN = os.environ.get("HF_TOKEN")

# --------------------------------------------------------------------------- #
# Knobs
# --------------------------------------------------------------------------- #
REF_MODEL   = "google/gemma-4-E2B"     # tokenizer + warm-start weights
CONTEXT_LEN = 2048                     # training seq len; also sets max_position_embeddings
NUM_LAYERS  = 15
WARM_START  = True                     # init from E2B text weights (100% at 15 layers)


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


def load_tokenizer():
    return AutoTokenizer.from_pretrained(REF_MODEL, token=HF_TOKEN)


def build_model(warm_start=WARM_START):
    """Instantiate the custom config and (optionally) warm-start from E2B.

    Returns a model with ``use_cache`` disabled, ready for gradient checkpointing.
    """
    model = AutoModelForCausalLM.from_config(build_config())
    if warm_start:
        print(f"warm-started {warm_start_from_e2b(model)} tensors from {REF_MODEL}")
    model.config.use_cache = False               # needed for gradient checkpointing
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", default="./gemma4-15L-init")
    ap.add_argument("--no-warm-start", action="store_true")
    args = ap.parse_args()

    model = build_model(warm_start=not args.no_warm_start)
    print(f"total parameters: {sum(p.numel() for p in model.parameters()):,}")

    model.config.use_cache = True                 # restore default for the saved config
    model.save_pretrained(args.save_dir)
    load_tokenizer().save_pretrained(args.save_dir)
    print(f"saved init model + tokenizer to {args.save_dir}")


if __name__ == "__main__":
    main()
