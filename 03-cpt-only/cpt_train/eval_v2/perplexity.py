"""Held-out perplexity: base vs cpt-v1 vs cpt-v2 on the CPT-v2 validation
split (03-new/dataset/cpt-v2/packed/valid.jsonl, 102 windows, never trained
on). Same held-out set for all three models so numbers are comparable.
Reuses mlx_lm's own eval loss (cross-entropy) -> perplexity = exp(loss).
"""
import gc
import json
import math
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.tuner.datasets import CacheDataset, load_local_dataset
from mlx_lm.tuner.trainer import evaluate
from mlx_lm.utils import load

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "03-new" / "dataset" / "cpt-v2" / "packed"
OUT = ROOT / "03-new" / "results" / "cpt-v2" / "json" / "perplexity.json"

MODELS = {
    "base": ROOT / "models" / "qwen-q4",
    "cpt_v1": ROOT / "models" / "qwen-cpt-v1-fused-q4",
    "cpt_v2": ROOT / "models" / "qwen-cpt-v2-fused-q4",
}

MAX_SEQ_LENGTH = 4096
BATCH_SIZE = 1


def main():
    results = {}
    for name, path in MODELS.items():
        print(f"=== {name}: loading {path} ===")
        model, tokenizer = load(str(path))
        _, valid_set, _ = load_local_dataset(DATA_DIR, tokenizer, SimpleNamespace())
        valid_set = CacheDataset(valid_set)
        print(f"    valid windows: {len(valid_set)}")
        loss = evaluate(
            model=model,
            dataset=valid_set,
            batch_size=BATCH_SIZE,
            num_batches=-1,
            max_seq_length=MAX_SEQ_LENGTH,
        )
        ppl = math.exp(loss)
        print(f"    {name}: loss={loss:.4f} ppl={ppl:.4f}")
        results[name] = {"loss": loss, "perplexity": ppl}
        del model, tokenizer
        gc.collect()
        mx.clear_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
