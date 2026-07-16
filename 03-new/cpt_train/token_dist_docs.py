"""Token-count distribution for one CPT source's raw.jsonl.

Usage: python token_dist_docs.py [path/to/raw.jsonl]
Default target: 03-new/dataset/cpt/docs/raw.jsonl
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_PATH = REPO_ROOT / "models" / "qwen-q4"
DEFAULT_JSONL = REPO_ROOT / "03-new" / "dataset" / "cpt" / "docs" / "raw.jsonl"


def main():
    jsonl_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSONL
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    counts = []
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            counts.append(len(tokenizer.encode(row["text"])))

    n = len(counts)
    total = sum(counts)
    avg = total / n
    counts_sorted = sorted(counts)
    median = counts_sorted[n // 2]
    p95 = counts_sorted[int(n * 0.95)]

    print(f"file: {jsonl_path}")
    print(f"rows: {n}")
    print(f"total tokens: {total}")
    print(f"avg tokens/row: {avg:.1f}")
    print(f"median tokens/row: {median}")
    print(f"min/max: {min(counts)}/{max(counts)}")
    print(f"p95: {p95}")

    out_png = jsonl_path.parent / "token_dist.png"
    plt.figure(figsize=(8, 5))
    plt.hist(counts, bins=60, color="#4C72B0", edgecolor="white")
    plt.axvline(avg, color="red", linestyle="--", label=f"mean={avg:.0f}")
    plt.axvline(median, color="green", linestyle="--", label=f"median={median}")
    plt.xlabel("tokens per row")
    plt.ylabel("row count")
    plt.title(f"Token count distribution — {jsonl_path.name} (n={n})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"saved: {out_png}")


if __name__ == "__main__":
    main()
