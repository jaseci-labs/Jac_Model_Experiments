"""Merges per-batch Fable curation outputs (one JSON file per Agent-tool
call, written by the operational step of Task 7) into one curation.json,
schema-validating every batch first -- a batch that fails validation aborts
the merge loudly rather than silently poisoning the corpus."""
import json
from pathlib import Path

from schemas import validate_curation_batch


def merge_batches(batch_files: list) -> dict:
    merged = {}
    for path in batch_files:
        batch = json.loads(Path(path).read_text())
        errors = validate_curation_batch(batch)
        if errors:
            raise ValueError(f"{path}: {'; '.join(errors)}")
        for item in batch:
            cid = item["chunk_id"]
            entry = {"verdict": item["verdict"], "reason": item["reason"]}
            if item["verdict"] == "upweight":
                entry["weight"] = item.get("weight", 2.0)
            if cid in merged and merged[cid] != entry:
                raise ValueError(f"conflicting verdicts for chunk_id {cid}: "
                                  f"{merged[cid]} vs {entry} (from {path})")
            merged[cid] = entry
    return merged


def main():
    import argparse
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_files", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    merged = merge_batches(args.batch_files)
    args.out.write_text(json.dumps(merged, indent=2))
    print(f"merged {len(args.batch_files)} batches -> {len(merged)} verdicts -> {args.out}")


if __name__ == "__main__":
    main()
