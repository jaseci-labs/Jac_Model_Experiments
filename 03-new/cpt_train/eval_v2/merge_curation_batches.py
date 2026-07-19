"""Merges per-batch Fable curation outputs (one JSON file per Agent-tool
call, written by the operational step of Task 7) into one curation.json,
schema-validating every batch first -- a batch that fails validation aborts
the merge loudly rather than silently poisoning the corpus."""
import json
from pathlib import Path

from schemas import validate_curation_batch


def _verdict_key(entry: dict) -> dict:
    """The fields that actually matter for correctness when comparing two
    curation entries for the same chunk_id. Free-text `reason` is
    deliberately excluded: two batches genuinely agreeing on a verdict
    (and weight, for upweight) should never be flagged as conflicting just
    because Fable phrased the justification differently on a retry."""
    key = {"verdict": entry["verdict"]}
    if entry["verdict"] == "upweight":
        key["weight"] = entry.get("weight", 2.0)
    return key


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
            if cid in merged and _verdict_key(merged[cid]) != _verdict_key(entry):
                raise ValueError(f"conflicting verdicts for chunk_id {cid}: "
                                  f"{merged[cid]} vs {entry} (from {path})")
            # Same verdict (and weight, if upweight) as any prior entry for
            # this chunk_id -- not a conflict. Keep the latest-seen reason.
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
