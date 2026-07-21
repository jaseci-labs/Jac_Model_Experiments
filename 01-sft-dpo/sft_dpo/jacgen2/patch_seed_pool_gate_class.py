"""
patch_seed_pool_gate_class.py -- ONE-TIME data patch, not a reusable module.

Fixes seed_pool.jsonl's gate_class bug (Task 9's investigation, see the
04-cpt-sft production dispatch notes): 813 of the pool's 1100
gate_class="behavioral" records have expected_output empty/null/missing,
meaning there's no real seed output to behaviorally re-verify a bug-injected
variant against. gen_debug's/gen_dpo's behavioral-gate logic does a
truthiness check on expected_output, so these records were silently treated
as "no divergence expected" -- producing most of the pilot's spurious
rejections (11/21 of Task 9's debug rejections were exactly this).

This script flips gate_class from "behavioral" to "compile_only" for every
record where gate_class == "behavioral" AND expected_output is None or
whitespace-only. All other fields, and record order, are preserved exactly.
Records are re-serialized with json.dumps(..., ensure_ascii=False) using
default key order (Python dicts preserve insertion order from json.loads,
so field order round-trips unchanged).

This intentionally invalidates the Task 7 freeze recorded in pool_hashes.json
(seed_pool.jsonl's hash will now mismatch) -- that's expected. freeze_pool.jac
is re-run later, separately, at the end of the full production run; this
script does not touch pool_hashes.json and does not invoke freeze_pool.jac.

Run once from repo root:
    /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/python \
        model-experiments/01-sft-dpo/sft_dpo/jacgen2/patch_seed_pool_gate_class.py
"""

import json

SEED_POOL_PATH = "model-experiments/04-cpt-sft/dataset/shared/seed_pool.jsonl"


def is_blank(expected_output) -> bool:
    if expected_output is None:
        return True
    if isinstance(expected_output, str) and expected_output.strip() == "":
        return True
    return False


def count_gate_classes(records: list[dict]) -> dict:
    counts: dict = {}
    for rec in records:
        gc = rec.get("gate_class")
        counts[gc] = counts.get(gc, 0) + 1
    return counts


def main() -> None:
    with open(SEED_POOL_PATH, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    records = [json.loads(line) for line in lines]

    before_counts = count_gate_classes(records)

    flipped = 0
    for rec in records:
        if rec.get("gate_class") == "behavioral" and is_blank(rec.get("expected_output")):
            rec["gate_class"] = "compile_only"
            flipped += 1

    after_counts = count_gate_classes(records)

    with open(SEED_POOL_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("=== seed_pool.jsonl gate_class patch ===")
    print(f"total records: {len(records)}")
    print(f"before: behavioral={before_counts.get('behavioral', 0)} "
          f"compile_only={before_counts.get('compile_only', 0)}")
    print(f"flipped (behavioral -> compile_only, blank expected_output): {flipped}")
    print(f"after:  behavioral={after_counts.get('behavioral', 0)} "
          f"compile_only={after_counts.get('compile_only', 0)}")
    print(f"wrote {SEED_POOL_PATH} in place")


if __name__ == "__main__":
    main()
