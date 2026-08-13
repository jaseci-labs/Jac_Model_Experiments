#!/usr/bin/env python3
"""Sanity checks for the Arm 5 (CPT-on-Spectrum) chained run.

Every check here exists because this repo has actually been bitten by the thing
it checks:

  * ``adapter``    -- the "BROKEN-zeroweights" incident
    (``sft_cptv2_probe/results/sft-spectrum-BROKEN-zeroweights/``): an
    in-place ``mx.load`` + overwrite truncated the lazy mmap and silently
    zeroed every trained LoRA key.  A file of the right *size* with the right
    *key count* was still garbage.  So: materialise with ``mx.eval`` first,
    then require every tensor to be finite and at least one nonzero element
    per tensor.
  * ``state``      -- CPT epoch-loop completion, leg count, NaN/Inf losses.
  * ``metrics``    -- an eval stage that "ran" but wrote no full-holdout row.

Read-only.  Never writes to any path it reads.
"""
import argparse
import json
import sys
from pathlib import Path

EXPECTED_KEYS = 256
PICKS = [0, 22, 23, 27, 30, 34, 36, 37, 38, 39, 41, 42, 43, 44, 45, 47]


def fail(msg: str) -> None:
    print(f"CHECK-FAIL: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"CHECK-OK: {msg}")


def check_adapter(path: str, min_bytes: int, expect_blocks: bool) -> None:
    import mlx.core as mx

    p = Path(path)
    if not p.exists():
        fail(f"adapter file does not exist: {p}")
    size = p.stat().st_size
    if size < min_bytes:
        fail(f"adapter file suspiciously small: {p} is {size} bytes (< {min_bytes})")

    w = dict(mx.load(str(p)))
    # Materialise BEFORE inspecting -- mx.load is a lazy mmap (lesson 1).
    mx.eval(list(w.values()))
    if not w:
        fail(f"adapter file holds zero tensors: {p}")

    zeros, nonfinite = [], []
    for k, v in w.items():
        amax = float(mx.max(mx.abs(v)).item())
        if amax != amax or amax == float("inf"):  # NaN or Inf
            nonfinite.append(k)
        elif amax == 0.0:
            zeros.append(k)
    if nonfinite:
        fail(f"{len(nonfinite)}/{len(w)} tensors contain NaN/Inf: {nonfinite[:5]}")
    if zeros:
        fail(
            f"{len(zeros)}/{len(w)} tensors are ALL-ZERO in {p} -- this is the "
            f"BROKEN-zeroweights signature: {zeros[:5]}"
        )

    if expect_blocks:
        if len(w) != EXPECTED_KEYS:
            fail(f"{p} holds {len(w)} keys, expected {EXPECTED_KEYS}")
        # keys are "model.layers.<N>.<module>.lora_{a,b}" -> index [2], matching
        # run_cpt_leg_spectrum.py:56. (cpt-spectrum-plan.md §2.2a's illustrative
        # snippet says [1]; that snippet is wrong, the shipped driver is right.)
        blocks = sorted({int(k.split(".")[2]) for k in w if k.startswith("model.layers.")})
        if blocks != PICKS:
            fail(f"{p} covers blocks {blocks}, expected the Spectrum picks {PICKS}")
        ok(f"{p.name}: {size/1e9:.2f} GB, {len(w)} keys, blocks == picks, all finite, none all-zero")
    else:
        ok(f"{p.name}: {size/1e9:.2f} GB, {len(w)} keys, all finite, none all-zero")


def check_state(path: str, min_legs: int) -> None:
    p = Path(path)
    if not p.exists():
        fail(f"training_state.json does not exist: {p}")
    st = json.loads(p.read_text())
    legs = st.get("legs", [])
    status = st.get("status")
    if status not in ("halt_keep_this", "halt_keep_previous"):
        fail(f"epoch loop did not reach a halt decision: status={status!r}, {len(legs)} legs done")
    if len(legs) < min_legs:
        fail(f"only {len(legs)} legs recorded, floor is {min_legs}")
    for lg in legs:
        for key in ("train_loss", "val_loss"):
            v = lg.get(key)
            if v is None:
                fail(f"leg {lg.get('leg')} has no {key} -- log parse failed")
            if v != v or v in (float("inf"), float("-inf")):
                fail(f"leg {lg.get('leg')} {key} is not finite: {v}")
    cf = [lg.get("cf_score") for lg in legs]
    ok(f"{len(legs)} legs, status={status}, losses all finite, CF scores={cf}")
    # Which checkpoint is the accepted one.
    last = legs[-1]
    if status == "halt_keep_previous":
        accepted = last["done_steps_after"] - 544
    else:
        accepted = last["done_steps_after"]
    print(f"ACCEPTED_STEP={accepted}")


def check_metrics(path: str, min_full_rows: int) -> None:
    p = Path(path)
    if not p.exists():
        fail(f"metrics file does not exist: {p}")
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        if r.get("category") == "__overall__" and r.get("total") == 855:
            rows.append(r)
    if len(rows) < min_full_rows:
        fail(f"{p} has {len(rows)} full-holdout (n=855) overall rows, expected >= {min_full_rows}")
    for r in rows:
        print(f"  step={r.get('step')} runs={r.get('runs')}/855 pct={r.get('runs_pct')}")
    ok(f"{p.name}: {len(rows)} full-holdout rows present")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("adapter")
    a.add_argument("path")
    a.add_argument("--min-bytes", type=int, default=1_000_000_000)
    a.add_argument("--no-block-check", action="store_true")

    s = sub.add_parser("state")
    s.add_argument("path")
    s.add_argument("--min-legs", type=int, default=6)

    m = sub.add_parser("metrics")
    m.add_argument("path")
    m.add_argument("--min-full-rows", type=int, default=1)

    args = ap.parse_args()
    if args.cmd == "adapter":
        check_adapter(args.path, args.min_bytes, not args.no_block_check)
    elif args.cmd == "state":
        check_state(args.path, args.min_legs)
    elif args.cmd == "metrics":
        check_metrics(args.path, args.min_full_rows)


if __name__ == "__main__":
    main()
