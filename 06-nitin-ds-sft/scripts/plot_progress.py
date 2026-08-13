#!/usr/bin/env python3
"""Regenerate training-progress PNGs from an mlx_lm train.log (and, optionally,
an eval-checkpoint pass-rate file). One shot: run it, it writes the PNGs, it
exits. The orchestrating session calls it on a loop; it does NOT loop itself.

    python plot_progress.py --train-log <path> --out <png> \
        [--eval-curve <jsonl|csv>] [--eval-out <png>] [--title <str>]

Log lines it understands (both real formats in this repo, ANSI-tolerant):

  mlx_lm.lora SFT/CPT:
    Iter 50: Train loss 2.712, Learning Rate 1.195e-06, It/sec 1.081, ...
    Iter 500: Val loss 1.317, Val took 3.003s
  dpo_fixed_train.py DPO:
    Iter 5: loss 0.724, chosen_r -0.064, ..., lr 1.000e-06, it/s 0.441, ...
    Iter 20: Val loss 0.662, Val chosen reward 0.000, ...

Runs here are segmented (watchdog relaunches mlx_lm per segment), so `Iter N`
restarts at 1 each segment. The `>>> ... (DONE/TOTAL done)` banner the run
scripts print before each segment gives the true global offset; steps are
de-duplicated last-write-wins so restarted-from-scratch attempts overwrite
rather than zig-zag.

Eval curve accepts either:
  * results/<stage>/metrics_functional.jsonl as written by eval_functional.jac
    ({"step","category","gate_class","total","runs","runs_pct"}) -> one line
    per category/gate_class plus a token-weighted OVERALL line, or
  * any JSONL/CSV of {"step": int, "pass_rate": float} -> a single line.
"""

import argparse
import csv
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*m")
SEGMENT = re.compile(r"^>>>.*?\((\d+)\s*/\s*\d+\s+done")
VAL = re.compile(r"^Iter\s+(\d+):\s+Val loss\s+([-\d.eE+]+)")
TRAIN = re.compile(r"^Iter\s+(\d+):\s+(?:Train loss|loss)\s+([-\d.eE+]+)")


def parse_train_log(path):
    """-> (train[(step, loss)], val[(step, loss)]), sorted, deduped."""
    train, val, offset = {}, {}, 0
    with open(path, "r", errors="replace") as fh:
        for raw in fh:
            line = ANSI.sub("", raw).strip()
            seg = SEGMENT.match(line)
            if seg:
                offset = int(seg.group(1))
                continue
            m = VAL.match(line)
            if m:
                val[offset + int(m.group(1))] = float(m.group(2))
                continue
            m = TRAIN.match(line)
            if m:
                train[offset + int(m.group(1))] = float(m.group(2))
    return sorted(train.items()), sorted(val.items())


def plot_loss(train, val, out, title):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if train:
        ax.plot(*zip(*train), lw=1.2, color="#2563eb", label="train loss")
    if val:
        ax.plot(*zip(*val), lw=1.6, marker="o", ms=4, color="#dc2626",
                label="val loss")
    ys = [y for _, y in train + val if y > 0]
    if ys and max(ys) / min(ys) > 20:
        ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if train or val:
        ax.legend()
    _save(fig, out)


def load_eval_curve(path):
    """-> {series_name: [(step, pct), ...]}"""
    rows = []
    if path.endswith(".csv"):
        with open(path) as fh:
            rows = [dict(r) for r in csv.DictReader(fh)]
    else:
        with open(path) as fh:
            rows = [json.loads(ln) for ln in fh if ln.strip().startswith("{")]
    if not rows:
        return {}

    series, totals = {}, {}
    for r in rows:
        step = int(float(r["step"]))
        if "pass_rate" in r:                       # simple {step, pass_rate}
            series.setdefault("pass rate", []).append((step, float(r["pass_rate"])))
            continue
        name = f"{r.get('category', '?')}/{r.get('gate_class', '?')}"
        series.setdefault(name, []).append((step, float(r["runs_pct"])))
        runs, total = totals.setdefault(step, [0.0, 0.0])
        totals[step] = [runs + float(r["runs"]), total + float(r["total"])]
    for step, (runs, total) in sorted(totals.items()):
        if total:
            series.setdefault("OVERALL", []).append((step, 100.0 * runs / total))
    return {k: sorted(v) for k, v in series.items()}


def plot_eval(series, out, title):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, pts in series.items():
        overall = name == "OVERALL"
        ax.plot(*zip(*pts), marker="o", ms=4, label=name,
                lw=2.5 if overall else 1.2,
                color="black" if overall else None,
                alpha=1.0 if overall else 0.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("pass rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if series:
        ax.legend(fontsize=8)
    _save(fig, out)


def _save(fig, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-log", help="path to train.log (or a dir containing one)")
    ap.add_argument("--out", help="output PNG for the loss curve")
    ap.add_argument("--eval-curve", help="metrics_functional.jsonl or {step,pass_rate} jsonl/csv")
    ap.add_argument("--eval-out", help="output PNG for the eval pass-rate curve")
    ap.add_argument("--title", default=None, help="plot title prefix")
    args = ap.parse_args()

    log = args.train_log
    if log and os.path.isdir(log):
        log = os.path.join(log, "train.log")
    label = args.title or (os.path.basename(os.path.dirname(os.path.abspath(log)))
                           if log else "run")

    if log and args.out:
        if not os.path.exists(log):
            print(f"[plot_progress] no train.log yet at {log}", file=sys.stderr)
        else:
            train, val = parse_train_log(log)
            if not train and not val:
                print(f"[plot_progress] no Iter lines yet in {log}", file=sys.stderr)
            else:
                plot_loss(train, val, args.out, f"{label} — loss")
                last = train[-1] if train else val[-1]
                print(f"[plot_progress] {args.out}: {len(train)} train / {len(val)} val "
                      f"points, last iter {last[0]} loss {last[1]:.4f}")

    if args.eval_curve and args.eval_out:
        if not os.path.exists(args.eval_curve):
            print(f"[plot_progress] no eval curve yet at {args.eval_curve}", file=sys.stderr)
        else:
            series = load_eval_curve(args.eval_curve)
            if not series:
                print(f"[plot_progress] eval curve {args.eval_curve} is empty", file=sys.stderr)
            else:
                plot_eval(series, args.eval_out, f"{label} — eval pass rate")
                n = sum(len(v) for v in series.values())
                print(f"[plot_progress] {args.eval_out}: {len(series)} series, {n} points")


if __name__ == "__main__":
    main()
