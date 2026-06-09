"""Qwen-vs-Gemma comparison graphs -> results/comparison/*.png.

- Learning curves: BOTH models on one axis, monotone-cubic smoothed (real curves,
  not 6 dots; the 6 checkpoints are marked).
- Train + val loss: both models overlaid (dense, naturally curvy).
- Accuracy: grouped bars (function base/SFT/DPO + graph SFT/DPO).
- Throughput + graph-idiom (similarity + correct%).

    python3 srccurrent/make_comparison.py
"""
import json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

OUT = "results/comparison"
os.makedirs(OUT, exist_ok=True)
COL = {"qwen": "#1f77b4", "gemma": "#d62728"}
LBL = {"qwen": "Qwen3-Coder-30B-A3B", "gemma": "Gemma-4-26B-A4B"}


def pchip(x, y, xs):
    """Fritsch-Carlson monotone cubic Hermite — smooth, no overshoot."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    h = np.diff(x); d = np.diff(y) / h
    m = np.zeros_like(y)
    for i in range(1, len(y) - 1):
        if d[i - 1] * d[i] > 0:
            m[i] = 2.0 / (1.0 / d[i - 1] + 1.0 / d[i])  # harmonic mean
    m[0], m[-1] = d[0], d[-1]
    xs = np.asarray(xs, float); out = np.empty_like(xs)
    for i, xv in enumerate(xs):
        k = min(max(np.searchsorted(x, xv) - 1, 0), len(x) - 2)
        t = (xv - x[k]) / h[k]
        h00 = 2 * t**3 - 3 * t**2 + 1; h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2; h11 = t**3 - t**2
        out[i] = h00 * y[k] + h10 * h[k] * m[k] + h01 * y[k + 1] + h11 * h[k] * m[k + 1]
    return out


def read_metrics(m):
    pts = []
    p = f"results/{m}/metrics.jsonl"
    if not os.path.exists(p): return pts
    seen = set()
    for l in open(p):
        if not l.strip(): continue
        d = json.loads(l)
        if d["step"] in seen: continue
        seen.add(d["step"]); pts.append((d["step"], d["test_pass_pct"], d.get("eval_tps", 0)))
    return sorted(pts)


def parse_log(m, pat):
    out = []
    p = f"results/{m}/train.log"
    if not os.path.exists(p): return out
    for line in open(p):
        mi = re.search(r"Iter (\d+)", line)
        if not mi: continue
        mv = re.search(pat, line)
        if mv: out.append((int(mi.group(1)), float(mv.group(1))))
    return out


# ---------- 1. learning curve (both, smoothed) ----------
plt.figure(figsize=(8, 5))
for m in ("qwen", "gemma"):
    pts = read_metrics(m)
    if not pts: continue
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    grid = np.linspace(min(xs), max(xs), 300)
    plt.plot(grid, pchip(xs, ys, grid), "-", color=COL[m], lw=2.2, label=LBL[m])
    plt.plot(xs, ys, "o", color=COL[m], ms=6)
plt.title("Holdout test-pass % per checkpoint — Qwen vs Gemma")
plt.xlabel("training iteration"); plt.ylabel("function holdout test-pass %")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT}/learning_curve_compare.png", dpi=120); plt.close()

# ---------- 2. train loss (both, dense) ----------
plt.figure(figsize=(8, 5))
for m in ("qwen", "gemma"):
    tl = parse_log(m, r"Train loss ([0-9.]+)")
    if tl: plt.plot([p[0] for p in tl], [p[1] for p in tl], "-", color=COL[m], lw=1.8, label=LBL[m])
plt.title("Training loss — Qwen vs Gemma"); plt.xlabel("iteration"); plt.ylabel("train loss")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT}/train_loss_compare.png", dpi=120); plt.close()

# ---------- 3. val loss (both) ----------
plt.figure(figsize=(8, 5))
for m in ("qwen", "gemma"):
    vl = parse_log(m, r"Val loss ([0-9.]+)")
    if vl: plt.plot([p[0] for p in vl], [p[1] for p in vl], "-o", color=COL[m], lw=2, ms=4, label=LBL[m])
plt.title("Validation loss — Qwen vs Gemma"); plt.xlabel("iteration"); plt.ylabel("val loss")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT}/val_loss_compare.png", dpi=120); plt.close()

# ---------- 4. accuracy grouped bars ----------
# (from the eval result files)
ACC = {
    "Function base":      {"qwen": 0,  "gemma": 0},
    "Function SFT":       {"qwen": 94, "gemma": 93},
    "Function DPO":       {"qwen": 93, "gemma": 93},
    "Graph SFT":          {"qwen": 46, "gemma": 15},
    "Graph DPO":          {"qwen": 61, "gemma": 15},
}
groups = list(ACC.keys()); x = np.arange(len(groups)); w = 0.38
plt.figure(figsize=(9.5, 5))
plt.bar(x - w/2, [ACC[g]["qwen"] for g in groups], w, color=COL["qwen"], label=LBL["qwen"])
plt.bar(x + w/2, [ACC[g]["gemma"] for g in groups], w, color=COL["gemma"], label=LBL["gemma"])
for i, g in enumerate(groups):
    plt.text(i - w/2, ACC[g]["qwen"] + 1, f"{ACC[g]['qwen']}%", ha="center", fontsize=8)
    plt.text(i + w/2, ACC[g]["gemma"] + 1, f"{ACC[g]['gemma']}%", ha="center", fontsize=8)
plt.title("Cross-compiled test-pass % — Qwen vs Gemma")
plt.xticks(x, groups, rotation=12); plt.ylabel("test-pass %"); plt.ylim(0, 105)
plt.grid(True, axis="y", alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT}/accuracy_compare.png", dpi=120); plt.close()

# ---------- 5. graph idiom: correct% + transpile-similarity (SFT vs DPO) ----------
SIM = {"qwen": {"SFT": 0.457, "DPO": 0.338}, "gemma": {"SFT": 0.667, "DPO": 0.667}}
COR = {"qwen": {"SFT": 46, "DPO": 61}, "gemma": {"SFT": 15, "DPO": 15}}
fig, ax1 = plt.subplots(figsize=(8.5, 5))
stages = ["SFT", "DPO"]; x = np.arange(2); w = 0.2
ax1.bar(x - 1.5*w, [COR["qwen"][s] for s in stages], w, color=COL["qwen"], label="Qwen correct%")
ax1.bar(x - 0.5*w, [COR["gemma"][s] for s in stages], w, color=COL["gemma"], label="Gemma correct%")
ax1.set_ylabel("graph correct %"); ax1.set_ylim(0, 80)
ax2 = ax1.twinx()
ax2.plot(x, [SIM["qwen"][s] for s in stages], "o--", color=COL["qwen"], label="Qwen sim")
ax2.plot(x, [SIM["gemma"][s] for s in stages], "o--", color=COL["gemma"], label="Gemma sim")
ax2.axhline(0.26, ls=":", color="green", alpha=0.6); ax2.text(1.4, 0.27, "idiomatic ref 0.26", color="green", fontsize=8)
ax2.set_ylabel("transpile-similarity (lower = more idiomatic)"); ax2.set_ylim(0, 1.0)
ax1.set_xticks(x); ax1.set_xticklabels(["SFT", "DPO"]); ax1.set_title("Graph idiom: correctness (bars) + similarity (lines)")
ax1.grid(True, axis="y", alpha=0.3)
l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="upper center"); plt.tight_layout()
plt.savefig(f"{OUT}/graph_idiom_compare.png", dpi=120); plt.close()

print("wrote:", ", ".join(sorted(os.listdir(OUT))))
