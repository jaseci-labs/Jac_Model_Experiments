"""Comprehensive bake-off comparison graphs for resultspub/initmodelchoice.

Reuses make_comparison's parsers (reads live results/<name>/ files) and adds two
all-in-one charts: every model, every stage, distinct color + legend.
  - combined_accuracy.png : function + graph holdout, base/SFT/DPO test-pass %
  - combined_idiom.png    : function + graph idiom similarity, SFT/DPO (lower=better)
Also (re)emits the standard per-tier charts + learning curves into the same dir.

    python3 sft_dpo/make_pub_graphs.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_comparison as mc
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

OUT = os.path.join(mc.ROOT, "resultspub", "initmodelchoice", "comparison")
os.makedirs(OUT, exist_ok=True)

fn = {r[0]: r for r in mc.build_matrix("")}        # name -> (name,label,base,sft,dpo,ssim,dsim)
gr = {r[0]: r for r in mc.build_matrix("graph-")}
MODELS = list(mc.COL.keys())


def grouped(stages, ylabel, title, fname, top, pct=True, ref=None):
    """stages: list of (label, table, idx). One bar group per stage, one color per
    model. Real values get numeric labels; missing values get an N/A jail bar."""
    x = np.arange(len(stages)); n = len(MODELS); w = 0.8 / n
    plt.figure(figsize=(15, 7))
    for i, m in enumerate(MODELS):
        vals = [(tbl.get(m)[idx] if tbl.get(m) else None) for (_, tbl, idx) in stages]
        mc.draw_group(x + (i - (n - 1) / 2) * w, vals, w, mc.COL[m], mc.LBL[m], top, pct=pct)
    if ref is not None:
        plt.axhline(ref, ls=":", color="green", alpha=0.7)
        plt.text(len(stages) - 1.4, ref + 0.01, f"idiomatic ref {ref}", color="green", fontsize=8)
    plt.xticks(x, [s[0] for s in stages]); plt.ylabel(ylabel); plt.ylim(0, top * 1.08)
    plt.title(title); plt.grid(True, axis="y", alpha=0.3); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/{fname}", dpi=130); plt.close()


# 1. combined accuracy — all stages, function + graph
grouped(
    [("func base", fn, 2), ("func SFT", fn, 3), ("func DPO", fn, 4),
     ("graph base", gr, 2), ("graph SFT", gr, 3), ("graph DPO", gr, 4)],
    "test-pass %", "Bake-off accuracy — all models, all stages (function 150 + graph 13)",
    "combined_accuracy.png", 100, pct=True,
)

# 2. combined idiom — SFT/DPO, function + graph (lower = more idiomatic)
grouped(
    [("func SFT", fn, 5), ("func DPO", fn, 6), ("graph SFT", gr, 5), ("graph DPO", gr, 6)],
    "avg transpile-similarity (lower = more idiomatic)",
    "Bake-off idiom — all models, SFT vs DPO (function + graph)",
    "combined_idiom.png", 1.0, pct=False, ref=0.26,
)

# 3. learning curves (function holdout subset per checkpoint) — distinct colors + legend
plt.figure(figsize=(10, 6)); any_c = False
for m in MODELS:
    pts = mc.read_metrics(m)
    if not pts:
        continue
    any_c = True
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    g = np.linspace(min(xs), max(xs), 300)
    plt.plot(g, mc.pchip(xs, ys, g), "-", color=mc.COL[m], lw=2.2, label=mc.LBL[m])
    plt.plot(xs, ys, "o", color=mc.COL[m], ms=5)
plt.title("Learning curve — holdout test-pass % per checkpoint")
plt.xlabel("training iteration"); plt.ylabel("function holdout test-pass %")
plt.grid(True, alpha=0.3)
if any_c:
    plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/combined_learning_curve.png", dpi=130); plt.close()

# 4. train + val loss (the "learning rates and everything")
for fname, pat, title, ylab, style in [
    ("combined_train_loss.png", r"Train loss ([0-9.]+)", "Training loss — all models", "train loss", "-"),
    ("combined_val_loss.png", r"Val loss ([0-9.]+)", "Validation loss — all models", "val loss", "-o"),
]:
    plt.figure(figsize=(10, 6)); has = False
    for m in MODELS:
        d = mc.parse_log(m, pat)
        if d:
            has = True
            plt.plot([p[0] for p in d], [p[1] for p in d], style, color=mc.COL[m], lw=1.7, ms=3, label=mc.LBL[m])
    plt.title(title); plt.xlabel("iteration"); plt.ylabel(ylab); plt.grid(True, alpha=0.3)
    if has:
        plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/{fname}", dpi=130); plt.close()

print("wrote ->", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f)
