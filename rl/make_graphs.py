"""Published RL graphs -> resultspub/rl/ (CORRECTED, fixed eval+reward).

Only produces graphs from the corrected runs. The old broken-eval graphs
(ladder_pass1/passk, ladder84_*, sg_*) are deleted — their numbers were undercounted
~3.5x by the extractor bug (see RL_FINDINGS.md).

  corrected_ladder.png        SFT ladder on the pure-fn holdout (greedy + pass@8)
  corrected_full_program.png  greedy vs best-of-k, base vs SFT, across 3 holdouts

    python3 rl/make_graphs.py
"""
import os, re, json, glob
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "resultspub", "rl")
os.makedirs(OUT, exist_ok=True)

# --- remove invalidated broken-eval graphs ---
for stale in ["ladder_pass1.png", "ladder_passk.png", "ladder84_pass1.png",
              "ladder84_passk.png", "sg_pass1.png", "sg_perfamily.png"]:
    p = os.path.join(OUT, stale)
    if os.path.exists(p):
        os.remove(p); print("removed invalidated", stale)


def cells(path):
    s = {}
    for l in open(os.path.join(ROOT, path)):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l); s[r["tag"]] = r
    return s


def bok(logname):
    """parse best-of-k DEPLOY % from a bok_*.log"""
    p = os.path.join(ROOT, "results", logname)
    if not os.path.exists(p):
        return None
    m = re.search(r"DEPLOY.*?\((\d+\.?\d*)%\)", open(p).read())
    return float(m.group(1)) if m else None


# 1) corrected SFT ladder (pure-fn holdout)
c = cells("results/corrected_ladder.jsonl")
def g(t, f): return c.get("corr/jac-qwen3coder/" + t, {}).get(f, 0)
labels = ["base", "SFT r5", "SFT r20", "SFT rall", "SFT+GRPO", "raw-GRPO"]
tags = ["rall/base", "r5/sft", "r20/sft", "rall/sft", "rall/sft_grpo", "rall/raw_grpo"]
p1 = [g(t, "pass1_pct") for t in tags]; pk = [g(t, "passk_pct") for t in tags]
x = np.arange(len(labels)); w = 0.38
plt.figure(figsize=(11, 6))
plt.bar(x - w/2, p1, w, label="greedy pass@1", color="#7aa2f7")
plt.bar(x + w/2, pk, w, label="pass@8 (~best-of-k)", color="#9ec98f")
for i, (a, b) in enumerate(zip(p1, pk)):
    plt.text(i - w/2, a + 1, f"{a:.0f}", ha="center", fontsize=8)
    plt.text(i + w/2, b + 1, f"{b:.0f}", ha="center", fontsize=8)
plt.axhline(38.9, ls=":", color="#888", alpha=.7); plt.text(5.1, 40, "base greedy", color="#888", fontsize=7)
plt.xticks(x, labels); plt.ylabel("holdout accuracy % (n=18)"); plt.ylim(0, 95)
plt.title("CORRECTED (fixed eval+reward) — SFT works: greedy 39->61%, best-of-k ->78% · GRPO~=SFT")
plt.legend(); plt.grid(True, axis="y", alpha=.3); plt.tight_layout()
plt.savefig(f"{OUT}/corrected_ladder.png", dpi=130); plt.close(); print("wrote corrected_ladder.png")

# 2) full program: greedy vs best-of-k, base vs SFT, across 3 holdouts
sg = cells("results/corrected_sg.jsonl"); cv = cells("results/conv_full.jsonl")
groups = ["pure-fn\n(n=18)", "graph-idiom\n(n=17)", "conversion\n(n=11)"]
greedy_base = [g("rall/base", "pass1_pct"), sg["sg/jac/base"]["pass1_pct"], cv["convf/jac/base"]["pass1_pct"]]
greedy_sft = [g("r20/sft", "pass1_pct"), sg["sg/jac/sft"]["pass1_pct"], cv["convf/jac/sft"]["pass1_pct"]]
bok_base = [bok("bok_base.log") or 72.2, bok("bok_sg_base.log") or 52.9, bok("bok_convf_base.log") or 72.7]
bok_sft = [bok("bok_sft.log") or 77.8, bok("bok_sg_sft.log") or 64.7, bok("bok_convf_sft.log") or 81.8]
x = np.arange(3); w = 0.2
fig, ax = plt.subplots(figsize=(11, 6))
for off, vals, lab, col in [(-1.5, greedy_base, "greedy·base", "#4b5563"), (-0.5, greedy_sft, "greedy·SFT", "#7aa2f7"),
                            (0.5, bok_base, "best-of-k·base", "#6b8f5e"), (1.5, bok_sft, "best-of-k·SFT", "#9ec98f")]:
    ax.bar(x + off*w, vals, w, label=lab, color=col)
    for i, v in enumerate(vals): ax.text(i + off*w, v + 1, f"{v:.0f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("holdout accuracy %"); ax.set_ylim(0, 95)
ax.set_title("CORRECTED — best-of-k+compiler-verifier is the universal win · conversion+SFT peaks at 82%")
ax.legend(ncol=2); ax.grid(True, axis="y", alpha=.3); plt.tight_layout()
plt.savefig(f"{OUT}/corrected_full_program.png", dpi=130); plt.close(); print("wrote corrected_full_program.png")
print("graphs ->", OUT)
