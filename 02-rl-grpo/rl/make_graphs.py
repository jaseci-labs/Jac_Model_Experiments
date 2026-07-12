"""Published RL graphs -> 02-rl-grpo/resultspub/rl/ (CORRECTED, fixed eval+reward).

Reads 02-rl-grpo/resultspub/rl/corrected_summary.json (run 02-rl-grpo/rl/make_summary.py first). The old
broken-eval graphs are deleted — undercounted ~3.5x by the extractor bug (RL_FINDINGS.md).

  corrected_journey.png        the 11%->94% story arc (headline)
  corrected_ladder.png         SFT ladder, pure-fn holdout (greedy + pass@8)
  corrected_all_holdouts.png   greedy vs best-of-k, base vs SFT, across 5 holdouts
  corrected_kscale.png         sampling budget (greedy / k=8 / k=32) vs accuracy
  corrected_followup.png       clean + bigger-holdout re-measures

    python3 02-rl-grpo/rl/make_summary.py && python3 02-rl-grpo/rl/make_graphs.py
"""
import os, json
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "02-rl-grpo", "resultspub", "rl")
os.makedirs(OUT, exist_ok=True)
S = json.load(open(os.path.join(OUT, "corrected_summary.json")))

for stale in ["ladder_pass1.png", "ladder_passk.png", "ladder84_pass1.png",
              "ladder84_passk.png", "sg_pass1.png", "sg_perfamily.png"]:
    p = os.path.join(OUT, stale)
    if os.path.exists(p):
        os.remove(p); print("removed invalidated", stale)


def save(name):
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}", dpi=130); plt.close(); print("wrote", name)


# 1) JOURNEY — the story arc (headline chart)
j = S["journey"]
labels = [d["stage"] for d in j]; vals = [d["acc"] for d in j]
plt.figure(figsize=(11, 6))
xs = np.arange(len(vals))
plt.plot(xs, vals, "-o", color="#9ec98f", linewidth=2, markersize=8)
for x, v in zip(xs, vals):
    plt.text(x, v + 3, f"{v:.0f}%", ha="center", fontsize=11, color="#e6e6e6")
plt.fill_between(xs, vals, alpha=0.08, color="#9ec98f")
plt.xticks(xs, labels, rotation=12, ha="right", fontsize=9)
plt.ylabel("holdout accuracy %"); plt.ylim(0, 100)
plt.title("The corrected story — a measurement bug hid an 11%→94% Jac generator")
plt.grid(True, axis="y", alpha=.3); save("corrected_journey.png")

# 2) SFT ladder (pure-fn)
lad = S["ladder"]
labels = [d["cell"] for d in lad]; p1 = [d["greedy"] for d in lad]; pk = [d["passk"] for d in lad]
x = np.arange(len(labels)); w = 0.38
plt.figure(figsize=(11, 6))
plt.bar(x - w/2, p1, w, label="greedy pass@1", color="#7aa2f7")
plt.bar(x + w/2, pk, w, label="pass@8 (~best-of-k)", color="#9ec98f")
for i in range(len(labels)):
    plt.text(i - w/2, p1[i] + 1, f"{p1[i]:.0f}", ha="center", fontsize=8)
    plt.text(i + w/2, pk[i] + 1, f"{pk[i]:.0f}", ha="center", fontsize=8)
plt.xticks(x, labels); plt.ylabel("holdout accuracy % (n=18)"); plt.ylim(0, 95)
plt.title("CORRECTED — SFT works: greedy 39→61% (peak rung-20) · GRPO≈SFT · raw-GRPO=base")
plt.legend(); plt.grid(True, axis="y", alpha=.3); save("corrected_ladder.png")

# 3) all holdouts (5) — greedy vs best-of-k, base vs SFT
H = S["holdouts"]
groups = [f"{h['name']}\n(n={h['n']})" for h in H]
x = np.arange(len(H)); w = 0.2
fig, ax = plt.subplots(figsize=(13, 6))
for off, key, lab, col in [(-1.5, "greedy_base", "greedy·base", "#4b5563"), (-0.5, "greedy_sft", "greedy·SFT", "#7aa2f7"),
                           (0.5, "bok_base", "best-of-k·base", "#6b8f5e"), (1.5, "bok_sft", "best-of-k·SFT", "#9ec98f")]:
    vals = [h.get(key) or 0 for h in H]
    ax.bar(x + off*w, vals, w, label=lab, color=col)
    for i, v in enumerate(vals): ax.text(i + off*w, v + 1, f"{v:.0f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("holdout accuracy %"); ax.set_ylim(0, 100)
ax.set_title("CORRECTED — best-of-k+compiler-verifier across 5 holdouts · clean ceiling 94% · conversion peak 82%")
ax.legend(ncol=2); ax.grid(True, axis="y", alpha=.3); save("corrected_all_holdouts.png")

# 4) k-scaling
K = S["kscale"]
plt.figure(figsize=(8, 5.5))
xs = np.arange(len(K)); vals = [d["acc"] for d in K]
plt.bar(xs, vals, 0.5, color=["#4b5563", "#7aa2f7", "#9ec98f"])
for x0, v in zip(xs, vals): plt.text(x0, v + 1, f"{v:.0f}%", ha="center", fontsize=11)
plt.xticks(xs, [d["k"] for d in K]); plt.ylabel("pure-fn holdout accuracy %"); plt.ylim(0, 100)
plt.title("Sampling budget vs accuracy — more samples, more solved (compiler picks)")
plt.grid(True, axis="y", alpha=.3); save("corrected_kscale.png")

# 5) follow-up (clean + big) — kept as its own view
fu = {h["name"]: h for h in H}
groups = ["clean pure-fn\n(n=16, junk dropped)", "bigger holdout\n(n=32, +16 fresh syn)"]
sel = [fu["clean"], fu["big+fresh"]]
x = np.arange(2); w = 0.2
fig, ax = plt.subplots(figsize=(10, 6))
for off, key, lab, col in [(-1.5, "greedy_base", "greedy·base", "#4b5563"), (-0.5, "greedy_sft", "greedy·SFT", "#7aa2f7"),
                           (0.5, "bok_base", "best-of-k·base", "#6b8f5e"), (1.5, "bok_sft", "best-of-k·SFT", "#9ec98f")]:
    vals = [h.get(key) or 0 for h in sel]
    ax.bar(x + off*w, vals, w, label=lab, color=col)
    for i, v in enumerate(vals): ax.text(i + off*w, v + 1, f"{v:.0f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("holdout accuracy %"); ax.set_ylim(0, 100)
ax.set_title("CORRECTED follow-up — clean 94% ceiling · SFT lift holds at bigger/fresher n=32")
ax.legend(ncol=2); ax.grid(True, axis="y", alpha=.3); save("corrected_followup.png")
print("graphs ->", OUT)
