"""Published RL ladder graphs -> resultspub/rl/ (matches sft_dpo/make_pub_graphs.py).

Reads the ladder result jsonls and emits the charts that tell the story:
  ladder_pass1.png   primary holdout: greedy pass@1 over rungs, line per model.cond (the flat ~27% null)
  ladder_passk.png   pass@8 mean by condition (SFT lifts sampling; GRPO ~= SFT)
  sg_pass1.png       step-7 sg-inclusive holdout: pass@1 over rungs (the one real movement)
  sg_perfamily.png   sg holdout base-vs-SFT per family (sg slice 0 -> 1/5)
Also copies the live Studio RL screenshots in.

    python3 rl/make_graphs.py
"""
import os, json, shutil
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "resultspub", "rl")
os.makedirs(OUT, exist_ok=True)
RUNGS = ["1", "3", "5", "10", "20", "all"]
RX = {r: i for i, r in enumerate(RUNGS)}
COND_COLOR = {"base": "#888888", "sft": "#7aa2f7", "sft_grpo": "#d8a657",
              "sft_grpo_tuned": "#e07a8b", "raw_grpo": "#9ec98f"}
MODEL_LS = {"qwen3coder": "-", "jac-qwen3coder": "--"}


def load(path):
    """path -> {(model,rung,cond): genrow}"""
    cells = {}
    if not os.path.exists(path):
        return cells
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        p = r.get("tag", "").split("/")
        if len(p) != 4 or p[3] != "gen":
            continue
        cells[(p[1], p[0].lstrip("r"), p[2])] = r   # (model,rung,cond) -> row, last wins
    return cells


def curve(cells, field, title, ylabel, fname):
    plt.figure(figsize=(11, 6))
    series = {}
    for (model, rung, cond), r in cells.items():
        v = r.get(field)
        if v is None:
            continue
        series.setdefault((model, cond), {})[rung] = v
    for (model, cond), pts in sorted(series.items()):
        xs = sorted(pts, key=lambda r: RX.get(r, 99))
        x = [RX[r] for r in xs]
        y = [pts[r] for r in xs]
        plt.plot(x, y, MODEL_LS.get(model, "-"), color=COND_COLOR.get(cond, "#333"),
                 marker="o", ms=4, lw=1.6, label=f"{model}.{cond}")
    plt.xticks(range(len(RUNGS)), RUNGS)
    plt.xlabel("train-N (rung)"); plt.ylabel(ylabel); plt.ylim(0, max(60, 5))
    plt.title(title); plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, ncol=2, loc="upper left")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, fname), dpi=130); plt.close()
    print("wrote", fname)


def passk_bars(cells, fname):
    conds = ["base", "sft", "sft_grpo", "sft_grpo_tuned", "raw_grpo"]
    means = []
    for c in conds:
        vs = [r.get("passk_pct") for (m, rg, cc), r in cells.items() if cc == c and r.get("passk_pct") is not None]
        means.append(sum(vs) / len(vs) if vs else 0)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(conds, means, color=[COND_COLOR[c] for c in conds])
    for b, v in zip(bars, means):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
    plt.axhline(means[0], ls=":", color="#888", alpha=0.7)
    plt.ylabel("mean pass@8 (holdout)"); plt.ylim(0, max(means) * 1.2)
    plt.title("pass@8 by condition — SFT lifts sampling, GRPO ≈ SFT (boundary unchanged)")
    plt.grid(True, axis="y", alpha=0.3); plt.xticks(rotation=15)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, fname), dpi=130); plt.close()
    print("wrote", fname)


def sg_perfamily(cells, fname):
    fams = ["gb", "graph", "lib", "sg"]
    # use rung 'all' base vs sft, per model
    def frac(model, cond):
        r = cells.get((model, "all", cond))
        out = {}
        if r and r.get("by_family"):
            for f, pair in r["by_family"].items():
                out[f] = 100.0 * pair[0] / pair[1] if pair[1] else 0
        return out
    import numpy as np
    x = np.arange(len(fams)); w = 0.2
    plt.figure(figsize=(9, 5))
    combos = [("qwen3coder", "base"), ("qwen3coder", "sft"),
              ("jac-qwen3coder", "base"), ("jac-qwen3coder", "sft")]
    cols = ["#bbbbbb", "#7aa2f7", "#888888", "#d8a657"]
    for i, (m, c) in enumerate(combos):
        fr = frac(m, c)
        plt.bar(x + (i - 1.5) * w, [fr.get(f, 0) for f in fams], w,
                color=cols[i], label=f"{m}.{c}")
    plt.xticks(x, fams); plt.ylabel("pass@1 % (sg holdout, rung=all)")
    plt.title("sg-inclusive holdout per family — sg slice cracked only by fresh SFT (0→1/5)")
    plt.grid(True, axis="y", alpha=0.3); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, fname), dpi=130); plt.close()
    print("wrote", fname)


prim = load(os.path.join(ROOT, "results", "rl_ladder.jsonl"))
sg = load(os.path.join(ROOT, "results", "rl_ladder_sg.jsonl"))
assert prim, "no primary results — run the ladder first"
curve(prim, "pass1_pct", "Primary holdout (gb+lib) — greedy pass@1 by rung: FLAT (the null)",
      "gen pass@1 %", "ladder_pass1.png")
passk_bars(prim, "ladder_passk.png")
if sg:
    curve(sg, "pass1_pct", "sg-inclusive holdout — greedy pass@1 by rung (fresh SFT moves it)",
          "gen pass@1 %", "sg_pass1.png")
    sg_perfamily(sg, "sg_perfamily.png")

# copy the live Studio screenshots alongside the generated charts
for s in ["studio-rl-full.png", "studio-rl-ladder.png"]:
    src = os.path.join(ROOT, "docs", "rl", "img", s)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT, s)); print("copied", s)
print("graphs ->", OUT)
