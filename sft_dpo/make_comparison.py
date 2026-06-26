"""6-model SFT+DPO bake-off comparison -> results/comparison/*.png + matrix.md.

Post-hoc: run AFTER all bake-off runs finish. Reads each results/<name>/ dir and
plots all models together. No live eval — data is recorded during the runs
(metrics.jsonl per checkpoint from run_probe's curve stage, train.log, idiom-metrics).

    python3 sft_dpo/make_comparison.py
"""
import json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

# Anchor to repo root so result paths resolve regardless of cwd.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "comparison")

# All bake-off models. qwen = incumbent (parsed, not re-run). Dict order = legend order.
COL = {
    "qwen":    "#1f77b4",
    "gptoss":  "#ff7f0e",
    "qwen3i":  "#2ca02c",
    "dscoder": "#d62728",
    "ling":    "#9467bd",
    "qwen25c": "#8c564b",
}
LBL = {
    "qwen":    "Qwen3-Coder-30B-A3B (incumbent)",
    "gptoss":  "gpt-oss-20b",
    "qwen3i":  "Qwen3-30B-A3B-Instruct",
    "dscoder": "DeepSeek-Coder-V2-Lite",
    "ling":    "Ling-Coder-lite",
    "qwen25c": "Qwen2.5-Coder-14B",
}
PASS_RE = re.compile(r"cross-compiled test pass:\s*(\d+)%")


def pchip(x, y, xs):
    """Fritsch-Carlson monotone cubic Hermite — smooth, no overshoot."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    h = np.diff(x); d = np.diff(y) / h
    m = np.zeros_like(y)
    for i in range(1, len(y) - 1):
        if d[i - 1] * d[i] > 0:
            m[i] = 2.0 / (1.0 / d[i - 1] + 1.0 / d[i])
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
    p = f"{RES}/{m}/metrics.jsonl"
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
    p = f"{RES}/{m}/train.log"
    if not os.path.exists(p): return out
    for line in open(p):
        mi = re.search(r"Iter (\d+)", line)
        if not mi: continue
        mv = re.search(pat, line)
        if mv: out.append((int(mi.group(1)), float(mv.group(1))))
    return out


def pass_pct(path):
    """Behavioral test-pass % from an eval_probe stdout dump, or None if absent."""
    try:
        m = PASS_RE.search(open(path).read())
        return int(m.group(1)) if m else None
    except FileNotFoundError:
        return None


def avg_sim(path):
    """Last-row avg transpile-similarity from an idiom-metrics jsonl, or None."""
    try:
        rows = [json.loads(l) for l in open(path) if l.strip()]
        return rows[-1].get("avg_sim") if rows else None
    except FileNotFoundError:
        return None


def build_matrix():
    """One row per model: (name, label, base%, sft%, dpo%, sft_sim, dpo_sim)."""
    rows = []
    for name, label in LBL.items():
        d = f"{RES}/{name}"
        rows.append((name, label,
                     pass_pct(f"{d}/base.txt"),
                     pass_pct(f"{d}/finetuned.txt"),
                     pass_pct(f"{d}/dpo/finetuned.txt"),
                     avg_sim(f"{d}/idiom-metrics.jsonl"),
                     avg_sim(f"{d}/dpo/idiom-metrics.jsonl")))
    return rows


def write_matrix():
    pct = lambda v: "—" if v is None else f"{v}%"
    sim = lambda v: "—" if v is None else f"{v:.3f}"
    out = ["| model | base | SFT | DPO | SFT sim | DPO sim | idiom gain |",
           "|---|---|---|---|---|---|---|"]
    for name, label, base, sft, dpo, ss, sd in build_matrix():
        gain = f"{ss - sd:+.3f}" if (ss is not None and sd is not None) else "—"
        out.append(f"| {label} | {pct(base)} | {pct(sft)} | {pct(dpo)} | "
                   f"{sim(ss)} | {sim(sd)} | {gain} |")
    text = "\n".join(out)
    open(f"{OUT}/matrix.md", "w").write(text + "\n")
    return text


def main():
    os.makedirs(OUT, exist_ok=True)

    # 1. learning curve (all models, smoothed)
    plt.figure(figsize=(9, 5.5))
    for m in COL:
        pts = read_metrics(m)
        if not pts: continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        grid = np.linspace(min(xs), max(xs), 300)
        plt.plot(grid, pchip(xs, ys, grid), "-", color=COL[m], lw=2.2, label=LBL[m])
        plt.plot(xs, ys, "o", color=COL[m], ms=5)
    plt.title("Holdout test-pass % per checkpoint — bake-off")
    plt.xlabel("training iteration"); plt.ylabel("function holdout test-pass %")
    plt.grid(True, alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{OUT}/learning_curve_compare.png", dpi=120); plt.close()

    # 2. train loss (all models, dense)
    plt.figure(figsize=(9, 5.5))
    for m in COL:
        tl = parse_log(m, r"Train loss ([0-9.]+)")
        if tl: plt.plot([p[0] for p in tl], [p[1] for p in tl], "-", color=COL[m], lw=1.6, label=LBL[m])
    plt.title("Training loss — bake-off"); plt.xlabel("iteration"); plt.ylabel("train loss")
    plt.grid(True, alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{OUT}/train_loss_compare.png", dpi=120); plt.close()

    # 3. val loss (all models)
    plt.figure(figsize=(9, 5.5))
    for m in COL:
        vl = parse_log(m, r"Val loss ([0-9.]+)")
        if vl: plt.plot([p[0] for p in vl], [p[1] for p in vl], "-o", color=COL[m], lw=1.8, ms=3, label=LBL[m])
    plt.title("Validation loss — bake-off"); plt.xlabel("iteration"); plt.ylabel("val loss")
    plt.grid(True, alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{OUT}/val_loss_compare.png", dpi=120); plt.close()

    mat = build_matrix()

    # 4. behavioral accuracy: base / SFT / DPO (grouped bars, all models with data)
    stages = ["base", "SFT", "DPO"]
    present = [r for r in mat if any(v is not None for v in r[2:5])]
    x = np.arange(len(stages)); n = max(len(present), 1); w = 0.8 / n
    plt.figure(figsize=(10, 5.5))
    for i, (name, label, base, sft, dpo, _ss, _sd) in enumerate(present):
        vals = [v if v is not None else 0 for v in (base, sft, dpo)]
        plt.bar(x + (i - (n - 1) / 2) * w, vals, w, color=COL[name], label=label)
    plt.title("Function holdout test-pass % — base / SFT / DPO")
    plt.xticks(x, stages); plt.ylabel("test-pass %"); plt.ylim(0, 105)
    plt.grid(True, axis="y", alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{OUT}/accuracy_compare.png", dpi=120); plt.close()

    # 5. idiom transpile-similarity: SFT vs DPO (lower = more idiomatic)
    sim_present = [r for r in mat if r[5] is not None or r[6] is not None]
    x = np.arange(2); n = max(len(sim_present), 1); w = 0.8 / n
    plt.figure(figsize=(10, 5.5))
    for i, (name, label, _b, _s, _d, ssft, sdpo) in enumerate(sim_present):
        vals = [ssft if ssft is not None else 0, sdpo if sdpo is not None else 0]
        plt.bar(x + (i - (n - 1) / 2) * w, vals, w, color=COL[name], label=label)
    plt.axhline(0.26, ls=":", color="green", alpha=0.7)
    plt.text(1.3, 0.27, "idiomatic ref 0.26", color="green", fontsize=8)
    plt.title("Idiom transpile-similarity — SFT vs DPO (lower = more idiomatic)")
    plt.xticks(x, ["SFT", "DPO"]); plt.ylabel("avg transpile-similarity"); plt.ylim(0, 1.0)
    plt.grid(True, axis="y", alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{OUT}/idiom_compare.png", dpi=120); plt.close()

    # 6. matrix table
    print(write_matrix())
    print("wrote:", ", ".join(sorted(os.listdir(OUT))))


if __name__ == "__main__":
    main()
