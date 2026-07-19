"""Static matplotlib chart set for the CPT-v2 readout -- companion to the
interactive HTML dashboard (built separately), reads the same source-of-
-truth JSON so numbers can't drift between the two. Every chart is saved as
its own PNG under 03-new/results/cpt-v2/images/ so each one can be dropped
into a doc or slide independently."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "03-new" / "results" / "cpt-v2"
JSON_DIR = RESULTS / "json"
CHARTS_DIR = RESULTS / "images"
DATASET_V1 = ROOT / "03-new" / "dataset" / "cpt"
DATASET_V2 = ROOT / "03-new" / "dataset" / "cpt-v2"

# ---- palette, matches the HTML dashboard's tokens for continuity ----
BASE = "#8A8F82"
V1 = "#6E86B8"
V2 = "#3854A6"
ORACLE = "#2E7D5B"
TIE = "#A9812F"
GOOD = "#2E7D5B"
BAD = "#AE3A2C"
INK = "#1E2420"
GRID = "#D8DBD2"
PAPER = "#FCFCFA"

plt.rcParams.update({
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "axes.edgecolor": "#B9C0AE",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.alpha": 0.9,
    "axes.axisbelow": True,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def load():
    legs = json.loads((JSON_DIR / "training_state.json").read_text())["legs"]
    track_a = json.loads((JSON_DIR / "track_a.json").read_text())
    track_b_raw = json.loads((JSON_DIR / "track_b_raw.json").read_text())
    track_b_agg = json.loads((JSON_DIR / "track_b.json").read_text())
    acceptance = json.loads((JSON_DIR / "acceptance.json").read_text())
    curation = json.loads((DATASET_V2 / "curation.json").read_text())
    manifest_v1 = json.loads((DATASET_V1 / "manifest.json").read_text())["sources"]
    manifest_v2 = json.loads((DATASET_V2 / "manifest.json").read_text())["sources"]
    return dict(legs=legs, track_a=track_a, track_b_raw=track_b_raw, track_b_agg=track_b_agg,
                acceptance=acceptance, curation=curation, manifest_v1=manifest_v1, manifest_v2=manifest_v2)


def save(fig, name):
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHARTS_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


# ============================================================
# 1-3: corpus
# ============================================================

def chart_corpus_composition(d):
    sources = ["docs", "osp_paper", "blogs", "rehearsal", "code"]
    labels = ["docs", "OSP paper", "blogs", "rehearsal", "code (dropped in v2)"]
    v1 = [d["manifest_v1"].get(s, {}).get("tokens", 0) for s in sources]
    v2 = [d["manifest_v2"].get(s, {}).get("tokens", 0) for s in sources]
    y = np.arange(len(sources))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(y + 0.19, v1, height=0.36, color=BASE, label=f"cpt-v1 ({sum(v1)/1e6:.2f}M total)")
    ax.barh(y - 0.19, v2, height=0.36, color=V2, label=f"cpt-v2 ({sum(v2)/1e6:.2f}M total)")
    for yi, (a, b) in enumerate(zip(v1, v2)):
        ax.text(a + max(v1) * 0.01, yi + 0.19, f"{a/1e6:.2f}M", va="center", fontsize=9)
        ax.text(b + max(v1) * 0.01, yi - 0.19, ("dropped" if b == 0 else f"{b/1e6:.2f}M"), va="center", fontsize=9, color=V2)
    ax.set_yticks(y, labels)
    ax.set_xlabel("tokens")
    ax.set_title("Corpus token composition: CPT-v1 → CPT-v2")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.legend(loc="lower right", frameon=False)
    ax.invert_yaxis()
    save(fig, "01_corpus_composition")


def chart_corpus_pie(d):
    sources = ["docs", "osp_paper", "blogs", "rehearsal"]
    labels = ["docs", "OSP paper", "blogs", "rehearsal"]
    v2 = [d["manifest_v2"].get(s, {}).get("tokens", 0) for s in sources]
    colors = [V2, "#7A93C9", "#A6B7DC", TIE]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    total = sum(v2)
    wedges, _, autotexts = ax.pie(
        v2, labels=labels, autopct=lambda p: f"{p:.1f}%\n({p*total/100/1e6:.2f}M)",
        colors=colors, startangle=90, pctdistance=0.72,
        wedgeprops=dict(edgecolor=PAPER, linewidth=1.5), textprops=dict(fontsize=10),
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontsize(9)
    ax.set_title(f"CPT-v2 corpus share by source (n={total/1e6:.2f}M tokens, code dropped)")
    save(fig, "02_corpus_share_pie")


def chart_curation_verdicts(d):
    verdicts = [v.get("verdict", "?") for v in d["curation"].values()]
    keep = verdicts.count("keep")
    upweight = verdicts.count("upweight")
    drop = verdicts.count("drop")
    total = len(verdicts)
    fig, ax = plt.subplots(figsize=(9, 2.6))
    left = 0
    for val, color, label in [(keep, GOOD, "keep"), (upweight, V2, "upweight"), (drop, BAD, "drop")]:
        ax.barh(0, val, left=left, color=color, height=0.6, label=f"{label} ({val:,}, {val/total*100:.1f}%)")
        ax.text(left + val / 2, 0, f"{val/total*100:.1f}%", ha="center", va="center", color="white", fontweight="bold")
        left += val
    ax.set_xlim(0, total)
    ax.set_yticks([])
    ax.set_xlabel(f"chunks reviewed by Fable curation pass (n={total:,})")
    ax.set_title("Curation pass verdicts, pre-pack")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=3, frameon=False)
    save(fig, "03_curation_verdicts")


# ============================================================
# 4-9: training
# ============================================================

def chart_loss_curves(d):
    legs = d["legs"]
    x = [l["leg"] for l in legs]
    train = [l["train_loss"] for l in legs]
    val = [l["val_loss"] for l in legs]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.margins(y=0.1)
    ax.plot(x, train, "-o", color=BASE, label="train loss", linewidth=2, markersize=5)
    ax.plot(x, val, "-o", color=V2, label="val loss", linewidth=2, markersize=5)
    blend = ax.get_xaxis_transform()
    for leg, label in [(6, "floor"), (8, "target"), (12, "ceiling")]:
        ax.axvline(leg, color="#888", linestyle="--", linewidth=1)
        ax.text(leg, 0.965, label, transform=blend, ha="center", fontsize=9, color="#666")
    ax.annotate(f"train {train[-1]:.3f}", (x[-1], train[-1]), textcoords="offset points", xytext=(-45, -14), color=BASE, fontweight="bold")
    ax.annotate(f"val {val[-1]:.3f}", (x[-1], val[-1]), textcoords="offset points", xytext=(-35, 10), color=V2, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xlabel("leg")
    ax.set_ylabel("loss")
    ax.set_title("CPT-v2: train & val loss across the 12-leg epoch loop")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    save(fig, "04_loss_curves")


def chart_val_loss_delta(d):
    legs = d["legs"]
    val = [l["val_loss"] for l in legs]
    deltas = [val[i] - val[i - 1] for i in range(1, len(val))]
    xs = [legs[i]["leg"] for i in range(1, len(val))]
    colors = [BAD if dv > 0 else GOOD for dv in deltas]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(xs, deltas, color=colors)
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xticks(xs)
    ax.set_xlabel("leg (vs. previous leg)")
    ax.set_ylabel("Δ val loss")
    ax.set_title("Leg-over-leg val loss change (red = uptick, green = improvement)")
    save(fig, "05_val_loss_delta")


def chart_lr_schedule(d):
    legs = d["legs"]
    x = [l["leg"] for l in legs]
    lr = [l["final_lr"] for l in legs]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, lr, "-o", color=V2, linewidth=2, markersize=5)
    ax.fill_between(x, lr, color=V2, alpha=0.12)
    peak_i = int(np.argmax(lr))
    ax.annotate(f"peak L{x[peak_i]}\n{lr[peak_i]*1e6:.2f}e-6", (x[peak_i], lr[peak_i]),
                textcoords="offset points", xytext=(10, 8))
    ax.annotate(f"floor L{x[-1]}\n{lr[-1]*1e6:.2f}e-6", (x[-1], lr[-1]),
                textcoords="offset points", xytext=(-70, 10))
    ax.set_xticks(x)
    ax.set_xlabel("leg")
    ax.set_ylabel("learning rate (end of leg)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*1e6:.1f}e-6"))
    ax.set_title("One coherent cosine LR decay across all 12 legs")
    save(fig, "06_lr_schedule")


def chart_leg_duration(d):
    legs = d["legs"]
    x = [l["leg"] for l in legs]
    mins = [l["duration_s"] / 60 for l in legs]
    total_h = sum(mins) / 60
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x, mins, color=V1)
    ax.axhline(np.mean(mins), color=BAD, linestyle="--", linewidth=1, label=f"mean {np.mean(mins):.0f} min/leg")
    ax.set_xticks(x)
    ax.set_xlabel("leg")
    ax.set_ylabel("wall-clock minutes")
    ax.set_title(f"Per-leg training duration (total {total_h:.1f}h across 12 legs)")
    ax.legend(loc="lower right", frameon=False)
    save(fig, "07_leg_duration")


def chart_train_vs_val_scatter(d):
    legs = d["legs"]
    train = [l["train_loss"] for l in legs]
    val = [l["val_loss"] for l in legs]
    legnum = [l["leg"] for l in legs]
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(train, val, c=legnum, cmap="viridis", s=90, edgecolor="white", linewidth=0.8, zorder=3)
    for t, v, n in zip(train, val, legnum):
        ax.annotate(str(n), (t, v), textcoords="offset points", xytext=(6, 4), fontsize=8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("leg")
    ax.set_xlabel("train loss")
    ax.set_ylabel("val loss")
    ax.set_title("Train vs. val loss per leg (color = leg number)")
    save(fig, "08_train_vs_val_scatter")


def chart_cf_check_strip(d):
    legs = d["legs"]
    x = [l["leg"] for l in legs]
    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.scatter(x, [1] * len(x), s=900, color=GOOD, alpha=0.18, zorder=1)
    ax.scatter(x, [1] * len(x), s=140, color=GOOD, zorder=2)
    for xi, l in zip(x, legs):
        ax.annotate(l["cf_score"], (xi, 1), ha="center", va="center", color="white", fontweight="bold", fontsize=8, zorder=3)
    halt_i = [i for i, l in enumerate(legs) if l["decision"].startswith("halt")]
    if halt_i:
        ax.scatter([x[halt_i[0]]], [1], s=1700, facecolor="none", edgecolor=V2, linewidth=2, zorder=1)
        ax.annotate("halt (ceiling)", (x[halt_i[0]], 1), textcoords="offset points", xytext=(0, -30), ha="center", color=V2, fontsize=9)
    ax.set_xticks(x, [f"L{i}" for i in x])
    ax.set_yticks([])
    ax.set_ylim(0.5, 1.5)
    ax.set_title("CF-check per leg -- 16/16 general-Python tasks, every single leg")
    ax.grid(False)
    save(fig, "09_cf_check_strip")


# ============================================================
# 10-15: Track A
# ============================================================

def track_a_means(track_a):
    n = len(track_a)
    base = sum(v["base"] for v in track_a.values()) / n
    v1 = sum(v["cpt_v1"] for v in track_a.values()) / n
    v2 = sum(v["cpt_v2"] for v in track_a.values()) / n
    return base, v1, v2


def chart_track_a_means(d):
    base, v1, v2 = track_a_means(d["track_a"])
    req_base, req_v1 = base + 0.03, v1 + 0.03
    labels = ["base", "cpt-v1", "cpt-v2"]
    vals = [base, v1, v2]
    colors = [BASE, V1, V2]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0015, f"{v:.4f}", ha="center", fontweight="bold")
    ax.hlines(req_base, -0.35, 0.35, color=BAD, linestyle="--")
    ax.text(0.35, req_base, f" need≥{req_base:.4f}", color=BAD, va="center", fontsize=9)
    ax.hlines(req_v1, 0.65, 1.35, color=BAD, linestyle="--")
    ax.text(1.35, req_v1, f" need≥{req_v1:.4f}", color=BAD, va="center", fontsize=9)
    ax.set_ylim(min(vals) - 0.01, max(req_base, req_v1) + 0.015)
    ax.set_ylabel("mean cosine-to-jac-gpt")
    ax.set_title("Track A: mean cosine similarity vs. required +0.03 margin")
    save(fig, "10_track_a_means")


def chart_track_a_delta_hist(d, key, label, fname):
    rows = d["track_a"]
    deltas = [v["cpt_v2"] - v[key] for v in rows.values()]
    fig, ax = plt.subplots(figsize=(8, 5))
    n, bins, patches = ax.hist(deltas, bins=18, edgecolor="white", linewidth=0.6)
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor(V2 if left >= 0 else BASE)
    ax.axvline(0, color=INK, linewidth=1.2)
    mean_d = np.mean(deltas)
    ax.axvline(mean_d, color=BAD, linestyle="--", linewidth=1.2, label=f"mean {mean_d:+.4f}")
    wins = sum(1 for x in deltas if x > 0)
    losses = sum(1 for x in deltas if x < 0)
    ax.set_xlabel(f"cosine delta, cpt-v2 − {label}")
    ax.set_ylabel("question count")
    ax.set_title(f"Per-question delta vs {label} (n=100, {wins} win / {losses} loss)")
    ax.legend(loc="upper right", frameon=False)
    save(fig, fname)


def chart_track_a_win_loss(d):
    rows = d["track_a"]
    vs_base_w = sum(1 for v in rows.values() if v["cpt_v2"] > v["base"])
    vs_base_l = sum(1 for v in rows.values() if v["cpt_v2"] < v["base"])
    vs_v1_w = sum(1 for v in rows.values() if v["cpt_v2"] > v["cpt_v1"])
    vs_v1_l = sum(1 for v in rows.values() if v["cpt_v2"] < v["cpt_v1"])
    groups = ["cpt-v2 vs base", "cpt-v2 vs cpt-v1"]
    wins = [vs_base_w, vs_v1_w]
    losses = [vs_base_l, vs_v1_l]
    y = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.barh(y, wins, color=V2, label="cpt-v2 wins")
    ax.barh(y, losses, left=wins, color=BASE, label="cpt-v2 losses")
    for yi, (w, l) in enumerate(zip(wins, losses)):
        ax.text(w / 2, yi, f"{w} win", ha="center", va="center", color="white", fontweight="bold")
        ax.text(w + l / 2, yi, f"{l} loss", ha="center", va="center", color="white", fontweight="bold")
    ax.axvline(50, color=BAD, linestyle="--", linewidth=1.2, label="50 / 50 line")
    ax.set_yticks(y, groups)
    ax.set_xlim(0, 100)
    ax.set_xlabel("of 100 paired questions")
    ax.set_title("Track A win/loss counts, cpt-v2 vs. each baseline")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False)
    save(fig, "13_track_a_win_loss")


def chart_track_a_boxplot(d):
    rows = d["track_a"]
    base = [v["base"] for v in rows.values()]
    v1 = [v["cpt_v1"] for v in rows.values()]
    v2 = [v["cpt_v2"] for v in rows.values()]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    bp = ax.boxplot([base, v1, v2], tick_labels=["base", "cpt-v1", "cpt-v2"], patch_artist=True,
                     medianprops=dict(color=INK, linewidth=1.6), widths=0.5,
                     flierprops=dict(marker="o", markersize=4, markerfacecolor="#999", markeredgecolor="none"))
    for patch, color in zip(bp["boxes"], [BASE, V1, V2]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylabel("cosine-to-jac-gpt score")
    ax.set_title("Distribution of cosine-to-jac-gpt scores, all 100 questions")
    save(fig, "14_track_a_boxplot")


def chart_track_a_scatter_v1_v2(d):
    rows = d["track_a"]
    v1 = np.array([v["cpt_v1"] for v in rows.values()])
    v2 = np.array([v["cpt_v2"] for v in rows.values()])
    lo, hi = min(v1.min(), v2.min()) - 0.01, max(v1.max(), v2.max()) + 0.01
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([lo, hi], [lo, hi], "--", color="#888", linewidth=1, label="y = x (no difference)")
    above = v2 > v1
    ax.scatter(v1[above], v2[above], color=V2, alpha=0.75, s=32, label=f"cpt-v2 wins ({above.sum()})")
    ax.scatter(v1[~above], v2[~above], color=BASE, alpha=0.75, s=32, label=f"cpt-v1 wins ({(~above).sum()})")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("cpt-v1 cosine-to-jac-gpt")
    ax.set_ylabel("cpt-v2 cosine-to-jac-gpt")
    ax.set_title("Per-question cosine score: cpt-v1 vs cpt-v2 (points scatter evenly → coin flip)")
    ax.legend(loc="upper left", frameon=False)
    ax.set_aspect("equal")
    save(fig, "15_track_a_scatter_v1_vs_v2")


# ============================================================
# 16-18: Track B
# ============================================================

def chart_track_b_outcome_bar(d):
    agg = d["track_b_agg"]
    total = agg["total"]
    segs = [("jac-gpt wins", agg["oracle_wins"], ORACLE), ("cpt-v2 wins", agg["cpt_v2_wins"], V2), ("ties", agg["ties"], TIE)]
    fig, ax = plt.subplots(figsize=(10, 2.4))
    left = 0
    for label, val, color in segs:
        ax.barh(0, val, left=left, color=color, height=0.6)
        if val > 3:
            ax.text(left + val / 2, 0, str(val), ha="center", va="center", color="white", fontweight="bold")
        left += val
    ax.axvline(50, color=BAD, linestyle="--", linewidth=1.6)
    ax.text(50, 0.42, "acceptance needs win-or-tie ≥ 50", color=BAD, ha="center", fontsize=9, fontweight="bold")
    win_or_tie = agg["cpt_v2_wins"] + agg["ties"]
    ax.text(total, -0.42, f"actual win-or-tie: {win_or_tie} / {total}", ha="right", fontsize=9, color="#555")
    ax.set_xlim(0, total)
    ax.set_yticks([])
    ax.set_title("Track B outcome, all 100 blind pairwise judgments")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in segs]
    ax.legend(handles, [f"{l} ({v})" for l, v, _ in segs], loc="upper center", bbox_to_anchor=(0.5, -0.5), ncol=3, frameon=False)
    save(fig, "16_track_b_outcome_bar")


def chart_track_b_pie(d):
    agg = d["track_b_agg"]
    vals = [agg["oracle_wins"], agg["cpt_v2_wins"], agg["ties"]]
    labels = ["jac-gpt wins", "cpt-v2 wins", "ties"]
    colors = [ORACLE, V2, TIE]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    wedges, _, autotexts = ax.pie(
        vals, labels=labels, autopct=lambda p: f"{p:.0f}%\n({round(p*sum(vals)/100)})",
        colors=colors, startangle=90, pctdistance=0.72,
        wedgeprops=dict(edgecolor=PAPER, linewidth=1.5), textprops=dict(fontsize=10),
        explode=(0.04, 0.04, 0.04),
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontsize(9)
    ax.set_title("Track B outcome share (n=100)")
    save(fig, "17_track_b_pie")


def chart_honest_gap_scatter(d):
    ta_by_id = {row_id: row for row_id, row in d["track_a"].items()}
    lanes = {"oracle": 2, "tie": 1, "cpt_v2": 0}
    lane_colors = {"oracle": ORACLE, "tie": TIE, "cpt_v2": V2}
    lane_labels = {"oracle": "jac-gpt won (blind)", "tie": "tie", "cpt_v2": "cpt-v2 won (blind)"}
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(11, 5))
    outlier_id = "b003-q-any-inference"
    for r in d["track_b_raw"]:
        ta = ta_by_id.get(r["id"])
        if not ta:
            continue
        cosine = ta["cpt_v2"]
        lane = lanes[r["winner"]]
        jitter = rng.uniform(-0.28, 0.28)
        is_outlier = r["id"] == outlier_id
        ax.scatter(cosine, lane + jitter, color=lane_colors[r["winner"]],
                   s=130 if is_outlier else 34, alpha=1.0 if is_outlier else 0.55,
                   edgecolor=INK if is_outlier else "none", linewidth=1.4 if is_outlier else 0, zorder=5 if is_outlier else 3)
        if is_outlier:
            ax.annotate("b003-q-any-inference: 0.9355 cosine\n(2nd-highest of all 100) — lost blind",
                        (cosine, lane + jitter), textcoords="offset points", xytext=(-160, 18), fontsize=9, fontweight="bold")
    ax.set_yticks(list(lanes.values()), [lane_labels[k] for k in lanes])
    ax.set_xlabel("cpt-v2 cosine-to-jac-gpt score")
    ax.set_title("Honest gap: cosine score vs. what the blind judge actually decided (n=100)")
    save(fig, "18_honest_gap_scatter")


# ============================================================
# 19-20: acceptance + summary
# ============================================================

def chart_acceptance_gauges(d):
    acc = d["acceptance"]
    items = [
        ("Track A vs base", acc["margin_vs_base"], 0.03, lambda v: f"{v:+.4f}"),
        ("Track A vs cpt-v1", acc["margin_vs_v1"], 0.03, lambda v: f"{v:+.4f}"),
        ("Track B win-or-tie", acc["win_or_tie_rate"], 0.5, lambda v: f"{v*100:.0f}%"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, (label, value, need, fmt) in zip(axes, items):
        passed = value >= need
        color = GOOD if passed else BAD
        domain = max(value, need) * 1.35
        ax.barh(0, domain, color="#E4E6DC", height=0.5)
        ax.barh(0, max(value, 0), color=color, height=0.5)
        ax.axvline(need, color=INK, linewidth=2)
        ax.set_xlim(0, domain)
        ax.set_yticks([])
        ax.set_title(f"{label}\n{fmt(value)} — {'PASS' if passed else 'FAIL'}", fontsize=11,
                     color=color)
        ax.grid(False)
    fig.suptitle("design.md §6.4 acceptance bar — 0 of 3 gates cleared", fontweight="bold", y=1.06)
    save(fig, "19_acceptance_gauges")


def chart_summary_dashboard(d):
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    legs = d["legs"]
    x = [l["leg"] for l in legs]
    ax1.plot(x, [l["train_loss"] for l in legs], "-o", color=BASE, label="train", markersize=3)
    ax1.plot(x, [l["val_loss"] for l in legs], "-o", color=V2, label="val", markersize=3)
    ax1.set_title("Loss per leg", fontsize=11)
    ax1.legend(fontsize=8, frameon=False)

    ax2 = fig.add_subplot(gs[0, 1])
    base, v1, v2 = track_a_means(d["track_a"])
    ax2.bar(["base", "v1", "v2"], [base, v1, v2], color=[BASE, V1, V2])
    ax2.set_ylim(min(base, v1, v2) - 0.01, max(base, v1, v2) + 0.01)
    ax2.set_title("Track A means", fontsize=11)

    ax3 = fig.add_subplot(gs[0, 2])
    agg = d["track_b_agg"]
    vals = [agg["oracle_wins"], agg["cpt_v2_wins"], agg["ties"]]
    ax3.pie(vals, colors=[ORACLE, V2, TIE], startangle=90,
            wedgeprops=dict(edgecolor=PAPER, linewidth=1))
    ax3.set_title(f"Track B vs jac-gpt: won {agg['oracle_wins']} / cpt-v2 {agg['cpt_v2_wins']} / tie {agg['ties']}", fontsize=11)

    ax4 = fig.add_subplot(gs[1, 0])
    sources = ["docs", "osp_paper", "blogs", "rehearsal", "code"]
    v2c = [d["manifest_v2"].get(s, {}).get("tokens", 0) for s in sources]
    ax4.barh(sources, v2c, color=V2)
    ax4.set_title("CPT-v2 corpus (tokens)", fontsize=11)
    ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))

    ax5 = fig.add_subplot(gs[1, 1])
    acc = d["acceptance"]
    items = [("A vs base", acc["margin_vs_base"], 0.03), ("A vs v1", acc["margin_vs_v1"], 0.03),
             ("B win-tie", acc["win_or_tie_rate"], 0.5)]
    yy = np.arange(len(items))
    ax5.barh(yy, [min(v, need * 1.3) for _, v, need in items], color=[BAD] * 3)
    for i, (_, v, need) in enumerate(items):
        ax5.axvline(need, color=INK, linewidth=1)
    ax5.set_yticks(yy, [l for l, _, _ in items])
    ax5.set_title("Acceptance gates (all FAIL)", fontsize=11)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    ax6.text(0.5, 0.6, "REJECTED", ha="center", va="center", fontsize=30, fontweight="bold", color=BAD,
              bbox=dict(boxstyle="round,pad=0.5", edgecolor=BAD, facecolor="none", linewidth=2))
    ax6.text(0.5, 0.15, "design.md §6.4 — 0 / 3 gates cleared", ha="center", va="center", fontsize=10, color="#555")

    fig.suptitle("CPT-v2 instrument readout — summary", fontsize=16, fontweight="bold")
    save(fig, "20_summary_dashboard")


# ============================================================
# 21-22: direct cpt-v2 vs jac-gpt comparison
# ============================================================

def chart_gap_to_jacgpt(d):
    base, v1, v2 = track_a_means(d["track_a"])
    labels = ["base", "cpt-v1", "cpt-v2"]
    gaps = [1 - base, 1 - v1, 1 - v2]
    colors = [BASE, V1, V2]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    bars = ax.bar(labels, gaps, color=colors, width=0.55)
    for b, g in zip(bars, gaps):
        ax.text(b.get_x() + b.get_width() / 2, g + 0.0015, f"{g:.4f}", ha="center", fontweight="bold")
    ax.set_ylabel("mean cosine distance from jac-gpt's answer (lower = closer)")
    ax.set_title("Gap to jac-gpt, by model")
    save(fig, "21_gap_to_jacgpt")


def chart_cptv2_vs_jacgpt_head_to_head(d):
    base, v1, v2 = track_a_means(d["track_a"])
    agg = d["track_b_agg"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.bar(["cpt-v2"], [v2], color=V2, width=0.4)
    ax.axhline(1.0, color=ORACLE, linestyle="--", linewidth=1.6, label="jac-gpt (self-similarity = 1.0)")
    ax.text(0, v2 + 0.01, f"{v2:.4f}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("mean cosine-to-jac-gpt")
    ax.set_title("Track A: cpt-v2 vs jac-gpt")
    ax.legend(loc="lower center", frameon=False)

    ax2 = axes[1]
    win_or_tie = agg["cpt_v2_wins"] + agg["ties"]
    lose = agg["oracle_wins"]
    ax2.bar(["cpt-v2\nwin-or-tie"], [win_or_tie], color=V2, width=0.4)
    ax2.bar(["jac-gpt\nwins"], [lose], color=ORACLE, width=0.4)
    for i, v in enumerate([win_or_tie, lose]):
        ax2.text(i, v + 1.5, str(v), ha="center", fontweight="bold")
    ax2.axhline(50, color=BAD, linestyle="--", linewidth=1.4, label="50/50 line")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("of 100 blind judgments")
    ax2.set_title("Track B: cpt-v2 vs jac-gpt")
    ax2.legend(loc="upper center", frameon=False)

    fig.suptitle("cpt-v2 vs jac-gpt, both tracks side by side", fontsize=15, fontweight="bold")
    save(fig, "22_cptv2_vs_jacgpt_head_to_head")


def main():
    d = load()
    print("Generating CPT-v2 matplotlib chart set...")
    chart_corpus_composition(d)
    chart_corpus_pie(d)
    chart_curation_verdicts(d)
    chart_loss_curves(d)
    chart_val_loss_delta(d)
    chart_lr_schedule(d)
    chart_leg_duration(d)
    chart_train_vs_val_scatter(d)
    chart_cf_check_strip(d)
    chart_track_a_means(d)
    chart_track_a_delta_hist(d, "cpt_v1", "cpt-v1", "11_track_a_delta_hist_vs_v1")
    chart_track_a_delta_hist(d, "base", "base", "12_track_a_delta_hist_vs_base")
    chart_track_a_win_loss(d)
    chart_track_a_boxplot(d)
    chart_track_a_scatter_v1_v2(d)
    chart_track_b_outcome_bar(d)
    chart_track_b_pie(d)
    chart_honest_gap_scatter(d)
    chart_acceptance_gauges(d)
    chart_summary_dashboard(d)
    chart_gap_to_jacgpt(d)
    chart_cptv2_vs_jacgpt_head_to_head(d)
    pngs = sorted(CHARTS_DIR.glob("*.png"))
    print(f"\n{len(pngs)} charts written to {CHARTS_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
