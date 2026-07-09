"""Generate every deck chart as a clean matplotlib figure (PDF, vector).
Run: python3 make_charts.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

BLUE = "#2F6FBF"
ORANGE = "#EF6136"
DARK = "#ED4834"
INK = "#212121"
INKMUTED = "#5A5A5A"
GRID = "#DDDDDD"

FONT = "Helvetica" if any("Helvetica" in f.name for f in fm.fontManager.ttflist) else "DejaVu Sans"

plt.rcParams.update({
    "font.family": FONT,
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INKMUTED,
    "axes.linewidth": 0.8,
    "xtick.color": INKMUTED,
    "ytick.color": INKMUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def clean_axes(ax, xgrid=False, ygrid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(INKMUTED)
    ax.tick_params(axis="both", length=0)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(f"{name}.pdf", bbox_inches="tight", pad_inches=0.06, transparent=True)
    plt.close(fig)


def label_bars(ax, bars, fmt="{:.1f}", dy=1.5, color=INK, size=8):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                 ha="center", va="bottom", fontsize=size, color=color)


# ---------------------------------------------------------------- chart 1
# Era 1 proof chart: three models, base vs after-intervention, OOM shown as OOM
def chart_era1():
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    groups = [
        ("jac-qwen3coder", [("base", 14.3, False), ("+GRPO", 14.3, False)]),
        ("qwen3coder\n(fresh)", [("base", 0, False), ("+warm", 14.3, False), ("+GRPO", 14.3, False)]),
        ("qwen3.6\n(dense 27B)", [("base", 0, False), ("+warm", None, True), ("+GRPO", None, True)]),
    ]
    x = 0
    xticks, xlabels = [], []
    group_centers = []
    bar_w = 0.8
    gap = 1.0
    for gname, bars in groups:
        start = x
        for label, val, is_oom in bars:
            color = BLUE if label == "base" else ORANGE
            if is_oom:
                ax.bar(x, 6, width=bar_w, facecolor="none", edgecolor=DARK,
                       hatch="////", linewidth=0.9, zorder=3)
                ax.text(x, 6 + 2, "OOM", ha="center", va="bottom",
                        fontsize=8, color=DARK, fontweight="medium")
            else:
                b = ax.bar(x, val, width=bar_w, color=color, zorder=3)
                label_bars(ax, b)
            xticks.append(x)
            xlabels.append(label)
            x += 1
        group_centers.append((start + x - 1) / 2)
        x += gap

    clean_axes(ax)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7.5)
    ax.set_ylim(0, 22)
    ax.set_ylabel("pass@1 (%)")
    for c, (gname, _) in zip(group_centers, groups):
        ax.text(c, -6.5, gname, ha="center", va="top", fontsize=8, color=INK)
    ax.set_xlim(-0.8, x - gap + 0.8)
    from matplotlib.patches import Patch
    handles = [Patch(color=BLUE, label="baseline read"),
               Patch(color=ORANGE, label="after intervention"),
               Patch(facecolor="none", edgecolor=DARK, hatch="////", label="OOM, never measured")]
    ax.legend(handles=handles, loc="upper right", fontsize=7.5, frameon=False, ncol=1)
    save(fig, "era1_interventions")


# ---------------------------------------------------------------- chart 2
def chart_era2_flat():
    fig, ax = plt.subplots(figsize=(6.9, 2.1))
    cats = ["base", "SFT r5", "SFT r20", "SFT all", "SFT+GRPO", "raw-GRPO"]
    vals = [11.1] * 6
    b = ax.bar(cats, vals, width=0.55, color=DARK, zorder=3)
    label_bars(ax, b)
    clean_axes(ax)
    ax.set_ylim(0, 22)
    ax.set_ylabel("greedy pass@1 (%)")
    save(fig, "era2_flat_ladder")


# ---------------------------------------------------------------- chart 3
def chart_era3_bug():
    fig, ax = plt.subplots(figsize=(4.3, 2.5))
    cats = ["fresh\nqwen3coder", "jac-qwen3coder"]
    broken = [11.1, 11.1]
    fixed = [33.3, 38.9]
    xpos = range(len(cats))
    w = 0.32
    b1 = ax.bar([p - w / 2 for p in xpos], broken, width=w, color=BLUE, zorder=3, label="broken extractor")
    b2 = ax.bar([p + w / 2 for p in xpos], fixed, width=w, color=ORANGE, zorder=3, label="fixed extractor")
    label_bars(ax, b1)
    label_bars(ax, b2)
    clean_axes(ax)
    ax.set_xticks(list(xpos))
    ax.set_xticklabels(cats)
    ax.set_ylim(0, 48)
    ax.set_ylabel("greedy pass@1 (%)")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    save(fig, "era3_undercount")


# ---------------------------------------------------------------- chart 4
def chart_corrected_ladder():
    fig, ax = plt.subplots(figsize=(6.9, 2.6))
    cats = ["base", "SFT r5", "SFT r20", "SFT all", "SFT+GRPO", "raw-GRPO"]
    vals = [38.9, 55.6, 61.1, 55.6, 55.6, 38.9]
    colors = [BLUE, ORANGE, ORANGE, ORANGE, ORANGE, BLUE]
    b = ax.bar(cats, vals, width=0.55, color=colors, zorder=3)
    label_bars(ax, b)
    clean_axes(ax)
    ax.set_ylim(0, 72)
    ax.set_ylabel("greedy pass@1 (%)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="no SFT"), Patch(color=ORANGE, label="SFT'd")],
               loc="upper right", fontsize=7.5, frameon=False)
    save(fig, "corrected_ladder")


# ---------------------------------------------------------------- chart 5
def chart_kscale():
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    xs = [1, 8, 32]
    ys = [38.9, 72.2, 88.9]
    ax.plot(xs, ys, color=ORANGE, linewidth=1.6, marker="o", markersize=4.5, zorder=3)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(["greedy", "k=8", "k=32"])
    clean_axes(ax)
    ax.set_ylim(0, 100)
    ax.set_ylabel("pass rate (%)")
    ax.set_title("base model, sampling budget only\nzero training", fontsize=8, color=INKMUTED, loc="left")
    save(fig, "kscale")


# ---------------------------------------------------------------- chart 6
def chart_holdouts():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6))
    cats = ["pure-fn", "graph", "conv", "clean", "big"]
    greedy_base = [38.9, 35.3, 63.6, 43.8, 34.4]
    greedy_sft = [61.1, 29.4, 72.7, 68.8, 43.8]
    bok_base = [72.2, 52.9, 72.7, 93.8, 62.5]
    bok_sft = [77.8, 64.7, 81.8, 87.5, 71.9]

    for ax, base, sft, title in [
        (axes[0], greedy_base, greedy_sft, "greedy pass@1 (%)"),
        (axes[1], bok_base, bok_sft, "best-of-k deploy (%)"),
    ]:
        xpos = range(len(cats))
        w = 0.34
        b1 = ax.bar([p - w / 2 for p in xpos], base, width=w, color=BLUE, zorder=3)
        b2 = ax.bar([p + w / 2 for p in xpos], sft, width=w, color=ORANGE, zorder=3)
        label_bars(ax, b2, size=6.5, dy=1.2)
        clean_axes(ax)
        ax.set_xticks(list(xpos))
        ax.set_xticklabels(cats, fontsize=7.5)
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=BLUE, label="base"), Patch(color=ORANGE, label="SFT")],
               loc="upper center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, "holdouts")


# ---------------------------------------------------------------- chart 7
def chart_journey():
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    stages = ["v1\n(broken eval)", "eval fixed\nbase", "+ SFT", "+ best-of-k", "+ drop\njunk tasks"]
    vals = [11.1, 38.9, 61.1, 77.8, 93.8]
    xs = range(len(stages))
    ax.plot(xs, vals, color=ORANGE, linewidth=1.6, marker="o", markersize=4.5, zorder=3)
    ax.scatter([0], [vals[0]], color=BLUE, zorder=4, s=26)
    ax.annotate("measured wrong", (0, vals[0]), textcoords="offset points", xytext=(8, -2),
                fontsize=7.5, color=BLUE, va="center")
    for x, y in zip(xs, vals):
        if x == 0:
            continue
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 7),
                    fontsize=8, color=INK, ha="center")
    clean_axes(ax)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(stages, fontsize=7.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("holdout accuracy (%)")
    save(fig, "journey")


# ---------------------------------------------------------------- chart 8
# The rung ladder: greedy pass@1 vs number of tasks trained on, real x-scale
def chart_rungs():
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    xs = list(range(7))  # evenly spaced positions, real counts in the labels
    ys = [38.9, 38.9, 50.0, 55.6, 50.0, 61.1, 55.6]
    labels = ["base\n(0)", "r1\n(1)", "r3\n(3)", "r5\n(5)", "r10\n(10)", "r20\n(20)", "all\n(84)"]
    peak_i = 5
    ax.plot(xs, ys, color=INKMUTED, linewidth=1.3, zorder=2, linestyle="--")
    colors = [BLUE] + [ORANGE] * 6
    for x, y, c in zip(xs, ys, colors):
        ax.scatter([x], [y], color=c, s=55, zorder=3, edgecolor="white", linewidth=0.8)
    # highlight the peak
    ax.scatter([peak_i], [ys[peak_i]], color=DARK, s=130, zorder=4, facecolor="none",
               edgecolor=DARK, linewidth=1.6)
    ax.annotate("peak", (peak_i, ys[peak_i]), textcoords="offset points", xytext=(0, 15),
                fontsize=8.5, color=DARK, ha="center", fontweight="medium")
    for x, y in zip(xs, ys):
        dy = -16 if y > 45 else 10
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, dy),
                    fontsize=7.5, color=INK, ha="center")
    clean_axes(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_xlim(-0.4, 6.4)
    ax.set_ylim(25, 70)
    ax.set_ylabel("greedy pass@1 (%)")
    ax.set_xlabel("tasks trained on (rung size)", fontsize=8, labelpad=6)
    save(fig, "rungs")


if __name__ == "__main__":
    chart_era1()
    chart_era2_flat()
    chart_era3_bug()
    chart_corrected_ladder()
    chart_kscale()
    chart_holdouts()
    chart_journey()
    chart_rungs()
    print("done")
