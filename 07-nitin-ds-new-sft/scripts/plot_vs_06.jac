"""Comparison-plot generator for 07-nitin-ds-new-sft.

Produces the two charts the final report embeds, analogous to 06's
`spectrum_vs_stock.png` and `nitin_vs_old_dataset.png` -- same chart shapes, new
comparison axis (07 vs 06 instead of 06 vs jacgen2):

  spectrum_vs_stock.png   stock vs spectrum WITHIN 07, both holdouts, all three
                          stages (SFT-final / DPO-best / DPO-final). Two panels,
                          one per holdout.
  07_vs_06.png            07 vs 06 on the SHARED holdout A, the four
                          cross-dataset cells (stock SFT-final, spectrum
                          SFT-final, stock DPO-final, spectrum DPO-final).

Data source is `metrics_functional.jsonl` as written by
`04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac` -- one JSON object per
(step, category, gate_class) with {step, category, gate_class, total, runs,
runs_pct}. Only the `__overall__` / `__all__` rows at FULL holdout size are
headline cells; the smaller-`total` rows are in-loop subset sweeps and are
ignored here.

Step encoding, as the eval runners write it (see eval_dpo_nofuse.sh /
eval_dpo_spectrum.sh):

    step == 0                     base model, no adapter
    0 < step < 500000             SFT-final (the step is TOTAL_ITERS, e.g. 8200)
    500000 <= step < 1000000      DPO-best  (500000 + the best step)
    step >= 1000000               DPO-final (1000000 + the last step)

06 is read READ-ONLY as the reference. This script never writes outside
--out-dir.

It is deliberately fail-fast: if 07's own metrics files do not exist yet (i.e.
training/eval has not run), it prints exactly which are missing and exits 1.
That is the expected state during scaffolding -- do not work around it.

Run (from anywhere; paths are resolved from this file's location):

    jac run model-experiments/07-nitin-ds-new-sft/scripts/plot_vs_06.jac
    jac run model-experiments/07-nitin-ds-new-sft/scripts/plot_vs_06.jac --out-dir /tmp/x

Alongside the PNGs it prints a plain-text table of every cell plus the
two-proportion z-test for each comparison, so the report-writing agent can copy
real numbers straight into `docs/reports/07-final-comparison.md` instead of
re-deriving them.
"""

import os;
import sys;
import json;
import math;
import argparse;
import from pathlib { Path }

import matplotlib;

with entry {
    matplotlib.use("Agg");
}

import matplotlib.pyplot as plt;

glob ROOT: Path = Path(str(__file__)).resolve().parents[3];
glob EXP07: Path = ROOT / "model-experiments" / "07-nitin-ds-new-sft";
glob EXP06: Path = ROOT / "model-experiments" / "06-nitin-ds-sft";

"""Results subdirectory per (holdout, arm, phase). Identical relative layout in
06 and 07 -- that is what makes the two directly comparable."""
glob DIRS: dict[str, dict[str, str]] = {
    "A": {
        "stock_sft": "stock_probe/results/sft",
        "spectrum_sft": "spectrum_probe/results/sft-spectrum",
        "stock_dpo": "stock_probe/results/dpo-nofuse",
        "spectrum_dpo": "spectrum_probe/results/dpo-spectrum"
    },
    "B": {
        "stock_sft": "stock_probe/results/sft-holdoutB",
        "spectrum_sft": "spectrum_probe/results/sft-spectrum-holdoutB",
        "stock_dpo": "stock_probe/results/dpo-nofuse-holdoutB",
        "spectrum_dpo": "spectrum_probe/results/dpo-spectrum-holdoutB"
    }
};

glob STAGES: list[str] = ["sft_final", "dpo_best", "dpo_final"];
glob STAGE_LABELS: dict[str, str] = {
    "base": "Base",
    "sft_final": "SFT-final",
    "dpo_best": "DPO-best",
    "dpo_final": "DPO-final"
};

glob STOCK_COLOR: str = "#4C72B0";
glob SPECTRUM_COLOR: str = "#DD8452";
glob PHASE07_COLOR: str = "#55A868";
glob PHASE06_COLOR: str = "#8172B3";


"""Load the __overall__/__all__ rows out of one metrics_functional.jsonl.
Returns [] when the file is absent so callers can report it as missing."""
def read_overall(path: Path) -> list {
    rows: list = [];
    if not path.exists() {
        return rows;
    }
    for line in open(str(path)) {
        line = line.strip();
        if not line {
            continue;
        }
        rec = json.loads(line);
        if rec.get("category") == "__overall__" and rec.get("gate_class") == "__all__" {
            rows.append(rec);
        }
    }
    return rows;
}


"""Keep only rows scored on the FULL holdout. The in-loop subset sweeps write
smaller `total`s into the same file; the full-holdout size is the max."""
def full_rows(rows: list) -> list {
    if not rows {
        return [];
    }
    n_full = max([int(r["total"]) for r in rows]);
    return [r for r in rows if int(r["total"]) == n_full];
}


"""Classify one full-holdout row by the runners' step encoding (see docstring)."""
def stage_of(step: int) -> str {
    if step == 0 {
        return "base";
    }
    if step >= 1000000 {
        return "dpo_final";
    }
    if step >= 500000 {
        return "dpo_best";
    }
    return "sft_final";
}


"""Collect one arm's cells for one holdout: {stage: {"runs": x, "total": n}}.
Later rows win, so a re-run of an eval overwrites its earlier value."""
def collect_arm(exp_root: Path, holdout: str, arm: str) -> dict {
    out: dict = {};
    for phase in ["sft", "dpo"] {
        rel = DIRS[holdout][f"{arm}_{phase}"];
        path = exp_root / rel / "metrics_functional.jsonl";
        for r in full_rows(read_overall(path)) {
            out[stage_of(int(r["step"]))] = {
                "runs": int(r["runs"]),
                "total": int(r["total"])
            };
        }
    }
    return out;
}


"""Every metrics file this holdout needs, as (label, path) -- used to report
exactly which ones are missing rather than failing on the first one."""
def expected_files(exp_root: Path, holdout: str) -> list[tuple[str, Path]] {
    out: list[tuple[str, Path]] = [];
    for (key, rel) in DIRS[holdout].items() {
        out.append((f"{holdout}/{key}", exp_root / rel / "metrics_functional.jsonl"));
    }
    return out;
}


"""Two-proportion z-test, the same formula every prior comparison in this
project used (spec.md section 7.1). Returns (z, p) with p two-sided."""
def ztest(x1: int, n1: int, x2: int, n2: int) -> tuple {
    if n1 <= 0 or n2 <= 0 {
        return (0.0, 1.0);
    }
    p1: float = x1 / n1;
    p2: float = x2 / n2;
    p_pool: float = (x1 + x2) / (n1 + n2);
    denom: float = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2));
    if denom == 0.0 {
        return (0.0, 1.0);
    }
    z: float = (p1 - p2) / denom;
    p: float = math.erfc(abs(z) / math.sqrt(2.0));  # jac:ignore[E1010,E1053]
    return (z, p);
}


"""Percentage for a cell, or None when the cell has not been produced yet."""
def pct(cell: any) -> float | None {
    if not cell {
        return None;
    }
    total: int = int(cell["total"]);
    if total == 0 {
        return None;
    }
    runs: int = int(cell["runs"]);
    return 100.0 * runs / total;
}


"""Bar height for matplotlib -- absent cells plot as 0 and get labelled n/a."""
def bar_h(cell: any) -> float {
    v = pct(cell);
    return 0.0 if v is None else v;
}


"""Text label above a bar: `72.4%\\n(619/855)`, or `n/a` for a missing cell."""
def bar_label(cell: any) -> str {
    v = pct(cell);
    if v is None {
        return "n/a";
    }
    return f"{v:.1f}%\n({int(cell['runs'])}/{int(cell['total'])})";
}


"""Chart 1: stock vs spectrum WITHIN 07, one panel per holdout."""
def plot_arms(data: any, out_path: Path) {
    holdouts = [h for h in ["A", "B"] if h in data];  # jac:ignore[E1111]
    (fig, axes) = plt.subplots(1, len(holdouts), figsize=(7.0 * len(holdouts), 5.0));
    ax_list: any = axes if len(holdouts) > 1 else [axes];

    for (i, h) in enumerate(holdouts) {
        ax = ax_list[i];
        xs = list(range(len(STAGES)));
        stock_h = [bar_h(data[h]["stock"].get(s, {})) for s in STAGES];
        spec_h = [bar_h(data[h]["spectrum"].get(s, {})) for s in STAGES];
        ax.bar([x - 0.2 for x in xs], stock_h, width=0.38, label="stock (trailing-16)", color=STOCK_COLOR);
        ax.bar([x + 0.2 for x in xs], spec_h, width=0.38, label="spectrum (SNR picks)", color=SPECTRUM_COLOR);
        for (j, s) in enumerate(STAGES) {
            ax.text(xs[j] - 0.2, stock_h[j] + 1.0, bar_label(data[h]["stock"].get(s, {})),
                    ha="center", va="bottom", fontsize=7);
            ax.text(xs[j] + 0.2, spec_h[j] + 1.0, bar_label(data[h]["spectrum"].get(s, {})),
                    ha="center", va="bottom", fontsize=7);
        }
        ax.set_xticks(xs);
        ax.set_xticklabels([STAGE_LABELS[s] for s in STAGES]);
        ax.set_ylim(0, 112);
        ax.set_ylabel("functional pass rate (%)");
        ax.set_title(f"07 -- holdout {h}");
        ax.legend(loc="lower left", fontsize=8);
        ax.grid(axis="y", alpha=0.3);
    }

    fig.suptitle("07-nitin-ds-new-sft: Spectrum vs stock, both holdouts");
    fig.tight_layout();
    fig.savefig(str(out_path), dpi=120);
    plt.close(fig);  # jac:ignore[E1054]
    print(f"[plot] wrote {out_path}");
}


"""Chart 2: 07 vs 06 on the SHARED holdout A -- the cross-dataset comparison."""
def plot_vs06(d07: any, d06: any, out_path: Path) {
    cells: list[tuple[str, str, str]] = [
        ("stock", "sft_final", "stock\nSFT-final"),
        ("spectrum", "sft_final", "spectrum\nSFT-final"),
        ("stock", "dpo_final", "stock\nDPO-final"),
        ("spectrum", "dpo_final", "spectrum\nDPO-final")
    ];
    (fig, ax) = plt.subplots(1, 1, figsize=(9.0, 5.0));
    xs = list(range(len(cells)));
    h07 = [bar_h(d07[c[0]].get(c[1], {})) for c in cells];
    h06 = [bar_h(d06[c[0]].get(c[1], {})) for c in cells];
    ax.bar([x - 0.2 for x in xs], h07, width=0.38, label="07 (new corpus)", color=PHASE07_COLOR);
    ax.bar([x + 0.2 for x in xs], h06, width=0.38, label="06 (previous corpus)", color=PHASE06_COLOR);
    for (j, c) in enumerate(cells) {
        ax.text(xs[j] - 0.2, h07[j] + 1.0, bar_label(d07[c[0]].get(c[1], {})),
                ha="center", va="bottom", fontsize=7);
        ax.text(xs[j] + 0.2, h06[j] + 1.0, bar_label(d06[c[0]].get(c[1], {})),
                ha="center", va="bottom", fontsize=7);
    }
    ax.set_xticks(xs);
    ax.set_xticklabels([c[2] for c in cells]);
    ax.set_ylim(0, 112);
    ax.set_ylabel("functional pass rate (%)");
    ax.set_title("07's new corpus vs 06's corpus -- same holdout A (855), same recipe");
    ax.legend(loc="lower left", fontsize=8);
    ax.grid(axis="y", alpha=0.3);
    fig.tight_layout();
    fig.savefig(str(out_path), dpi=120);
    plt.close(fig);  # jac:ignore[E1054]
    print(f"[plot] wrote {out_path}");
}


"""Print one z-test line. Absent cells print as SKIP rather than a fake 0."""
def report_line(label: str, a: any, b: any) {
    if not a or not b {
        print(f"  {label:<44} SKIP (a cell is missing)");
        return;
    }
    (z, p) = ztest(int(a["runs"]), int(a["total"]), int(b["runs"]), int(b["total"]));
    pa: float = pct(a);  # jac:ignore[E1001]
    pb: float = pct(b);  # jac:ignore[E1001]
    d: float = pa - pb;
    sig = "YES" if p < 0.05 else "no";
    print(f"  {label:<44} {pa:5.1f}% vs {pb:5.1f}%  d={d:+5.1f}pp  z={z:+6.3f}  p={p:.4f}  sig={sig}");
}


def main() -> int {
    ap = argparse.ArgumentParser(description="07 comparison plots: arms within 07, and 07 vs 06.");
    ap.add_argument("--out-dir", default=str(EXP07 / "docs" / "reports"),
                    help="where to write the PNGs (default: 07's docs/reports)");
    args = ap.parse_args();
    out_dir = Path(str(args.out_dir));  # jac:ignore
    out_dir.mkdir(parents=True, exist_ok=True);

    # Holdout A is mandatory -- it is the shared instrument every headline number
    # and every cross-phase comparison rides on. Fail fast and say exactly which
    # files are missing; during scaffolding ALL of them are, and that is correct.
    missing: list = [];
    for (label, path) in expected_files(EXP07, "A") {
        if not path.exists() {
            missing.append(f"{label}: {path}");
        }
    }
    if missing {
        print("ERROR: 07 has no holdout-A metrics yet -- training/eval has not run.", file=sys.stderr);  # jac:ignore[E1054]
        print("Missing:", file=sys.stderr);  # jac:ignore[E1054]
        for m in missing {
            print(f"  - {m}", file=sys.stderr);  # jac:ignore[E1054]
        }
        print("Run Stages 3-5 of docs/workflow.md first, then re-run this script.", file=sys.stderr);  # jac:ignore[E1054]
        return 1;
    }

    data: dict = {};
    for h in ["A", "B"] {
        have = all([pair[1].exists() for pair in expected_files(EXP07, h)]);  # jac:ignore[E1030]
        if not have {
            print(f"[warn] 07 holdout {h}: metrics incomplete, panel skipped");
            continue;
        }
        data[h] = {
            "stock": collect_arm(EXP07, h, "stock"),
            "spectrum": collect_arm(EXP07, h, "spectrum")
        };
    }

    plot_arms(data, out_dir / "spectrum_vs_stock.png");

    # 06 is the reference for the cross-dataset chart. Read-only; if it is not
    # present the arms chart still stands on its own.
    d06: dict = {
        "stock": collect_arm(EXP06, "A", "stock"),
        "spectrum": collect_arm(EXP06, "A", "spectrum")
    };
    if d06["stock"] and d06["spectrum"] {
        plot_vs06(data["A"], d06, out_dir / "07_vs_06.png");
    } else {
        print(f"[warn] 06 holdout-A metrics not found under {EXP06} -- 07_vs_06.png skipped");
    }

    print("");
    print("=== RQ1: Spectrum replication -- stock vs spectrum WITHIN 07 ===");
    for h in ["A", "B"] {
        if h not in data {
            continue;
        }
        for s in STAGES {
            report_line(f"holdout {h} / {STAGE_LABELS[s]} spectrum vs stock",
                        data[h]["spectrum"].get(s, {}), data[h]["stock"].get(s, {}));
        }
    }

    print("");
    print("=== RQ2: dataset comparison -- 07 vs 06, shared holdout A ===");
    if d06["stock"] and d06["spectrum"] {
        for (arm, stage) in [("stock", "sft_final"), ("spectrum", "sft_final"),
                             ("stock", "dpo_best"), ("spectrum", "dpo_best"),
                             ("stock", "dpo_final"), ("spectrum", "dpo_final")] {
            report_line(f"{arm} {STAGE_LABELS[stage]} 07 vs 06",
                        data["A"][arm].get(stage, {}), d06[arm].get(stage, {}));
        }
    } else {
        print("  SKIP -- 06 metrics unavailable");
    }

    print("");
    print("NOTE: these are UNPAIRED two-proportion z-tests. spec.md section 7.1 requires");
    print("paired McNemar wherever both sides answered identical items, which is every");
    print("comparison above. Use scripts/grade_eval_detail.jac's per-row output to build");
    print("the discordant-pair counts; the numbers here are the marginal-rate view only.");
    return 0;
}


with entry:__main__ {
    sys.exit(main());
}
