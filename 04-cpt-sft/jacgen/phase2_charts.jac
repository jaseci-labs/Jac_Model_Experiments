"""Phase 2 charts: SFT sweet-spot overlay, DPO full-curve overlay (both arms,
no early stop), DPO full-holdout 4-bar comparison, and an updated apples-to-
apples 3-stage bar chart using the full-budget (not early-stopped) DPO
numbers for both arms. Fixed categorical colors reused from every prior
chart in this directory (fresh=#4C72B0 blue, cptv2=#C44E52 red, base=#888888
gray) -- never reassigned per-chart, so color always means the same arm.
"""

import json;
import matplotlib;
import from matplotlib { pyplot }
import from pathlib { Path }

glob FRESH_SFT: str = "model-experiments/04-cpt-sft/sft_fresh_probe/results/sft/metrics_functional.jsonl";
glob FRESH_DPO: str = "model-experiments/04-cpt-sft/sft_fresh_probe/results/dpo/metrics_functional.jsonl";
glob CPTV2_SFT: str = "model-experiments/04-cpt-sft/sft_cptv2_probe/results/sft/metrics_functional.jsonl";
glob CPTV2_DPO_V3: str = "model-experiments/04-cpt-sft/sft_cptv2_probe/results/dpo-v3/metrics_functional.jsonl";
glob OUT_DIR: str = "model-experiments/04-cpt-sft/results/images";

glob FRESH_COLOR: str = "#4C72B0";
glob CPTV2_COLOR: str = "#C44E52";
glob BASE_COLOR: str = "#888888";


def load_rows(path: str) -> list[dict] {
    rows: list[dict] = [];
    p = Path(path);
    if not p.exists() { return rows; }
    for line in p.read_text().split("\n") {
        if line.strip().startswith("{") { rows.append(dict(json.loads(line))); }
    }
    return rows;
}


def subset_curve(rows: list[dict]) -> list[tuple] {
    # (step, runs_pct) for every __overall__ row with total==100 (subset evals),
    # sorted by step. Excludes the base row (step 0, total 855, already
    # filtered by total!=100) and any offset-encoded full-holdout rows from
    # eval_dpo.sh's +1000000/+500000 convention -- real training steps go up
    # to 8200 (SFT) so the offset cutoff must sit well above that, not at a
    # low round number that would clip legitimate late-SFT checkpoints.
    pts: list[tuple] = [];
    for r in rows {
        if r["category"] != "__overall__" { continue; }
        if r["total"] != 100 { continue; }
        if r["step"] > 100000 { continue; }
        pts.append((r["step"], r["runs_pct"]));
    }
    pts.sort();
    return pts;
}


with entry {
    matplotlib.use("Agg");
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True);

    fresh_sft_rows: list[dict] = load_rows(FRESH_SFT);
    cptv2_sft_rows: list[dict] = load_rows(CPTV2_SFT);
    fresh_dpo_rows: list[dict] = load_rows(FRESH_DPO);
    cptv2_dpo_rows: list[dict] = load_rows(CPTV2_DPO_V3);

    # --- Chart 1: SFT sweet-spot overlay ---
    fresh_sft_curve: list[tuple] = subset_curve(fresh_sft_rows);
    cptv2_sft_curve: list[tuple] = subset_curve(cptv2_sft_rows);

    pyplot.figure(figsize=(10, 6));  # jac:ignore[E1051]
    pyplot.plot([p[0] for p in fresh_sft_curve], [p[1] for p in fresh_sft_curve],
                marker="o", linewidth=2, markersize=8, color=FRESH_COLOR, label="fresh (no CPT)");  # jac:ignore[E1051,E1052]
    pyplot.plot([p[0] for p in cptv2_sft_curve], [p[1] for p in cptv2_sft_curve],
                marker="o", linewidth=2, markersize=8, color=CPTV2_COLOR, label="CPT-v2");  # jac:ignore[E1051,E1052]
    # Mark each arm's peak explicitly -- both peaks tie the final iteration,
    # this is the whole point of the sweet-spot check.
    fresh_peak: tuple = (0, 0);
    for p in fresh_sft_curve { if p[1] >= fresh_peak[1] { fresh_peak = p; } }
    cptv2_peak: tuple = (0, 0);
    for p in cptv2_sft_curve { if p[1] >= cptv2_peak[1] { cptv2_peak = p; } }
    pyplot.plot([fresh_peak[0]], [fresh_peak[1]], marker="o", markersize=16, markerfacecolor="none",
                markeredgecolor=FRESH_COLOR, markeredgewidth=2, linestyle="None");  # jac:ignore[E1051,E1052]
    pyplot.plot([cptv2_peak[0]], [cptv2_peak[1]], marker="o", markersize=16, markerfacecolor="none",
                markeredgecolor=CPTV2_COLOR, markeredgewidth=2, linestyle="None");  # jac:ignore[E1051,E1052]
    pyplot.title("SFT sweet-spot search -- both arms peak at (or tie) their final checkpoint");  # jac:ignore[E1051]
    pyplot.xlabel("SFT iteration");  # jac:ignore[E1051]
    pyplot.ylabel("functional pass rate % (subset=100, code_gen-only)");  # jac:ignore[E1051]
    pyplot.ylim(0, 100);  # jac:ignore[E1051]
    pyplot.legend(loc="lower right");  # jac:ignore[E1051]
    pyplot.grid(True, alpha=0.3);  # jac:ignore[E1051]
    pyplot.tight_layout();  # jac:ignore[E1051]
    pyplot.savefig(OUT_DIR + "/sft_sweetspot_curves.png", dpi=120);  # jac:ignore[E1051]
    pyplot.close();  # jac:ignore[E1051]
    print("wrote " + OUT_DIR + "/sft_sweetspot_curves.png");

    # --- Chart 2: DPO full-curve overlay (both arms, no early stop) ---
    fresh_dpo_curve: list[tuple] = subset_curve(fresh_dpo_rows);
    cptv2_dpo_curve: list[tuple] = subset_curve(cptv2_dpo_rows);

    pyplot.figure(figsize=(10, 6));  # jac:ignore[E1051]
    pyplot.plot([p[0] for p in fresh_dpo_curve], [p[1] for p in fresh_dpo_curve],
                marker="o", linewidth=2, markersize=8, color=FRESH_COLOR, label="fresh (no CPT) -- flat at 2%");  # jac:ignore[E1051,E1052]
    pyplot.plot([p[0] for p in cptv2_dpo_curve], [p[1] for p in cptv2_dpo_curve],
                marker="s", linewidth=2, markersize=8, color=CPTV2_COLOR, label="CPT-v2 -- flat at 1%");  # jac:ignore[E1051,E1052]
    pyplot.axhline(y=69.0 * 0.5, color=FRESH_COLOR, linestyle="--", linewidth=1, alpha=0.5, label="fresh collapse-gate threshold (34.5%)");  # jac:ignore[E1051,E1052]
    pyplot.axhline(y=72.0 * 0.5, color=CPTV2_COLOR, linestyle="--", linewidth=1, alpha=0.5, label="cptv2 collapse-gate threshold (36.0%)");  # jac:ignore[E1051,E1052]
    pyplot.title("DPO full 250-iter curve, no early stop -- total permanent collapse, both arms");  # jac:ignore[E1051]
    pyplot.xlabel("DPO iteration");  # jac:ignore[E1051]
    pyplot.ylabel("functional pass rate % (subset=100, code_gen-only)");  # jac:ignore[E1051]
    pyplot.ylim(0, 100);  # jac:ignore[E1051]
    pyplot.legend(loc="upper right", fontsize=9);  # jac:ignore[E1051]
    pyplot.grid(True, alpha=0.3);  # jac:ignore[E1051]
    pyplot.tight_layout();  # jac:ignore[E1051]
    pyplot.savefig(OUT_DIR + "/dpo_full_curve_both_arms.png", dpi=120);  # jac:ignore[E1051]
    pyplot.close();  # jac:ignore[E1051]
    print("wrote " + OUT_DIR + "/dpo_full_curve_both_arms.png");

    # --- Chart 3: DPO full-holdout 4-bar comparison ---
    labels: list[str] = ["fresh\nlast (step 40/250)", "fresh\nbest (step 20)", "cptv2\nlast (step 250)", "cptv2\nbest (step 20)"];
    values: list[float] = [12.0, 12.5, 12.5, 13.5];
    colors: list[str] = [FRESH_COLOR, FRESH_COLOR, CPTV2_COLOR, CPTV2_COLOR];
    counts: list[str] = ["103/855", "107/855", "107/855", "115/855"];

    pyplot.figure(figsize=(9, 6));  # jac:ignore[E1051]
    pyplot.bar(labels, values, color=colors);  # jac:ignore[E1051,E1052]
    for (i, v) in enumerate(values) {
        pyplot.text(i, v + 0.3, str(v) + "%\n(" + counts[i] + ")", ha="center", fontsize=9);  # jac:ignore[E1051]
    }
    pyplot.title("DPO full-holdout (855-row) result -- 4 measurements, both arms, both checkpoints");  # jac:ignore[E1051]
    pyplot.ylabel("functional pass rate %");  # jac:ignore[E1051]
    pyplot.ylim(0, 20);  # jac:ignore[E1051]
    pyplot.grid(True, axis="y", alpha=0.3);  # jac:ignore[E1051]
    pyplot.tight_layout();  # jac:ignore[E1051]
    pyplot.savefig(OUT_DIR + "/dpo_fullholdout_4bar.png", dpi=120);  # jac:ignore[E1051]
    pyplot.close();  # jac:ignore[E1051]
    print("wrote " + OUT_DIR + "/dpo_fullholdout_4bar.png");

    # --- Chart 4: apples-to-apples 3-stage comparison, full DPO budget both arms ---
    stage_labels: list[str] = ["base", "+SFT", "+SFT+DPO\n(full 250, last ckpt)"];
    fresh_vals: list[float] = [10.5, 69.8, 12.0];
    cptv2_vals: list[float] = [47.3, 72.6, 12.5];

    x: list[int] = list(range(len(stage_labels)));
    width: float = 0.35;
    pyplot.figure(figsize=(9, 6));  # jac:ignore[E1051]
    pyplot.bar([i - width / 2 for i in x], fresh_vals, width, label="fresh (no CPT)", color=FRESH_COLOR);  # jac:ignore[E1051,E1052]
    pyplot.bar([i + width / 2 for i in x], cptv2_vals, width, label="CPT-v2", color=CPTV2_COLOR);  # jac:ignore[E1051,E1052]
    for (i, v) in enumerate(fresh_vals) { pyplot.text(i - width / 2, v + 1, str(v) + "%", ha="center", fontsize=9); }  # jac:ignore[E1051]
    for (i, v) in enumerate(cptv2_vals) { pyplot.text(i + width / 2, v + 1, str(v) + "%", ha="center", fontsize=9); }  # jac:ignore[E1051]
    pyplot.xticks(x, stage_labels);  # jac:ignore[E1051]
    pyplot.ylabel("functional pass rate %");  # jac:ignore[E1051]
    pyplot.ylim(0, 90);  # jac:ignore[E1051]
    pyplot.title("Did CPT-v2 help? Full-budget comparison (both arms run to completion, no early stop)");  # jac:ignore[E1051]
    pyplot.legend();  # jac:ignore[E1051]
    pyplot.grid(True, axis="y", alpha=0.3);  # jac:ignore[E1051]
    pyplot.tight_layout();  # jac:ignore[E1051]
    pyplot.savefig(OUT_DIR + "/cpt_vs_fresh_full_budget.png", dpi=120);  # jac:ignore[E1051]
    pyplot.close();  # jac:ignore[E1051]
    print("wrote " + OUT_DIR + "/cpt_vs_fresh_full_budget.png");
}
