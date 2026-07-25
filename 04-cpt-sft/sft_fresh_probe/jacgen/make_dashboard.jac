"""Final 3-way comparison: base (no CPT) vs +SFT vs +SFT+DPO, fresh probe.

Mirrors sft_cptv2_probe/jacgen/make_dashboard.jac exactly, repointed at this
probe's own metrics files.
"""

import json;
import os;
import matplotlib;
import from matplotlib { pyplot }
import from pathlib { Path }

glob SFT_METRICS: str = "model-experiments/04-cpt-sft/sft_fresh_probe/results/sft/metrics_functional.jsonl";
glob DPO_METRICS: str = "model-experiments/04-cpt-sft/sft_fresh_probe/results/dpo/metrics_functional.jsonl";
glob OUT_DIR: str = "model-experiments/04-cpt-sft/sft_fresh_probe/results/images";


def load_rows(path: str) -> list[dict] {
    rows: list[dict] = [];
    p = Path(path);
    if not p.exists() { return rows; }
    for line in p.read_text().split("\n") {
        if line.strip().startswith("{") { rows.append(dict(json.loads(line))); }
    }
    return rows;
}


def category_breakdown(rows: list[dict], step: int, shape: dict) -> dict {
    out: dict = {};
    for r in rows {
        if r["category"] == "__overall__" { continue; }
        if r["step"] != step { continue; }
        key: tuple = (r["category"], r["gate_class"]);
        if key in shape and shape[key] != r["total"] { continue; }
        cat: str = str(r["category"]);
        if cat not in out { out[cat] = {"total": 0, "runs": 0}; }
        out[cat]["total"] += int(r["total"]);
        out[cat]["runs"] += int(r["runs"]);
    }
    return out;
}


def base_shape(rows: list[dict], step: int) -> dict {
    shape: dict = {};
    for r in rows {
        if r["category"] == "__overall__" { continue; }
        if r["step"] != step { continue; }
        shape[(r["category"], r["gate_class"])] = int(r["total"]);
    }
    return shape;
}


with entry {
    matplotlib.use("Agg");
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True);

    sft_rows: list[dict] = load_rows(SFT_METRICS);
    dpo_rows: list[dict] = load_rows(DPO_METRICS);

    base_overall: dict = {};
    sft_final_overall: dict = {};
    max_sft_step: int = 0;
    for r in sft_rows {
        if r["category"] != "__overall__" { continue; }
        if r["step"] == 0 { base_overall = r; }
        if r["step"] >= max_sft_step { max_sft_step = r["step"]; sft_final_overall = r; }
    }
    dpo_final_overall: dict = {};
    max_dpo_step: int = -1;
    for r in dpo_rows {
        if r["category"] != "__overall__" { continue; }
        if r["step"] >= max_dpo_step { max_dpo_step = r["step"]; dpo_final_overall = r; }
    }

    labels: list[str] = ["base (no CPT)", "+SFT"];
    values: list[float] = [float(base_overall.get("runs_pct", 0)), float(sft_final_overall.get("runs_pct", 0))];
    if len(dpo_final_overall) > 0 {
        labels.append("+SFT+DPO");
        values.append(float(dpo_final_overall.get("runs_pct", 0)));
    } else {
        print("NOTE: no DPO metrics_functional.jsonl found -- run Task 9 first.");
    }

    pyplot.figure(figsize=(7, 5));  # jac:ignore[E1051]
    pyplot.bar(labels, values, color=["#888888", "#4C72B0", "#C44E52"][:len(labels)]);  # jac:ignore[E1051]
    pyplot.title("Functional pass rate (runs %) -- fresh-base probe");  # jac:ignore[E1051]
    pyplot.ylabel("runs %");  # jac:ignore[E1051]
    pyplot.ylim(0, 100);  # jac:ignore[E1051]
    for (i, v) in enumerate(values) { pyplot.text(i, v + 1, str(round(v, 1)) + "%", ha="center"); }  # jac:ignore[E1051]
    pyplot.grid(True, axis="y");  # jac:ignore[E1051]
    pyplot.savefig(OUT_DIR + "/comparison_overall.png");  # jac:ignore[E1051]
    pyplot.close();  # jac:ignore[E1051]
    print("wrote " + OUT_DIR + "/comparison_overall.png");
    print(json.dumps({"base": base_overall, "sft_final": sft_final_overall, "dpo_final": dpo_final_overall}, indent=2));

    shape: dict = base_shape(sft_rows, 0);
    base_cats: dict = category_breakdown(sft_rows, 0, shape);
    sft_cats: dict = category_breakdown(sft_rows, max_sft_step, shape);
    dpo_cats: dict = category_breakdown(dpo_rows, max_dpo_step, shape);

    cat_names: list[str] = sorted(list(base_cats.keys()));
    base_pcts: list[float] = [];
    sft_pcts: list[float] = [];
    dpo_pcts: list[float] = [];
    for cat in cat_names {
        b: dict = base_cats.get(cat, {"total": 0, "runs": 0});
        s: dict = sft_cats.get(cat, {"total": 0, "runs": 0});
        d: dict = dpo_cats.get(cat, {"total": 0, "runs": 0});
        base_pcts.append(round(100.0 * b["runs"] / b["total"], 1) if b["total"] > 0 else 0.0);
        sft_pcts.append(round(100.0 * s["runs"] / s["total"], 1) if s["total"] > 0 else 0.0);
        dpo_pcts.append(round(100.0 * d["runs"] / d["total"], 1) if d["total"] > 0 else 0.0);
    }

    x: list[int] = list(range(len(cat_names)));
    width: float = 0.25;
    pyplot.figure(figsize=(11, 6));  # jac:ignore[E1051]
    pyplot.bar([i - width for i in x], base_pcts, width, label="base (no CPT)", color="#888888");  # jac:ignore[E1051]
    pyplot.bar(x, sft_pcts, width, label="+SFT", color="#4C72B0");  # jac:ignore[E1051]
    pyplot.bar([i + width for i in x], dpo_pcts, width, label="+SFT+DPO", color="#C44E52");  # jac:ignore[E1051]
    pyplot.xticks(x, cat_names, rotation=20);  # jac:ignore[E1051]
    pyplot.ylabel("runs %");  # jac:ignore[E1051]
    pyplot.ylim(0, 100);  # jac:ignore[E1051]
    pyplot.title("Functional pass rate by category -- fresh-base probe");  # jac:ignore[E1051]
    pyplot.legend();  # jac:ignore[E1051]
    pyplot.grid(True, axis="y");  # jac:ignore[E1051]
    pyplot.tight_layout();  # jac:ignore[E1051]
    pyplot.savefig(OUT_DIR + "/comparison_by_category.png");  # jac:ignore[E1051]
    pyplot.close();  # jac:ignore[E1051]
    print("wrote " + OUT_DIR + "/comparison_by_category.png");
    print(json.dumps({"categories": cat_names, "base": base_pcts, "sft": sft_pcts, "dpo": dpo_pcts}, indent=2));
}
