"""Cross-arm comparison: does CPT-v2 help once SFT/DPO trains on top of it?
Reads both probes' metrics_functional.jsonl (both already share the identical
1428-row holdout -- verified in sft_fresh_probe's Task 1 via MD5) and computes
the CPT delta at each stage.
"""

import json;
import os;
import matplotlib;
import from matplotlib { pyplot }
import from pathlib { Path }

glob CPTV2_SFT_METRICS: str = "model-experiments/04-cpt-sft/sft_cptv2_probe/results/sft/metrics_functional.jsonl";
glob CPTV2_DPO_METRICS: str = "model-experiments/04-cpt-sft/sft_cptv2_probe/results/dpo/metrics_functional.jsonl";
glob FRESH_SFT_METRICS: str = "model-experiments/04-cpt-sft/sft_fresh_probe/results/sft/metrics_functional.jsonl";
glob FRESH_DPO_METRICS: str = "model-experiments/04-cpt-sft/sft_fresh_probe/results/dpo/metrics_functional.jsonl";
glob OUT_DIR: str = "model-experiments/04-cpt-sft/results/images";


def load_rows(path: str) -> list[dict] {
    rows: list[dict] = [];
    p = Path(path);
    if not p.exists() { return rows; }
    for line in p.read_text().split("\n") {
        if line.strip().startswith("{") { rows.append(dict(json.loads(line))); }
    }
    return rows;
}


def overall_stages(rows: list[dict]) -> dict {
    base: dict = {}; final: dict = {}; max_step: int = 0;
    for r in rows {
        if r["category"] != "__overall__" { continue; }
        if r["step"] == 0 { base = r; }
        if r["step"] >= max_step { max_step = r["step"]; final = r; }
    }
    return {"base": base, "final": final};
}


def dpo_overall(rows: list[dict]) -> dict {
    out: dict = {}; max_step: int = -1;
    for r in rows {
        if r["category"] != "__overall__" { continue; }
        if r["step"] >= max_step { max_step = r["step"]; out = r; }
    }
    return out;
}


with entry {
    matplotlib.use("Agg");
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True);

    cptv2_sft: dict = overall_stages(load_rows(CPTV2_SFT_METRICS));
    fresh_sft: dict = overall_stages(load_rows(FRESH_SFT_METRICS));
    cptv2_dpo: dict = dpo_overall(load_rows(CPTV2_DPO_METRICS));
    fresh_dpo: dict = dpo_overall(load_rows(FRESH_DPO_METRICS));

    cptv2_base_pct: float = float(cptv2_sft["base"].get("runs_pct", 0));
    fresh_base_pct: float = float(fresh_sft["base"].get("runs_pct", 0));
    cptv2_sft_pct: float = float(cptv2_sft["final"].get("runs_pct", 0));
    fresh_sft_pct: float = float(fresh_sft["final"].get("runs_pct", 0));
    cptv2_dpo_pct: float = float(cptv2_dpo.get("runs_pct", 0));
    fresh_dpo_pct: float = float(fresh_dpo.get("runs_pct", 0));

    labels: list[str] = ["base", "+SFT", "+SFT+DPO"];
    fresh_vals: list[float] = [fresh_base_pct, fresh_sft_pct, fresh_dpo_pct];
    cptv2_vals: list[float] = [cptv2_base_pct, cptv2_sft_pct, cptv2_dpo_pct];

    x: list[int] = list(range(len(labels)));
    width: float = 0.35;
    pyplot.figure(figsize=(9, 6));  # jac:ignore[E1051]
    pyplot.bar([i - width / 2 for i in x], fresh_vals, width, label="fresh base (no CPT)", color="#4C72B0");  # jac:ignore[E1051]
    pyplot.bar([i + width / 2 for i in x], cptv2_vals, width, label="CPT-v2 base", color="#C44E52");  # jac:ignore[E1051]
    for (i, v) in enumerate(fresh_vals) { pyplot.text(i - width / 2, v + 1, str(round(v, 1)) + "%", ha="center"); }  # jac:ignore[E1051]
    for (i, v) in enumerate(cptv2_vals) { pyplot.text(i + width / 2, v + 1, str(round(v, 1)) + "%", ha="center"); }  # jac:ignore[E1051]
    pyplot.xticks(x, labels);  # jac:ignore[E1051]
    pyplot.ylabel("runs %");  # jac:ignore[E1051]
    pyplot.ylim(0, 100);  # jac:ignore[E1051]
    pyplot.title("Did CPT-v2 help? fresh-base vs CPT-v2-base, same SFT/DPO recipe+data");  # jac:ignore[E1051]
    pyplot.legend();  # jac:ignore[E1051]
    pyplot.grid(True, axis="y");  # jac:ignore[E1051]
    pyplot.savefig(OUT_DIR + "/cpt_vs_fresh_overall.png");  # jac:ignore[E1051]
    pyplot.close();  # jac:ignore[E1051]
    print("wrote " + OUT_DIR + "/cpt_vs_fresh_overall.png");

    deltas: dict = {
        "base_delta_cptv2_minus_fresh": round(cptv2_base_pct - fresh_base_pct, 2),
        "sft_delta_cptv2_minus_fresh": round(cptv2_sft_pct - fresh_sft_pct, 2),
        "dpo_delta_cptv2_minus_fresh": round(cptv2_dpo_pct - fresh_dpo_pct, 2),
        "fresh_base_pct": fresh_base_pct, "cptv2_base_pct": cptv2_base_pct,
        "fresh_sft_pct": fresh_sft_pct, "cptv2_sft_pct": cptv2_sft_pct,
        "fresh_dpo_pct": fresh_dpo_pct, "cptv2_dpo_pct": cptv2_dpo_pct,
    };
    print(json.dumps(deltas, indent=2));
    Path(OUT_DIR + "/../cpt_vs_fresh_deltas.json").write_text(json.dumps(deltas, indent=2));
}
