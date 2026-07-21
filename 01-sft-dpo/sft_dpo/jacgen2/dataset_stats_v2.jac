"""dataset_stats_v2.jac -- composition report over the real sft_train.jsonl / dpo_train.jsonl (Task 17).

Reads `build_manifest_v2.jac`'s output (does not rebuild it -- run that
first; if its outputs are missing this script warns and reports zero
counts rather than crashing) and reports composition by category,
task_type, seed_tier, and generator, plus a weight-target comparison
against `datagen/spec.md`'s 36/16/10/10/10/6/4/8-buffer category table.

At the current real pilot scale the actual percentages will NOT match the
12,500-example production targets -- that is expected and reported
honestly here, not smoothed over.

Also looks for real logged Opus/Fable call data at
`<run_tag>/logs/generation/calls_<run_tag>.jsonl` (the convention already
in use by this build, see `common.jac` callers) and reports call counts /
success rate / prompt+response character volume per model if present.
No token counts are fabricated -- the real log only carries character
counts, not token counts, and that distinction is reported as-is.

Run (from repo root, jacgen convention -- AFTER build_manifest_v2.jac):
    JAC_RUN_TAG=fresh /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/dataset_stats_v2.jac
"""

import json;
import from pathlib { Path }
import from common { require_run_tag, data_root }

# datagen/spec.md category weight targets (target: 12,500 examples)
glob CATEGORY_TARGETS_PCT: dict = {
    "code_gen": 36, "debug": 16, "explanation": 10, "conversion": 10,
    "trajectory": 10, "documentation": 6, "migration": 4, "buffer": 8
};

def load_jsonl(path: str) -> list[dict] {
    p = Path(path);
    if not p.exists() { return []; }
    out: list[dict] = [];
    for line in p.read_text().split("\n") {
        s: str = line.strip();
        if s { out.append(dict(json.loads(s))); }
    }
    return out;
}

def bucket_count(records: list, field: str) -> dict {
    counts: dict = {};
    for rec in records {
        key: str = str(rec.get(field, "unknown"));
        counts[key] = counts.get(key, 0) + 1;
    }
    return counts;
}

def print_counts(label: str, counts: dict) {
    print(label + ":");
    for k in sorted(counts.keys()) {
        print("  " + str(k) + ": " + str(counts[k]));
    }
}

with entry:__main__ {
    tag: str = require_run_tag();
    base: str = data_root(tag);

    print("=== dataset_stats_v2: composition report (Task 17) ===");
    print("run_tag=" + tag);

    sft_path: str = base + "/releases/sft_train.jsonl";
    dpo_path: str = base + "/releases/dpo_train.jsonl";

    if not Path(sft_path).exists() {
        print("WARNING: " + sft_path + " not found -- run build_manifest_v2.jac first. Reporting zero counts.");
    }
    if not Path(dpo_path).exists() {
        print("WARNING: " + dpo_path + " not found -- run build_manifest_v2.jac first. Reporting zero counts.");
    }

    sft_records: list[dict] = load_jsonl(sft_path);
    dpo_records: list[dict] = load_jsonl(dpo_path);

    by_category: dict = bucket_count(sft_records, "category");
    by_task_type: dict = bucket_count(sft_records, "task_type");
    by_seed_tier: dict = bucket_count(sft_records, "seed_tier");
    by_generator: dict = bucket_count(sft_records, "generator");
    dpo_by_axis: dict = bucket_count(dpo_records, "axis");

    total_sft: int = len(sft_records);
    total_dpo: int = len(dpo_records);

    print("--- SFT composition (total=" + str(total_sft) + ") ---");
    print_counts("by category", by_category);
    print_counts("by task_type", by_task_type);
    print_counts("by seed_tier", by_seed_tier);
    print_counts("by generator", by_generator);

    print("--- DPO composition (total=" + str(total_dpo) + ") ---");
    print_counts("by axis", dpo_by_axis);

    print("--- weight-target comparison (spec.md category table, target @ 12,500) ---");
    weight_comparison: dict = {};
    for cat in CATEGORY_TARGETS_PCT {
        target_pct: int = CATEGORY_TARGETS_PCT[cat];
        actual_n: int = by_category.get(cat, 0);
        actual_pct: float = (actual_n / total_sft * 100.0) if total_sft > 0 else 0.0;
        weight_comparison[cat] = {"target_pct": target_pct, "actual_n": actual_n, "actual_pct": round(actual_pct, 2)};
        print(
            "  " + cat + ": target=" + str(target_pct) + "%  actual=" + str(actual_n)
            + " records (" + str(round(actual_pct, 2)) + "%)"
        );
    }
    print(
        "NOTE: at real pilot scale (~" + str(total_sft) + " total SFT records vs the "
        + "12,500-example production target) these percentages are not expected "
        + "to match spec.md's targets -- reported honestly, not smoothed over. "
        + "`conversion` in particular is dominated by the reused legacy "
        + "function-tier pool (model-experiments/01-sft-dpo/dataset/conversion/), "
        + "which already exceeds its 10% target on its own, per spec.md section 4."
    );

    # -- generation call/token logs, if any were actually written -----------
    log_path: str = base + "/logs/generation/calls_" + tag + ".jsonl";
    call_logs: list[dict] = load_jsonl(log_path);
    call_stats: dict = {};
    if len(call_logs) == 0 {
        print("--- generation call logs ---");
        print("no call/token logs found at " + log_path);
    } else {
        print("--- generation call logs (" + log_path + ") ---");
        per_model: dict = {};
        for rec in call_logs {
            model: str = str(rec.get("model", "unknown"));
            if model not in per_model {
                per_model[model] = {"calls": 0, "ok": 0, "failed": 0, "prompt_chars": 0, "resp_chars": 0};
            }
            per_model[model]["calls"] += 1;
            if rec.get("ok", False) {
                per_model[model]["ok"] += 1;
            } else {
                per_model[model]["failed"] += 1;
            }
            per_model[model]["prompt_chars"] += int(rec.get("prompt_chars", 0));
            per_model[model]["resp_chars"] += int(rec.get("resp_chars", 0));
        }
        for model in sorted(per_model.keys()) {
            stats: dict = per_model[model];
            print(
                "  " + model + ": " + str(stats["calls"]) + " calls (" + str(stats["ok"]) + " ok, "
                + str(stats["failed"]) + " failed), prompt_chars=" + str(stats["prompt_chars"])
                + ", resp_chars=" + str(stats["resp_chars"])
                + " -- character counts, NOT token counts (log does not carry token usage)"
            );
        }
        call_stats = per_model;
    }

    stats_report: dict = {
        "task": "dataset_stats_v2 (Task 17)",
        "run_tag": tag,
        "sft_total": total_sft,
        "sft_by_category": by_category,
        "sft_by_task_type": by_task_type,
        "sft_by_seed_tier": by_seed_tier,
        "sft_by_generator": by_generator,
        "dpo_total": total_dpo,
        "dpo_by_axis": dpo_by_axis,
        "weight_target_comparison": weight_comparison,
        "generation_call_log_path": log_path,
        "generation_call_stats": call_stats,
        "pilot_scale_note": (
            "Real pilot-scale data (~" + str(total_sft) + " SFT records), not the "
            + "12,500-example production target. Percentages/targets above are "
            + "reported honestly at this scale."
        )
    };

    stats_out: str = base + "/logs/dataset_stats_v2_report.json";
    Path(stats_out).parent.mkdir(parents=True, exist_ok=True);
    with open(stats_out, "w") as f {
        f.write(json.dumps(stats_report, indent=2));
    }
    print("report written to " + stats_out);
}
