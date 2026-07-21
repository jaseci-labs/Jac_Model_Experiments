"""build_manifest_v2.jac -- assemble the unified sft_train.jsonl / dpo_train.jsonl (Task 17).

Reads every real generated source for this run-tag, excludes any record
whose `id` was carved into a holdout by `holdout_v2.jac`, and writes the
two release deliverables:

    <run_tag>/releases/sft_train.jsonl   -- all SFT categories, one file
    <run_tag>/releases/dpo_train.jsonl   -- dpo_train_free.jsonl + dpo_train_llm.jsonl, unified

SFT sources (per `datagen/spec.md`):
  - the six new-pipeline categories: clean_dataset/{code_gen,debug,
    documentation,explanation,migration,trajectory}/examples.jsonl
  - `conversion`, folded from two places per spec.md §4:
      * shared/graph_conversion_growth.jsonl (the new graph tier,
        `task_type=python_to_jac_graph`, already tagged on the record)
      * the existing (pre-jacgen2) function tier, reused as-is:
        model-experiments/01-sft-dpo/dataset/conversion/{sft.jsonl,sft_auto.jsonl}
        if that path exists -- read gracefully, not required. These
        records predate the v2 schema (no `id`/`category`/`task_type`
        fields, just `{messages, meta}`), so they're normalized into the
        v2 shape here with a content-hashed id and
        `task_type=python_to_jac_function` (the transpile-tier majority,
        per spec.md §4).
  Conversion deliberately carries NO holdout exclusion of its own --
  `holdout_v2.jac` does not carve conversion (it has its own separate
  holdout mechanisms, see that module's docstring); the exclusion filter
  below still runs over conversion records for architectural uniformity,
  it is just a no-op there in practice.

DPO sources: releases/dpo_train_free.jsonl + releases/dpo_train_llm.jsonl
(Task 15's internal `_free`/`_llm` staging split), unified into one
`dpo_train.jsonl` -- the filename `spec.md`/`dpo-plan.md` actually expect.

Idempotent-safe by FULL REBUILD, not append: every run reads all source
files fresh and overwrites `sft_train.jsonl`/`dpo_train.jsonl` from
scratch. This is correct here (unlike the generators' append-only
resumability pattern) because this module is a deterministic derived view
over already-generated content, not a place new content accumulates.

Run (from repo root, jacgen convention -- AFTER holdout_v2.jac ideally,
but tolerates it not having run yet):
    JAC_RUN_TAG=fresh /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/build_manifest_v2.jac
"""

import json;
import hashlib;
import from pathlib { Path }
import from common { require_run_tag, data_root }

glob SFT_CATEGORIES: list[str] = [
    "code_gen", "debug", "documentation", "explanation", "migration", "trajectory"
];
glob LEGACY_CONVERSION_DIR: str = "model-experiments/01-sft-dpo/dataset/conversion";
glob GRAPH_CONVERSION_PATH: str = "model-experiments/04-cpt-sft/dataset/shared/graph_conversion_growth.jsonl";

# SHARED, non-tag-scoped -- must match holdout_v2.jac's HOLDOUT_SHARED_DIR
# exactly (fix, follow-up to Task 17: holdouts are carved once from the
# fresh build and shared across both tags, per spec.md §5 / workflow.md §5).
glob HOLDOUT_MANIFEST_PATH: str = "model-experiments/04-cpt-sft/dataset/shared/holdouts/holdout_ids.json";

# Legacy-conversion schema-backfill constants (Finding 2 fix): these
# records predate the v2 pipeline entirely -- distinct, clearly-labeled
# "legacy" placeholders, never confused with the new pipeline's
# prompt-v2.0/jac-context-v2/claude-opus-4-*/claude-fable-5 constants
# (see gate.jac's PROMPT_VER/CONTEXT_BUNDLE_VER/VALIDATOR_VER/DATASET_VER).
glob LEGACY_GENERATOR_MODEL_ID: str = "legacy-conversion-pipeline";
glob LEGACY_GENERATION_DATE: str = "unknown-pre-v2-pipeline";  # exact date not tracked; not a fabricated timestamp
glob LEGACY_VERSION_TAG: str = "legacy-pre-v2";

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

def write_jsonl(path: str, records: list) {
    p = Path(path);
    p.parent.mkdir(parents=True, exist_ok=True);
    with open(path, "w") as f {
        for rec in records {
            f.write(json.dumps(rec) + "\n");
        }
    }
}

def load_holdout_exclusions(tag: str) -> dict {
    manifest_path: str = HOLDOUT_MANIFEST_PATH;
    p = Path(manifest_path);
    excl: dict = {};
    if not p.exists() {
        raise RuntimeError(
            "build_manifest_v2.jac hard-fail: shared holdout manifest not found at "
            + manifest_path + ". This manifest is shared (non-tag-scoped) and is "
            + "expected to always exist by the time this script runs, for BOTH "
            + "fresh and post_cptv2 builds -- it must be carved once, under "
            + "JAC_RUN_TAG=fresh, by holdout_v2.jac, before build_manifest_v2.jac "
            + "ever runs (see spec.md §5 / workflow.md §5 'Holdout rules'). A "
            + "missing manifest here is a real pipeline-ordering error, not "
            + "something to paper over with zero exclusions -- run: "
            + "JAC_RUN_TAG=fresh " + "/Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac "
            + "run model-experiments/01-sft-dpo/sft_dpo/jacgen2/holdout_v2.jac first."
        );
    }
    manifest: dict = dict(json.loads(p.read_text()));
    for cat in manifest {
        for id_ in manifest[cat] { excl[str(id_)] = True; }
    }
    print("loaded " + str(len(excl)) + " holdout ids to exclude from " + manifest_path);
    return excl;
}

"""Backfill the full v2 schema (spec.md §6, 17 fields) onto a legacy
(pre-v2-pipeline) conversion record. These records genuinely come from
a different, already-existing generator (the old jacgen/ pipeline's
`jac run` + exact-output-match gate -- confirmed real via
compiler_pass=True/behavioral_pass=True on every sampled legacy
record), so they get honestly-labeled LEGACY placeholders, distinct
from the new pipeline's prompt-v2.0/jac-context-v2/claude-opus-4-*
constants (gate.jac's make_example_v2) -- never pretending these are
new-pipeline gaps."""
def normalize_legacy_conversion(rec: dict, task_type: str, id_prefix: str, run_tag: str) -> dict {
    meta: dict = dict(rec.get("meta", {}));
    messages: list = list(rec.get("messages", []));
    basis: str = json.dumps(messages, sort_keys=True);
    content_hash: str = hashlib.md5(basis.encode()).hexdigest()[:12];
    rid: str = id_prefix + "_" + content_hash;
    return {
        "id": rid,
        "category": "conversion",
        "task_type": task_type,
        "complexity": meta.get("difficulty", "unknown"),
        "compiler_pass": meta.get("compiler_pass", None),
        "test_pass": meta.get("behavioral_pass", None),
        "manually_reviewed": False,
        "generator": meta.get("generator", "unknown"),
        "generator_model_id": LEGACY_GENERATOR_MODEL_ID,
        "gate_class": "behavioral",
        "variant_idx": 0,
        "generation_date": LEGACY_GENERATION_DATE,
        "source_prompt_version": LEGACY_VERSION_TAG,
        "context_bundle_version": LEGACY_VERSION_TAG,
        "validator_version": LEGACY_VERSION_TAG,
        "dataset_version": LEGACY_VERSION_TAG,
        "run_tag": run_tag,
        "seed_id": "legacy_" + content_hash,
        "seed_source": meta.get("source", "unknown"),
        "seed_tier": 0,
        "messages": messages
    };
}

with entry:__main__ {
    tag: str = require_run_tag();
    base: str = data_root(tag);

    print("=== build_manifest_v2: assembling sft_train.jsonl / dpo_train.jsonl (Task 17) ===");
    print("run_tag=" + tag);

    exclude_ids: dict = load_holdout_exclusions(tag);

    sft_records: list = [];
    per_category_counts: dict = {};
    per_category_excluded: dict = {};

    # -- the six new-pipeline SFT categories --------------------------------
    for cat in SFT_CATEGORIES {
        src_path: str = base + "/clean_dataset/" + cat + "/examples.jsonl";
        records: list[dict] = load_jsonl(src_path);
        kept: int = 0;
        excluded: int = 0;
        for rec in records {
            if str(rec.get("id", "")) in exclude_ids {
                excluded += 1;
            } else {
                sft_records.append(rec);
                kept += 1;
            }
        }
        per_category_counts[cat] = kept;
        per_category_excluded[cat] = excluded;
        print(cat + ": " + str(kept) + " kept, " + str(excluded) + " excluded (holdout), from " + src_path);
    }

    # -- conversion: graph tier (new, jacgen2) -------------------------------
    graph_records: list[dict] = load_jsonl(GRAPH_CONVERSION_PATH);
    conv_kept: int = 0;
    conv_excluded: int = 0;
    for rec in graph_records {
        if str(rec.get("id", "")) in exclude_ids {
            conv_excluded += 1;
        } else {
            sft_records.append(rec);
            conv_kept += 1;
        }
    }
    print("conversion (graph, " + GRAPH_CONVERSION_PATH + "): " + str(conv_kept) + " kept, " + str(conv_excluded) + " excluded");

    # -- conversion: legacy function tier (existing jacgen/ pipeline, reused) ---
    legacy_kept: int = 0;
    legacy_files: list = [
        (LEGACY_CONVERSION_DIR + "/sft.jsonl", "sft"),
        (LEGACY_CONVERSION_DIR + "/sft_auto.jsonl", "sftauto"),
    ];
    for pair in legacy_files {
        path: str = pair[0];
        prefix: str = pair[1];
        if not Path(path).exists() {
            print("conversion (legacy " + prefix + "): " + path + " not found -- skipping gracefully");
            continue;
        }
        raw: list[dict] = load_jsonl(path);
        for rec in raw {
            norm: dict = normalize_legacy_conversion(rec, "python_to_jac_function", "convtier_" + prefix, tag);
            if norm["id"] in exclude_ids {
                continue;
            }
            sft_records.append(norm);
            legacy_kept += 1;
        }
        print("conversion (legacy " + prefix + "): " + str(len(raw)) + " read, folded in from " + path);
    }
    per_category_counts["conversion"] = conv_kept + legacy_kept;
    per_category_excluded["conversion"] = conv_excluded;

    sft_out: str = base + "/releases/sft_train.jsonl";
    write_jsonl(sft_out, sft_records);
    print("sft_train.jsonl written: " + str(len(sft_records)) + " total records -> " + sft_out);

    # -- DPO: unify the two staging files ------------------------------------
    dpo_sources: list = [
        base + "/releases/dpo_train_free.jsonl",
        base + "/releases/dpo_train_llm.jsonl",
    ];
    dpo_records: list = [];
    dpo_excluded: int = 0;
    per_axis_counts: dict = {};
    for path in dpo_sources {
        raw: list[dict] = load_jsonl(path);
        for rec in raw {
            if str(rec.get("id", "")) in exclude_ids {
                dpo_excluded += 1;
                continue;
            }
            dpo_records.append(rec);
            axis: str = str(rec.get("axis", "unknown"));
            per_axis_counts[axis] = per_axis_counts.get(axis, 0) + 1;
        }
        print(path + ": " + str(len(raw)) + " read");
    }

    dpo_out: str = base + "/releases/dpo_train.jsonl";
    write_jsonl(dpo_out, dpo_records);
    print("dpo_train.jsonl written: " + str(len(dpo_records)) + " total pairs -> " + dpo_out);

    print("=== SUMMARY ===");
    print("SFT by category (post-exclusion): " + str(per_category_counts));
    print("SFT holdout-excluded by category: " + str(per_category_excluded));
    print("SFT total: " + str(len(sft_records)));
    print("DPO by axis: " + str(per_axis_counts));
    print("DPO holdout-excluded: " + str(dpo_excluded));
    print("DPO total: " + str(len(dpo_records)));
}
