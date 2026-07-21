"""holdout_v2.jac -- carve per-category eval holdouts, once, fixed thereafter (Task 17).

Per `datagen/spec.md`/`workflow.md`, holdouts are carved ONCE from the
`fresh` build and reused unchanged to score every arm (Arm A/C/B) -- never
re-carved per run-tag. Categories carved here are the six that
`workflow.md` Panel 1/2 actually score off `holdout_v2.jac` slices:
`code_gen`, `debug`, `trajectory`, `migration` (Panel 1, voting),
`documentation` (Panel 1, voting), `explanation` (Panel 2, reported only).

`conversion` is deliberately NOT carved here -- it already has two separate,
pre-existing holdout mechanisms (`eval_probe.jac`'s 150-example function
holdout, `graph_holdout.jac`'s graph holdout grown to >=100), both outside
this pipeline's scope.

PILOT-SCALE NOTE: `workflow.md` targets >=100 holdout items per category.
At the current real generation scale (4-30 records per category -- these
are pilot runs of `gen_code_gen.jac` etc, not the full 12,500-example
build), that target is not reachable. What this script proves instead is
the CARVING MECHANISM: a deterministic, idempotent, per-category slice is
held back and its ids are recorded so `build_manifest_v2.jac` can exclude
them from training data. Reaching the real >=100/category target requires
the actual full-scale generation run (future work), not a change to this
script's logic -- the same code works correctly at any scale.

Selection is deterministic (last N records in file order, no RNG): holdout
count per category = round(15% of category size) via integer math,
minimum 1 once a category has >=3 records, 0 below that (never
holdout-starve a near-empty category).

Idempotent: if `holdouts/holdout_ids.json` already exists, this is a
no-op that reports the existing holdout instead of re-carving (holdouts
are carved once and fixed thereafter -- re-running must never
re-randomize or resize them).

Run (from repo root, jacgen convention):
    JAC_RUN_TAG=fresh /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/holdout_v2.jac
"""

import json;
import from pathlib { Path }
import from common { require_run_tag, data_root }

glob HOLDOUT_CATEGORIES: list[str] = [
    "code_gen", "debug", "documentation", "explanation", "migration", "trajectory"
];
glob HOLDOUT_FRACTION_PCT: int = 15;  # integer percent, avoids float rounding surprises

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

def holdout_count(n: int) -> int {
    if n < 3 { return 0; }
    h: int = (n * HOLDOUT_FRACTION_PCT + 50) // 100;
    if h < 1 { h = 1; }
    return h;
}

with entry:__main__ {
    tag: str = require_run_tag();
    base: str = data_root(tag);
    holdout_dir: str = base + "/holdouts";
    manifest_path: str = holdout_dir + "/holdout_ids.json";

    print("=== holdout_v2: per-category eval holdout carving (Task 17) ===");
    print("run_tag=" + tag);

    if Path(manifest_path).exists() {
        print(
            "holdout_ids.json already exists -- holdouts are carved ONCE and "
            + "fixed thereafter. This run is a no-op; reporting the EXISTING "
            + "holdout (not re-carving)."
        );
        existing: dict = dict(json.loads(Path(manifest_path).read_text()));
        for cat in HOLDOUT_CATEGORIES {
            ids: list = list(existing.get(cat, []));
            src_path: str = base + "/clean_dataset/" + cat + "/examples.jsonl";
            total: int = len(load_jsonl(src_path));
            print(cat + ": " + str(len(ids)) + " held out / " + str(total) + " total");
        }
        print("=== no-op complete: existing holdout reported above ===");
    } else {
        manifest: dict = {};
        total_all: int = 0;
        total_holdout: int = 0;
        for cat in HOLDOUT_CATEGORIES {
            src_path: str = base + "/clean_dataset/" + cat + "/examples.jsonl";
            records: list[dict] = load_jsonl(src_path);
            n: int = len(records);
            h: int = holdout_count(n);
            holdout_records: list = records[n - h:] if h > 0 else [];
            ids: list[str] = [];
            for rec in holdout_records { ids.append(str(rec["id"])); }
            manifest[cat] = ids;

            out_file: str = holdout_dir + "/" + cat + "_holdout.jsonl";
            write_jsonl(out_file, holdout_records);

            total_all += n;
            total_holdout += h;
            print(cat + ": " + str(h) + " held out / " + str(n) + " total -> " + out_file);
        }

        Path(holdout_dir).mkdir(parents=True, exist_ok=True);
        with open(manifest_path, "w") as f {
            f.write(json.dumps(manifest, indent=2));
        }
        print("holdout manifest written to " + manifest_path);
        print(
            "=== TOTAL: " + str(total_holdout) + " held out / " + str(total_all)
            + " real records across " + str(len(HOLDOUT_CATEGORIES)) + " categories ==="
        );
        print(
            "PILOT-SCALE HOLDOUT: proves the carving mechanism works correctly; "
            + "does not meet the production >=100/category target (see module "
            + "docstring / workflow.md Panel 1-2). Reaching that target needs "
            + "the full-scale generation run, not a change to this script."
        );
    }
}
