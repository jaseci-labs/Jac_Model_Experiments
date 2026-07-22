"""snapshot_conversion.jac -- one-time snapshot of the conversion category
slice, per spec.md (04-cpt-sft) section 8 rollout step 7: "Snapshot the
conversion slice + freeze/hash seed_pool.jsonl".

Concatenates the two existing (pre-jacgen2) function-tier files with the
new graph-tier growth file into one canonical, shared (non-tag-scoped)
snapshot -- same convention as seed_pool.jsonl / holdout_ids.json:

    model-experiments/01-sft-dpo/dataset/conversion/sft.jsonl        (function tier)
    model-experiments/01-sft-dpo/dataset/conversion/sft_auto.jsonl   (function tier)
    model-experiments/04-cpt-sft/dataset/shared/graph_conversion_growth.jsonl (graph tier)
    -> model-experiments/04-cpt-sft/dataset/shared/conversion_slice.jsonl

Each record gets a `task_type` field backfilled if missing (function-tier
records have none -- `python_to_jac_graph`-labeled records from Task 14
already carry their own `task_type` and are left untouched). Row count +
sha256 of the resulting file are merged into pool_hashes.json (Task 7's
freeze_pool.jac remains the ongoing source of truth for that file; this
script's write is just the initial record + idempotency marker).

Idempotent: a second run, once the recorded hash in pool_hashes.json
matches the file on disk, is a no-op -- this mirrors seed_pool.jsonl's
frozen-after-fresh-build convention (spec.md §8).

Set JAC_RESNAPSHOT_CONVERSION=1 to force a re-snapshot even when the
existing output's hash matches the recorded one -- needed when an input
source (e.g. graph_conversion_growth.jsonl) has grown and the snapshot
needs to pick up the new rows.

Run (from repo root):
    JAC_RUN_TAG=fresh /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/snapshot_conversion.jac
"""

import os;
import json;
import hashlib;
import from pathlib { Path }
import from common { require_run_tag }

glob FUNCTION_TIER_FILES: list[str] = [
    "model-experiments/01-sft-dpo/dataset/conversion/sft.jsonl",
    "model-experiments/01-sft-dpo/dataset/conversion/sft_auto.jsonl"
];
glob GRAPH_TIER_FILE: str = "model-experiments/04-cpt-sft/dataset/shared/graph_conversion_growth.jsonl";
glob OUT_PATH: str = "model-experiments/04-cpt-sft/dataset/shared/conversion_slice.jsonl";
glob HASHES_PATH: str = "model-experiments/04-cpt-sft/dataset/shared/pool_hashes.json";


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


def backfill_task_type(rec: dict, default_type: str) -> dict {
    if not rec.get("task_type") {
        rec["task_type"] = default_type;
    }
    return rec;
}


def sha256_file(path: str) -> str {
    h = hashlib.sha256();
    with open(path, "rb") as f {
        for chunk in iter(lambda: f.read(65536), b"") {
            h.update(chunk);
        }
    }
    return h.hexdigest();
}


def already_snapshotted(out_path: str, hashes_path: str) -> bool {
    if not Path(out_path).exists() { return False; }
    if not Path(hashes_path).exists() { return False; }
    recorded: dict = dict(json.loads(Path(hashes_path).read_text())).get("hashes", {});
    if "conversion_slice.jsonl" not in recorded { return False; }
    return recorded["conversion_slice.jsonl"] == sha256_file(out_path);
}


def merge_hash(hashes_path: str, file_hash: str) {
    record: dict = {"hashes": {}, "frozen_at": None};
    if Path(hashes_path).exists() {
        record = dict(json.loads(Path(hashes_path).read_text()));
    }
    hashes: dict = dict(record.get("hashes", {}));
    hashes["conversion_slice.jsonl"] = file_hash;
    record["hashes"] = hashes;
    Path(hashes_path).parent.mkdir(parents=True, exist_ok=True);
    with open(hashes_path, "w") as f {
        f.write(json.dumps(record, indent=2));
    }
}


with entry:__main__ {
    require_run_tag();

    force_resnapshot: bool = os.environ.get("JAC_RESNAPSHOT_CONVERSION") == "1";
    if (not force_resnapshot) and already_snapshotted(OUT_PATH, HASHES_PATH) {
        print("conversion_slice.jsonl already snapshotted and hash-verified at " + OUT_PATH + " -- no-op");
    } else {
        records: list[dict] = [];
        for path in FUNCTION_TIER_FILES {
            raw: list[dict] = load_jsonl(path);
            for rec in raw {
                records.append(backfill_task_type(rec, "python_to_jac_function"));
            }
            print(path + ": " + str(len(raw)) + " records (task_type=python_to_jac_function)");
        }
        graph_raw: list[dict] = load_jsonl(GRAPH_TIER_FILE);
        for rec in graph_raw {
            records.append(backfill_task_type(rec, "python_to_jac_graph"));
        }
        print(GRAPH_TIER_FILE + ": " + str(len(graph_raw)) + " records (task_type preserved if present)");

        Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True);
        with open(OUT_PATH, "w") as f {
            for rec in records {
                f.write(json.dumps(rec) + "\n");
            }
        }
        print("conversion_slice.jsonl written: " + str(len(records)) + " total records -> " + OUT_PATH);

        file_hash: str = sha256_file(OUT_PATH);
        merge_hash(HASHES_PATH, file_hash);
        print("conversion_slice.jsonl: " + str(len(records)) + " rows, sha256=" + file_hash);
        print("merged conversion_slice.jsonl hash into " + HASHES_PATH);
    }
}
