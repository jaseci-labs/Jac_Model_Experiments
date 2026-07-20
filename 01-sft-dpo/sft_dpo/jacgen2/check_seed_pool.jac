"""
Self-check for seed_pool.jac's output: seed_pool.jsonl is non-empty, every
record has all required keys, no duplicate seed_ids, and at least 1 seed
exists from each of tiers 1/2/3/5.

Run after seed_pool.jac:
    JAC_RUN_TAG=fresh jac run check_seed_pool.jac
"""

import json;
import from pathlib { Path }

glob SEED_POOL_PATH: str = "/Volumes/ExtremePro/JaseciLabs/jac_model_studio/model-experiments/04-cpt-sft/dataset/shared/seed_pool.jsonl";
glob REQUIRED_FIELDS: list[str] = [
    "seed_id", "seed_tier", "origin", "jac_code", "gate_class",
    "expected_output", "domains", "doc_chunk"
];

with entry {
    path: Path = Path(SEED_POOL_PATH);
    assert path.exists(), "seed_pool.jsonl does not exist at " + SEED_POOL_PATH;

    with open(SEED_POOL_PATH, "r") as f {
        lines: list[str] = f.readlines();
    }
    records: list = [];
    for line in lines {
        s: str = line.strip();
        if s {
            records.append(json.loads(s));
        }
    }

    assert len(records) > 0, "seed_pool.jsonl is empty";

    seen_ids: dict[str, bool] = {};
    tiers_present: dict[int, int] = {};
    for rec in records {
        for field in REQUIRED_FIELDS {
            assert field in rec, "record missing field '" + field + "': " + str(rec.get("seed_id", "?"));
        }
        assert rec["gate_class"] in ["behavioral", "compile_only"], "bad gate_class: " + str(rec["gate_class"]);
        if rec["gate_class"] == "behavioral" {
            assert rec["expected_output"] is not None, "behavioral seed with null expected_output: " + str(rec["seed_id"]);
        }

        sid: str = str(rec["seed_id"]);
        assert sid not in seen_ids, "duplicate seed_id: " + sid;
        seen_ids[sid] = True;

        tier: int = int(rec["seed_tier"]);
        tiers_present[tier] = tiers_present.get(tier, 0) + 1;
    }

    for expected_tier in [1, 2, 3, 5] {
        count: int = tiers_present.get(expected_tier, 0);
        assert count >= 1, "no seeds found from tier " + str(expected_tier) + " -- investigate before treating this as genuine absence";
    }

    print("check_seed_pool: " + str(len(records)) + " records, " + str(len(seen_ids)) + " unique seed_ids");
    for tier in sorted(tiers_present.keys()) {
        print("  tier " + str(tier) + ": " + str(tiers_present[tier]) + " seeds");
    }
    print("all seed_pool.jac checks passed");
}
