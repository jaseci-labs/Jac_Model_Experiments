"""
Freeze/verify the shared seed-pool inputs by sha256, per spec.md (04-cpt-sft)
section 8.6 point 5's "frozen after the fresh build" rule.

Two modes via JAC_VERIFY_POOL:
  - unset / not "1" -> freeze mode: compute hashes, write pool_hashes.json.
  - "1"              -> verify mode: recompute hashes, compare against the
                         recorded pool_hashes.json, hard-fail on any mismatch
                         or missing file.

JAC_RUN_TAG is required (same guard every jacgen2 entrypoint uses) even
though the hashed paths are not run-tag-scoped -- it's a safety rail against
running pipeline scripts with no tag set at all.
"""

import os;
import json;
import hashlib;
import from pathlib { Path }
import from datetime { datetime }
import from common { require_run_tag }

glob POOL_FILES: list = [
    "model-experiments/04-cpt-sft/dataset/shared/seed_pool.jsonl",
    "model-experiments/04-cpt-sft/dataset/shared/doc_chunks.jsonl",
    "model-experiments/01-sft-dpo/sft_dpo/jacgen2/data/deprecated_inventory.jsonl",
    "model-experiments/01-sft-dpo/sft_dpo/jacgen2/data/app_ideas.jsonl",
    "model-experiments/04-cpt-sft/dataset/shared/conversion_slice.jsonl"
];

glob HASHES_PATH: str = "model-experiments/04-cpt-sft/dataset/shared/pool_hashes.json";


def sha256_file(path: str) -> str {
    h = hashlib.sha256();
    with open(path, "rb") as f {
        for chunk in iter(lambda: f.read(65536), b"") {
            h.update(chunk);
        }
    }
    return h.hexdigest();
}


def compute_pool_hashes() -> dict {
    hashes: dict = {};
    for rel_path in POOL_FILES {
        if Path(rel_path).exists() {
            name: str = Path(rel_path).name;
            hashes[name] = sha256_file(rel_path);
        }
    }
    return hashes;
}


def freeze(hashes: dict) {
    Path(HASHES_PATH).parent.mkdir(parents=True, exist_ok=True);
    record: dict = {
        "hashes": hashes,
        "frozen_at": datetime.now().isoformat()
    };
    with open(HASHES_PATH, "w") as f {
        f.write(json.dumps(record, indent=2));
    }
    print("froze " + str(len(hashes)) + " file hashes to " + HASHES_PATH);
    for name in hashes {
        print("  " + name + ": " + str(hashes[name]));
    }
}


def verify(hashes: dict) {
    if not Path(HASHES_PATH).exists() {
        raise RuntimeError("verify mode: no frozen " + HASHES_PATH + " found -- run freeze mode first");
    }
    with open(HASHES_PATH, "r") as f {
        recorded: dict = json.load(f)["hashes"];
    }
    problems: list = [];
    for name in recorded {
        if name not in hashes {
            problems.append(name + ": missing (was frozen, now absent)");
        } elif hashes[name] != recorded[name] {
            problems.append(name + ": hash changed (frozen=" + str(recorded[name])[:12] + "..., now=" + str(hashes[name])[:12] + "...)");
        }
    }
    for name in hashes {
        if name not in recorded {
            problems.append(name + ": new file not in frozen record");
        }
    }
    if problems {
        raise RuntimeError("pool verification FAILED, changed/missing file(s): " + "; ".join(problems));
    }
    print("pool verification OK -- all " + str(len(recorded)) + " frozen hashes match");
}


with entry:__main__ {
    require_run_tag();
    hashes: dict = compute_pool_hashes();
    verify_mode: bool = os.environ.get("JAC_VERIFY_POOL") == "1";
    if verify_mode {
        verify(hashes);
    } else {
        freeze(hashes);
    }
}
