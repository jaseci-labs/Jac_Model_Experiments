"""
Shared helpers for the jacgen2 pipeline: run-tag guard, path helpers,
example-key derivation, .env loading, and resumability (existing_keys).
"""

import os;
import json;
import from pathlib { Path }

def require_run_tag() -> str {
    tag = os.environ.get("JAC_RUN_TAG");
    valid_tags = ["fresh", "post_cptv2"];
    if not tag or tag not in valid_tags {
        raise ValueError("JAC_RUN_TAG must be set to one of " + str(valid_tags) + ", got: " + str(tag));
    }
    return tag;
}

def data_root(tag: str) -> str {
    return "model-experiments/04-cpt-sft/dataset/" + tag;
}

def out_path(tag: str, area: str, category: str) -> str {
    return data_root(tag) + "/" + area + "/" + category;
}

def example_key(seed_id: str, task_type: str, variant_idx: int) -> str {
    return seed_id + "__" + task_type + "__" + str(variant_idx);
}

def load_env() {
    env_path = Path(__file__).parent / ".env";
    if env_path.exists() {
        with open(env_path, 'r') as f {
            lines = f.readlines();
        }
        for line in lines {
            s = line.strip();
            if s and not s.startswith('#') and '=' in s {
                parts = s.split('=', 1);
                os.environ.setdefault(parts[0].strip(), parts[1].strip());
            }
        }
    }
}

def append_jsonl(path: str, rec: dict) {
    p = Path(path);
    p.parent.mkdir(parents=True, exist_ok=True);
    with open(path, "a") as f {
        f.write(json.dumps(rec) + "\n");
    }
}

def load_model_name(role: str) -> str {
    models_path = Path(__file__).parent / "models.json";
    with open(models_path, 'r') as f {
        data = json.load(f);
    }
    return data[role]["model_name"];
}

def existing_keys(jsonl_path: str) -> dict {
    keys = {};
    path = Path(jsonl_path);
    if not path.exists() {
        return keys;
    }
    with open(path, 'r') as f {
        lines = f.readlines();
    }
    for line in lines {
        s = line.strip();
        if s {
            rec = json.loads(s);
            keys[rec["id"]] = True;
        }
    }
    return keys;
}
