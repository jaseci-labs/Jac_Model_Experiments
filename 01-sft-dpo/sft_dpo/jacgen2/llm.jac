"""
Dual-model (opus/fable) LLM wrappers for the jacgen2 pipeline: retry with
backoff, hard call-budget enforcement, and JSONL call logging.

Model snapshot IDs come from models.json (never hardcoded); construction
follows check_llm.jac's proven `Model(model_name=...)` pattern (Task 1).
"""

import os;
import time;
import json;
import from pathlib { Path }
import from datetime { datetime }
import from byllm.lib { Model }
import from common { load_env, require_run_tag, data_root }

# NOTE: jacgen/writer.append_jsonl (old pipeline) is functionally identical
# to this, but a real cross-directory relative import (`..jacgen.writer`)
# type-checks fine yet fails at runtime here ("attempted relative import
# with no known parent package") -- jac run executes each entry script
# without real Python package context, and path segments like
# "01-sft-dpo" aren't valid Python identifiers anyway, so package-qualified
# absolute imports aren't an option either. Inlined instead (5 lines,
# stdlib-only, verbatim match of writer.append_jsonl).
def append_jsonl(path: str, rec: dict) {
    p = Path(path);
    p.parent.mkdir(parents=True, exist_ok=True);
    with open(path, "a") as f {
        f.write(json.dumps(rec) + "\n");
    }
}

def load_model_name(key: str) -> str {
    models_path = Path(__file__).parent / "models.json";
    with open(models_path, 'r') as f {
        data = json.load(f);
    }
    return data[key]["model_name"];
}

glob llm_opus: Model = Model(model_name=load_model_name("opus"));
glob llm_fable: Model = Model(model_name=load_model_name("fable"));

def gen_text_opus(prompt: str) -> str by llm_opus();
sem gen_text_opus = "Follow the instructions in the prompt exactly and return only the requested text.";
sem gen_text_opus.prompt = "Full generation instructions to follow.";

def gen_text_fable(prompt: str) -> str by llm_fable();
sem gen_text_fable = "Follow the instructions in the prompt exactly and return only the requested text.";
sem gen_text_fable.prompt = "Full generation instructions to follow.";

glob CALLS_MADE: int = 0;
glob RETRY_SLEEPS: list[int] = [2, 8];

def get_calls_made() -> int {
    return CALLS_MADE;
}

def require_budget() -> int {
    val: str | None = os.environ.get("JAC_CALL_BUDGET");
    if not val {
        raise ValueError("JAC_CALL_BUDGET must be set (no default -- an unlimited budget is not allowed)");
    }
    return int(val);
}

def call_with_retry(fn_name: str, prompt: str, model: str) -> dict {
    global CALLS_MADE;

    load_env();

    if model != "opus" and model != "fable" {
        raise ValueError("model must be 'opus' or 'fable', got: " + str(model));
    }

    budget: int = require_budget();
    if CALLS_MADE >= budget {
        raise RuntimeError("JAC_CALL_BUDGET exhausted (" + str(CALLS_MADE) + "/" + str(budget) + ")");
    }

    model_id: str = load_model_name(model);
    tag: str = require_run_tag();
    log_path: str = data_root(tag) + "/logs/generation/calls_" + tag + ".jsonl";

    attempt: int = 1;
    while attempt <= 3 {
        CALLS_MADE += 1;
        ok: bool = True;
        text: str = "";
        err: str = "";
        try {
            if model == "opus" {
                text = gen_text_opus(prompt);
            } else {
                text = gen_text_fable(prompt);
            }
        } except Exception as e {
            ok = False;
            err = str(e);
        }

        rec: dict = {
            "ts": datetime.now().isoformat(),
            "model": model,
            "model_id": model_id,
            "fn": fn_name,
            "ok": ok,
            "attempt": attempt,
            "prompt_chars": len(prompt),
            "resp_chars": len(text) if ok else 0
        };
        append_jsonl(log_path, rec);

        if ok {
            return {"ok": True, "text": text, "err": ""};
        }

        if attempt < 3 {
            time.sleep(float(RETRY_SLEEPS[attempt - 1]));
            attempt += 1;
        } else {
            return {"ok": False, "text": "", "err": err};
        }
    }

    return {"ok": False, "text": "", "err": "unreachable"};
}
