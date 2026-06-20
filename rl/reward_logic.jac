"""GRPO reward logic, in Jac: the verifiable `jac run` gate as a reward function.

All reward logic lives here. `rl/reward.py` is a 2-line shim that imports this
module so `mlx_lm_lora`'s Python loader (`spec_from_file_location`, which needs a
`.py`) triggers the registration below. For each sampled completion: splice it
into the task template (`__HOLE__`), run it with `jac` (isolated cwd so the
persistent `root` graph never accumulates), and score:

    0.3*compiles + 0.3*runs + 0.3*output_match + 0.1*idiom_bonus

Non-running output earns no idiom credit. Mirrors sft_dpo/jacgen/writer.jac:run_jac
and the idiom markers in sft_dpo/jacgen/idiom_eval.jac.
"""

import os;
import json;
import difflib;
import subprocess;
import tempfile;
import from functools { lru_cache }
import from pathlib { Path }
import from mlx_lm_lora.trainer.grpo_reward_functions { register_reward_function }

glob ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));
glob TEMPLATES: str = os.path.join(ROOT, "dataset", "rl", "templates");
glob REFBODIES: str = os.path.join(ROOT, "dataset", "rl", "refbodies");
glob JAC: str = os.path.join(ROOT, ".venv", "bin", "jac") if os.path.exists(os.path.join(ROOT, ".venv", "bin", "jac")) else "jac";

# Weighted idiom markers: graph-traversal/object-spatial ops (the hard, distinctly
# Jac constructs) count for more than plain declarations. Rewards genuinely
# graph-spatial bodies over function-shaped ones.
glob IDIOM_WEIGHTS: dict[str, int] = {
    "visit ": 3, "-->": 3, "<--": 3, "++>": 3, "<++": 3, "spawn": 3, "disengage": 3,
    "report ": 2, " here": 2, " root": 2, "walker ": 2, "node ": 2, "edge ": 2,
    "with entry": 1, "with exit": 1, "can ": 1, " obj ": 1, "has ": 1
};

def extract_jac(completion: str) -> str {
    if "```jac" in completion {
        part = completion.split("```jac", 1)[1];
        return part.split("```", 1)[0].strip();
    }
    if "```" in completion {
        part = completion.split("```", 1)[1];
        return part.split("```", 1)[0].strip();
    }
    return completion.strip();
}

@lru_cache(maxsize=512)
def template_for(task_id: str) -> str {
    return Path(os.path.join(TEMPLATES, task_id + ".jac")).read_text();
}

@lru_cache(maxsize=512)
def refbody_for(task_id: str) -> str {
    p = os.path.join(REFBODIES, task_id + ".txt");
    return Path(p).read_text() if os.path.exists(p) else "";
}

def run_jac(code: str, args: list, timeout: int) -> tuple[int, str, str] {
    with tempfile.TemporaryDirectory() as d {
        f = Path(str(d)) / "snippet.jac";
        f.write_text(code);
        try {
            # cwd=d so jac's `.jac/` persistence is fresh per run.
            p = subprocess.run(
                [JAC] + args + [str(f)], capture_output=True, text=True, timeout=timeout, cwd=str(d)
            );
            return (p.returncode, p.stdout, p.stderr);
        } except (subprocess.SubprocessError, OSError) as e {
            return (124, "", str(e));
        }
    }
}

def idiom_bonus(body: str) -> float {
    n = sum([body.count(k) * w for (k, w) in IDIOM_WEIGHTS.items()]);
    return min(n / 8.0, 1.0);
}

def output_score(out: str, expected: str) -> float {
    # Exact match dominates (1.0); a near-miss earns half its similarity ratio,
    # so behaviourally-close code gets a softer gradient without rivalling exact.
    o = out.strip(); e = expected.strip();
    if o == e {
        return 1.0;
    }
    return 0.5 * difflib.SequenceMatcher(None, o, e).ratio();
}

def score_one(completion: str, answer_json: str) -> float {
    try {
        meta = json.loads(answer_json);
        task_id = str(meta["id"]);
        expected = str(meta["expected_output"]);
        timeout = int(meta.get("timeout", 20));
        template = template_for(task_id);
    } except (ValueError, KeyError, OSError) {
        return 0.0;
    }

    body = extract_jac(completion);
    if not body or "__HOLE__" not in template {
        return 0.0;
    }
    spliced = template.replace("__HOLE__", body);

    (code, out, err) = run_jac(spliced, ["run"], timeout);
    runs = code == 0;
    if runs {
        compiles = True;
    } else {
        (c_code, o2, e2) = run_jac(spliced, ["check", "-p"], min(timeout, 15));
        compiles = c_code == 0;
    }
    osc = output_score(out, expected) if runs else 0.0;
    idiom = idiom_bonus(body) if runs else 0.0;
    # Dense shaping: textual similarity to the reference body. Always defined
    # (even for non-compiling output), so groups get within-group reward variance
    # -> non-zero GRPO advantage -> a learnable gradient at cold start. Verifiable
    # terms still dominate (0.75 of the weight); sim is the tiebreaker.
    ref = refbody_for(task_id);
    sim = difflib.SequenceMatcher(None, body, ref).ratio() if ref else 0.0;

    return round(
        0.25 * float(compiles) + 0.25 * float(runs)
        + 0.25 * osc + 0.10 * idiom + 0.15 * sim,
        4
    );
}

def jac_behavioral(prompts: any = None, completions: any = None, answer: any = None, types: any = None) -> list {
    # Dedup cache: GRPO samples `group_size` rollouts per prompt; identical
    # completions (common early in training) are scored once, not re-run. Keyed
    # by (task_id, extracted body) so it is safe across the whole batch.
    comps = completions or [];
    answers = answer or [];
    cache: dict[str, float] = {};
    out: list = [];
    for (i, comp) in enumerate(comps) {
        ans = answers[i] if i < len(answers) else (answers[0] if answers else "{}");
        key = ans + "\x00" + extract_jac(comp);
        if key not in cache {
            cache[key] = score_one(comp, ans);
        }
        out.append(cache[key]);
    }
    return out;
}

with entry {
    # Runs on import (Python shim or `jac run`): registers into mlx_lm_lora's
    # reward registry so --reward-functions jac_behavioral resolves.
    register_reward_function(name="jac_behavioral")(jac_behavioral);
}
