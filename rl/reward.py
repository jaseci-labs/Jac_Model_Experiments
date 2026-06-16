"""GRPO reward: the verifiable `jac run` gate, as a reward function.

Loaded by `mlx_lm_lora` via `--reward-functions-file rl/reward.py
--reward-functions jac_behavioral`. For each sampled completion the policy
produces, we splice it into the task's template (replacing the `__HOLE__`
sentinel), run it with `jac`, and score:

    0.3 * compiles + 0.3 * runs + 0.3 * output_match + 0.1 * idiom_bonus

A completion that does not even compile scores 0. `idiom_bonus` only applies to
completions that actually run, so non-running garbage can never earn partial
credit. Paths (templates, the `jac` binary) resolve relative to the repo root,
so the reward works regardless of the trainer's cwd.

Mirrors the behavioral gate in sft_dpo/jacgen/writer.jac:run_jac and the idiom
markers in sft_dpo/jacgen/idiom_eval.jac.
"""

import json
import os
import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "dataset" / "rl" / "templates"
_VENV_JAC = ROOT / ".venv" / "bin" / "jac"
JAC = str(_VENV_JAC) if _VENV_JAC.exists() else "jac"

# Graph-spatial / Jac-native constructs — reward idiomatic output (idiom_eval.jac).
IDIOM_MARKERS = [
    "walker ", "node ", "edge ", "with entry", "with exit", "can ",
    "visit ", "spawn", "disengage", "-->", "<--", "++>", "<++",
    " obj ", "has ", "report ", " root", " here",
]


def _extract_jac(completion: str) -> str:
    """Pull the Jac body out of a model completion (strip ``` fences / prose)."""
    if "```jac" in completion:
        completion = completion.split("```jac", 1)[1]
        return completion.split("```", 1)[0].strip()
    if "```" in completion:
        completion = completion.split("```", 1)[1]
        return completion.split("```", 1)[0].strip()
    return completion.strip()


@lru_cache(maxsize=512)
def _template(task_id: str) -> str:
    return (TEMPLATES / (task_id + ".jac")).read_text()


def _run(code: str, args, timeout: int):
    """Run a jac subcommand over `code` in a temp file -> (exit, stdout, stderr)."""
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "snippet.jac"
        f.write_text(code)
        try:
            # cwd=d so jac's `.jac/` persistence is fresh + discarded per run;
            # otherwise a persistent `root` graph accumulates state across runs.
            p = subprocess.run(
                [JAC] + args + [str(f)],
                capture_output=True, text=True, timeout=timeout, cwd=d,
            )
            return p.returncode, p.stdout, p.stderr
        except (subprocess.SubprocessError, OSError) as e:
            return 124, "", str(e)


def _idiom_bonus(body: str) -> float:
    n = sum(body.count(m) for m in IDIOM_MARKERS)
    return min(n / 4.0, 1.0)


def _score_one(completion: str, answer_json: str) -> float:
    try:
        meta = json.loads(answer_json)
        task_id = meta["id"]
        expected = str(meta["expected_output"])
        timeout = int(meta.get("timeout", 20))
        template = _template(task_id)
    except (ValueError, KeyError, OSError):
        return 0.0

    body = _extract_jac(completion)
    if not body or "__HOLE__" not in template:
        return 0.0
    spliced = template.replace("__HOLE__", body)

    code, out, _err = _run(spliced, ["run"], timeout)
    runs = code == 0
    if runs:
        compiles = True
    else:
        # exit != 0: distinguish a compile error from a runtime error.
        c_code, _o, _e = _run(spliced, ["check", "-p"], min(timeout, 15))
        compiles = c_code == 0

    output_match = runs and out.strip() == expected.strip()
    idiom = _idiom_bonus(body) if runs else 0.0

    return round(
        0.3 * float(compiles)
        + 0.3 * float(runs)
        + 0.3 * float(output_match)
        + 0.1 * idiom,
        4,
    )


def jac_behavioral(prompts=None, completions=None, answer=None, **kwargs):
    """GRPO reward callable. `completions` and `answer` are parallel lists."""
    completions = completions or []
    answers = answer or []
    out = []
    for i, comp in enumerate(completions):
        ans = answers[i] if i < len(answers) else (answers[0] if answers else "{}")
        out.append(_score_one(comp, ans))
    return out


# Register with mlx_lm_lora's reward registry when available (import is optional
# so the module stays unit-testable without the trainer installed).
try:
    from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function

    register_reward_function(name="jac_behavioral")(jac_behavioral)
except Exception:  # pragma: no cover - trainer not installed / API drift
    pass
