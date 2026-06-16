"""Tests for the GRPO jac-behavioral reward.

Run from the repo root or from rl/:  ../.venv/bin/pytest rl/tests -v
The reward resolves templates + jac relative to the repo root, so cwd does not
matter. These tests require `jac` (the repo .venv) and a built task set
(`jac run rl/build_tasks.jac`).
"""

import json
import re
from pathlib import Path

import pytest

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rl"))

import reward as R  # noqa: E402

TASKS = ROOT / "dataset" / "rl" / "tasks.jsonl"
DRIVERS = ROOT / "rl" / "drivers"


def _load_tasks():
    if not TASKS.exists():
        pytest.skip("no dataset/rl/tasks.jsonl — run `jac run rl/build_tasks.jac` first")
    return [json.loads(l) for l in TASKS.read_text().splitlines() if l.strip()]


def _reference_body(task_id: str) -> str:
    """The original body the driver author wrote, between the HOLE sentinels."""
    for p in DRIVERS.glob("*.jac"):
        text = p.read_text()
        m = re.search(r'>>>HOLE\s+id="%s"' % re.escape(task_id), text)
        if not m:
            continue
        lines = text.split("\n")
        start = next(i for i, l in enumerate(lines) if ">>>HOLE" in l and ('id="%s"' % task_id) in l)
        end = next(i for i, l in enumerate(lines) if "<<<HOLE" in l and i > start)
        return "\n".join(lines[start + 1:end])
    raise AssertionError("no driver for task id %s" % task_id)


def test_perfect_completion_scores_high():
    """Each task's own reference body, fed back as the completion, scores ~1.0."""
    tasks = _load_tasks()
    for t in tasks:
        ans = json.loads(t["answer"])
        body = _reference_body(ans["id"])
        completion = "```jac\n" + body + "\n```"
        score = R.jac_behavioral(prompts=[t["prompt"]], completions=[completion], answer=[t["answer"]])
        assert score[0] >= 0.9, "task %s scored %.3f" % (ans["id"], score[0])


def test_noncompiling_scores_zero():
    tasks = _load_tasks()
    t = tasks[0]
    score = R.jac_behavioral(
        prompts=[t["prompt"]],
        completions=["```jac\nthis is not valid jac @@@ !!!\n```"],
        answer=[t["answer"]],
    )
    assert score[0] == 0.0


def test_wrong_output_partial():
    """Compiles + runs but prints the wrong thing → partial credit, below a pass."""
    tasks = _load_tasks()
    # lib_lang_label is a plain function; a body that returns a constant compiles
    # and runs but mismatches expected output.
    t = next(x for x in tasks if json.loads(x["answer"])["id"] == "lib_lang_label")
    score = R.jac_behavioral(
        prompts=[t["prompt"]],
        completions=['```jac\nreturn "WRONG";\n```'],
        answer=[t["answer"]],
    )
    assert 0.0 < score[0] < 0.9


def test_batch_length_matches():
    tasks = _load_tasks()
    comps = ["```jac\nreturn 1;\n```"] * len(tasks)
    answers = [t["answer"] for t in tasks]
    score = R.jac_behavioral(prompts=[t["prompt"] for t in tasks], completions=comps, answer=answers)
    assert len(score) == len(tasks)
