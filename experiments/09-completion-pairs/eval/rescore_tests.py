"""Per-test-case scores for a run_function_eval.sh output dir (partial credit + full compile errors).

jac-data-gen's grader is all-or-nothing per task and keeps only ~600 chars of each error. For
every sample of a stage this re-assembles the source exactly like the grader (its own
assemble_source / test_blocks), then
  - compile failures: re-runs `jac check` and records the first full `error[Exxxx]` line
  - compiled samples: runs `jac test` on source + hidden tests, parses "Ran N tests" / failures / errors
Writes <stage>/tests.jsonl (counts and first error only, never test text), read by
summarize_function_eval.py.

    .venv/bin/python experiments/08-nitin-new2-ds/eval/rescore_tests.py \
        experiments/08-nitin-new2-ds/results/function_v1_test [reference base adapter]
"""
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
JDG = ROOT.parent / "jac-data-gen"
sys.path.insert(0, str(JDG / "scripts" / "eval"))
import eval_jac  # noqa: E402  jac-data-gen's grader: same source assembly and test blocks

OUT = Path(sys.argv[1])
STAGES = sys.argv[2:] or ["reference", "base", "adapter"]
SPLIT = OUT.name.rsplit("_", 1)[-1]  # function_v1_test -> test
JAC = str(ROOT / ".venv" / "bin" / "jac")
PROBLEMS = {r["id"]: r for r in map(json.loads, (JDG / f"evals/function/v1/private/{SPLIT}.jsonl").open())}


def run(args: list[str], cwd: str) -> tuple[int | None, str]:
    try:
        p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=120)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return None, "timeout"


def score(sample: dict, verdict: dict) -> dict:
    prob = PROBLEMS[sample["problem_id"]]
    tests = eval_jac.test_blocks(prob)
    n = len(re.findall(r"^test\s", tests, re.M))
    row = {"problem_id": sample["problem_id"], "task": sample["task"], "status": verdict["status"],
           "check_pass": bool(verdict.get("check_pass")), "n_tests": n, "n_passed": 0}
    src, _ = eval_jac.assemble_source(prob, sample)
    if src is None:
        return row
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "candidate.jac").write_text(src.rstrip() + "\n")
        if not verdict.get("check_pass"):
            _, out = run([JAC, "check", "candidate.jac"], d)
            m = re.search(r"error\[(E\d+)\]: ([^\n]*)", out)
            row["first_error"] = f"{m.group(1)}: {m.group(2)}" if m else (out.strip().splitlines() or [""])[-1][:120]
            return row
        (Path(d) / "guard.jac").write_text(src.rstrip() + "\n\n" + tests + "\n")
        _, out = run([JAC, "test", "guard.jac"], d)
        ran = re.search(r"Ran (\d+) tests?", out)
        bad = sum(int(x) for x in re.findall(r"(?:failures|errors)=(\d+)", out))
        if ran and int(ran.group(1)) == n:
            row["n_passed"] = n - bad
        else:
            row["test_run_error"] = (out.strip().splitlines() or [""])[-1][:120]
    return row


for st in STAGES:
    smp = [json.loads(l) for l in (OUT / st / "samples.jsonl").open()]
    ver = {r["problem_id"]: r for r in map(json.loads, (OUT / st / "results.jsonl").open())}
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(lambda s: score(s, ver[s["problem_id"]]), smp))
    (OUT / st / "tests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    passed = [r for r in rows if r["status"] == "pass"]
    print(f"{st}: test cases passed {sum(r['n_passed'] for r in rows)}/{sum(r['n_tests'] for r in rows)}; "
          f"task-pass rows not scoring n/n (should be 0): {sum(r['n_passed'] < r['n_tests'] for r in passed)}; "
          f"compiled rows whose test run broke: {sum('test_run_error' in r for r in rows)}")
