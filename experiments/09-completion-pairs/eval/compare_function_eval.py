"""Paired comparison of two adapters on jac-data-gen function eval v1 (same tasks, same grader).

    .venv/bin/python experiments/09-completion-pairs/eval/compare_function_eval.py \
        experiments/08-nitin-new2-ds/results/function_v1_test/adapter \
        experiments/09-completion-pairs/results/function_v1_test/adapter

Each dir holds results.jsonl (grader) and tests.jsonl (rescore_tests.py). Prints markdown:
grader pass@1 and strict pass@1 (all hidden tests pass on re-run) per task type, with an exact
McNemar test on the discordant tasks (A-only vs B-only).
"""
import json
import math
import sys
from pathlib import Path

A, B = Path(sys.argv[1]), Path(sys.argv[2])


def load(p: Path) -> dict:
    return {r["problem_id"]: r for r in map(json.loads, p.open())}


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def verdicts(d: Path) -> tuple[dict, dict]:
    res, tst = load(d / "results.jsonl"), load(d / "tests.jsonl")
    grader = {p: bool(r.get("task_success")) for p, r in res.items()}
    strict = {p: grader[p] and tst[p]["n_passed"] == tst[p]["n_tests"] for p in res}
    return grader, strict


ga, sa = verdicts(A)
gb, sb = verdicts(B)
ids = sorted(set(ga) & set(gb))
print(f"A = {A}\nB = {B}\n")
print("| verdict | scope | n | A | B | both | A only | B only | delta B-A | p |")
print("|---|---|---|---|---|---|---|---|---|---|")
for name, va, vb in (("grader", ga, gb), ("strict", sa, sb)):
    for scope in ("all", "completion", "translation"):
        s = [p for p in ids if scope == "all" or p.startswith("fn-complete" if scope == "completion" else "fn-translate")]
        a, b = sum(va[p] for p in s), sum(vb[p] for p in s)
        both = sum(va[p] and vb[p] for p in s)
        ao, bo = a - both, b - both
        print(f"| {name} | {scope} | {len(s)} | {100 * a / len(s):.1f}% | {100 * b / len(s):.1f}% | {both} | {ao} | {bo} "
              f"| {100 * (b - a) / len(s):+.1f} pp | {mcnemar_p(ao, bo):.2g} |")
