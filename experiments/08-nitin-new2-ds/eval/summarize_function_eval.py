"""Summarize a run_function_eval.sh output dir into markdown tables (pasted into report.md).

    .venv/bin/python experiments/08-nitin-new2-ds/eval/summarize_function_eval.py \
        experiments/08-nitin-new2-ds/results/function_v1_test

Score = jac-data-gen's `task_success` (jac check + hidden jac tests + feature contract), pass@1.
"Reference-valid" = tasks whose own reference solution scores task_success under the same grader;
the fair denominator when the grader toolchain rejects some references.
"""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

OUT = Path(sys.argv[1])
STAGES = ["reference", "base", "adapter"]


def load(p: Path) -> dict | None:
    return {r["problem_id"]: r for r in map(json.loads, p.open())} if p.exists() else None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fmt(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}% ({k}/{n}) [{100 * lo:.1f}-{100 * hi:.1f}]"


def mcnemar_p(b: int, c: int) -> float:  # exact two-sided binomial test on the discordant pairs
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def task(pid: str) -> str:
    return "completion" if pid.startswith("fn-complete") else "translation"


def ok(rows: dict, pids) -> tuple[int, int]:
    pids = [p for p in pids if p in rows]
    return sum(bool(rows[p].get("task_success")) for p in pids), len(pids)


res = {s: load(OUT / s / "results.jsonl") for s in STAGES}
res = {s: r for s, r in res.items() if r}
cross = {s: load(OUT / s / "jac-0.36.1-subset" / "results.jsonl") for s in res}
ref = res.get("reference")
valid = {p for p, r in ref.items() if r.get("task_success")} if ref else set()
print(f"_source: {OUT}; provenance: {(OUT / 'provenance.json').read_text().strip() if (OUT / 'provenance.json').exists() else 'n/a'}_\n")

print("### pass@1 (task_success), jac 0.16.1 grader — rate (k/n) [95% Wilson CI]\n")
print("| stage | all tasks | completion | translation | reference-valid tasks |")
print("|---|---|---|---|---|")
for s, rows in res.items():
    ids = list(rows)
    cols = [fmt(*ok(rows, ids))]
    cols += [fmt(*ok(rows, [p for p in ids if task(p) == t])) for t in ("completion", "translation")]
    cols.append(fmt(*ok(rows, [p for p in ids if p in valid])) if valid else "n/a")
    print(f"| {s} | " + " | ".join(cols) + " |")

print("\n### where samples fail (jac 0.16.1 grader)\n")
print("| stage | compiles (`jac check`) | passes hidden tests | status counts |")
print("|---|---|---|---|")
for s, rows in res.items():
    n = len(rows)
    chk = sum(bool(r.get("check_pass")) for r in rows.values())
    tst = sum(bool(r.get("test_pass")) for r in rows.values())
    sc = ", ".join(f"{k} {v}" for k, v in Counter(r["status"] for r in rows.values()).most_common())
    print(f"| {s} | {fmt(chk, n)} | {fmt(tst, n)} | {sc} |")

if "base" in res and "adapter" in res:
    print("\n### base vs 08 adapter, paired on the same tasks (exact McNemar)\n")
    print("| scope | n | both pass | base only | adapter only | neither | delta (pp) | p |")
    print("|---|---|---|---|---|---|---|---|")
    b, a = res["base"], res["adapter"]
    scopes = [("all tasks", set(b) & set(a))]
    scopes += [(t, {p for p in set(b) & set(a) if task(p) == t}) for t in ("completion", "translation")]
    if valid:
        scopes.append(("reference-valid", set(b) & set(a) & valid))
    for name, ids in scopes:
        bb = [bool(b[p].get("task_success")) for p in ids]
        aa = [bool(a[p].get("task_success")) for p in ids]
        both = sum(x and y for x, y in zip(bb, aa))
        bo = sum(x and not y for x, y in zip(bb, aa))
        ao = sum(y and not x for x, y in zip(bb, aa))
        n = len(ids)
        print(f"| {name} | {n} | {both} | {bo} | {ao} | {n - both - bo - ao} | "
              f"{100 * (ao - bo) / n:+.1f} | {mcnemar_p(bo, ao):.2g} |")

models = [s for s in ("base", "adapter") if s in res]
if models:
    print("\n### output format vs the prompt's rules\n")
    print("| stage | completion: fenced (rule: no fences) | of which ```python | completion repeats the visible signature | translation: Jac `def f(...) {` | translation: `func` keyword (not Jac) |")
    print("|---|---|---|---|---|---|")
    for s in models:
        smp = [json.loads(l) for l in (OUT / s / "samples.jsonl").open()]
        comp = [x for x in smp if x["task"] == "completion"]
        tr = [x["output"] for x in smp if x["task"] == "translation"]
        sig = sum(bool(re.search(r"\bdef \w+\(", x["completion"])) for x in comp)
        print(f"| {s} | {fmt(sum('```' in x['completion'] for x in comp), len(comp))} "
              f"| {sum(bool(re.search(r'```(python|py)\b', x['completion'])) for x in comp)} "
              f"| {fmt(sig, len(comp))} "
              f"| {fmt(sum(bool(re.search(r'\bdef \w+\(.*\)\s*(->[^{]*)?\{', t)) for t in tr), len(tr))} "
              f"| {fmt(sum(bool(re.search(r'\bfunc \w+', t)) for t in tr), len(tr))} |")

if any(cross.values()):
    print("\n### toolchain cross-check: same samples graded under jac 0.36.1 (every 20th task)\n")
    print("| stage | n | success, 0.16.1 | success, 0.36.1 | same verdict | 0.36.1 status counts |")
    print("|---|---|---|---|---|---|")
    for s, c in cross.items():
        if not c:
            continue
        ids = list(c)
        agree = sum(bool(c[p].get("task_success")) == bool(res[s][p].get("task_success")) for p in ids)
        sc = ", ".join(f"{k} {v}" for k, v in Counter(r["status"] for r in c.values()).most_common())
        print(f"| {s} | {len(ids)} | {fmt(*ok(res[s], ids))} | {fmt(*ok(c, ids))} | {agree}/{len(ids)} | {sc} |")

if ref:
    bad = [r for r in ref.values() if not r.get("task_success")]
    print(f"\n### reference solutions that fail under jac 0.16.1: {len(bad)}/{len(ref)}\n")
    for st, v in Counter(r["status"] for r in bad).most_common():
        print(f"- {st}: {v}")
    errs = Counter()
    for r in bad:
        text = str(r.get("error", ""))  # the grader keeps only ~600 chars, often all warnings
        m = re.search(r"error\[(E\d+)\]: ([^\n]*)", text)
        errs[f"{m.group(1)}: {re.sub(r'return type .* but', 'return type T but', re.sub(r"'[^']*'", "'x'", m.group(2)))[:70]}" if m
             else "AssertionError (hidden test)" if "AssertionError" in text
             else "error text cut off by the grader"] += 1
    for e, v in errs.most_common(6):
        print(f"  - `{e}` x{v}")
