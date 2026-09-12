# 08 on jac-data-gen's function eval v1 (Nitin's test cases)

**Question.** Does 08's Spectrum SFT adapter write Jac functions that pass hidden
tests, compared with the untrained base? Up to now every 08 number came from
`eval/eval_functional.jac` ("does the generated Jac compile and run"), which
never checks behavior. This eval does.

## Verdict

**08 turns a model that cannot write Jac (3%) into one that passes Nitin's
hidden tests on 55–57% of tasks.** 57.0% is the grader's own number; 54.9% is
the strict number after removing 21 grader false-passes (below). Either way the
gain is +52–54 pp on the same 1,000 tasks (exact McNemar p ≈ 5e-153).

| per task (pass@1) | all 1,000 tasks | completion (500) | translation (500) |
|---|---|---|---|
| reference solutions | 85.8% / strict 84.6% | 85.8% / 84.6% | 85.8% / 84.6% |
| base (Qwen3-Coder-30B-A3B, 4-bit) | 3.0% / strict 2.9% | 6.0% / 5.8% | 0.0% / 0.0% |
| **base + 08 adapter** | **57.0% / strict 54.9%** | **30.8% / 27.6%** | **83.2% / 82.2%** |

| per hidden test case (29,892 `test` blocks) | all | completion | translation |
|---|---|---|---|
| reference solutions | 86.5% | 86.5% | 86.5% |
| base | 4.5% | 9.1% | 0.0% |
| **base + 08 adapter** | **62.9%** | **42.3%** | **83.6%** |

- **Translation (Python → Jac) is solved to the ceiling:** 83.2% vs the
  references' 85.8%, and 99.3% of 08's translations that compile pass every
  hidden test.
- **Completion (finish a partly written Jac function) is 08's weak spot:**
  30.8%. When the visible prefix stops inside an open block, 08 continues as if
  the block were closed. Experiment 09 targets exactly this.
- 71.5% of 08's answers compile, close to its 70.5% "compiles and runs" on
  holdout (a); 80% of the compiling answers (570/715) also pass all hidden tests.

## Two grader/toolchain problems found (and handled)

1. **Under jaclang 0.16.1 the grader counts tests that crash as passes.**
   `jac test` 0.16.1 exits 0 when a test *errors* (raises an exception) rather
   than *fails* an assert — it prints `FAILED (errors=N)` but returns success —
   and jac-data-gen's `eval_jac.py` trusts the exit code. Re-running every
   sample's tests (`eval/rescore_tests.py`) found 34 such false passes:
   - **08: 21** — 17 are real runtime bugs in 08's code (ValueError 6,
     KeyError 5, TypeError 3, IndexError 2, RecursionError, AttributeError,
     NameError), 4 are a 0.16.1 bug (`NameError: __jac_lambda_1` in tests that
     use lambdas);
   - **reference: 12**, all the `__jac_lambda_1` toolchain bug; **base: 1**.
   The "strict" numbers count all of them as failures, and the reference-valid
   set excludes the 12 tasks whose own tests cannot run under 0.16.1.
   *Worth reporting upstream:* the grader should parse `errors=` or grade
   under 0.36.1 (pytest-based).
2. **142 references fail under 0.16.1 because they are written for jac
   0.36.1.** 138 fail `jac check` — 78 use syntax 0.16.1 cannot parse (a body
   that is a bare trailing expression, e.g.
   `-> list[int] {[(1 & (n >> i)) for i in range(bits)]}`, reported as
   `E0002 Missing ';'`), the rest trip 0.16.1's stricter type checker
   (E1054, E1002, E1053, …); **136 of the 138 pass `jac check` under 0.36.1.**
   4 more fail their own hidden tests. (An earlier draft of this report called
   them all type-checker strictness; the grader's 600-char excerpt was showing a
   later error in the same output.)

## Results

pass@1 = jac-data-gen `task_success` (compiles + hidden tests pass + feature
contract), graded under jaclang 0.16.1. Rate (k/n) [95% Wilson CI].
Reference-valid = the 846 tasks whose reference passes strictly.

| stage | all tasks | completion | translation | reference-valid (846) |
|---|---|---|---|---|
| reference | 85.8% (858/1000) [83.5-87.8] | 85.8% (429/500) | 85.8% (429/500) | 100% (846/846) |
| base | 3.0% (30/1000) [2.1-4.3] | 6.0% (30/500) [4.2-8.4] | 0.0% (0/500) [0.0-0.8] | 3.4% (29/846) [2.4-4.9] |
| **08** | **57.0%** (570/1000) [53.9-60.0] | **30.8%** (154/500) [26.9-35.0] | **83.2%** (416/500) [79.7-86.2] | **59.7%** (505/846) [56.4-62.9] |

**Strict pass@1** (grader pass AND every hidden test passes on re-run)

| stage | grader | strict | completion | translation | grader passes with errored tests |
|---|---|---|---|---|---|
| reference | 85.8% (858) | 84.6% (846/1000) [82.2-86.7] | 84.6% (423/500) | 84.6% (423/500) | 12 |
| base | 3.0% (30) | 2.9% (29/1000) [2.0-4.1] | 5.8% (29/500) | 0.0% (0/500) | 1 |
| **08** | 57.0% (570) | **54.9%** (549/1000) [51.8-58.0] | **27.6%** (138/500) [23.9-31.7] | **82.2%** (411/500) [78.6-85.3] | 21 |

**Base vs 08, paired on the same tasks (exact McNemar, grader verdicts)**

| scope | n | both pass | base only | 08 only | neither | delta | p |
|---|---|---|---|---|---|---|---|
| all tasks | 1000 | 24 | 6 | 546 | 424 | +54.0 pp | 5e-153 |
| completion | 500 | 24 | 6 | 130 | 340 | +24.8 pp | 2e-31 |
| translation | 500 | 0 | 0 | 416 | 84 | +83.2 pp | 1e-125 |
| reference-valid | 846 | 23 | 6 | 482 | 335 | +56.3 pp | 5e-134 |

**Per hidden test case** (partial credit: each `test` block scored on its own;
test cases cluster within tasks, so no CIs)

| stage | all test cases | completion | translation | mean per-task share | among answers that compile |
|---|---|---|---|---|---|
| reference | 86.5% (25870/29892) | 86.5% (12935/14946) | 86.5% (12935/14946) | 85.4% | 98.7% (25870/26218) |
| base | 4.5% (1355/29892) | 9.1% (1355/14946) | 0.0% (0/14946) | 4.3% | 77.4% (1355/1751) |
| **08** | **62.9%** (18817/29892) | **42.3%** (6327/14946) | **83.6%** (12490/14946) | 61.7% | 88.3% (18817/21303) |

**Hidden tests among answers that compile** (isolates behavior from syntax)

| stage | compiles | all tests pass, of those | completion | translation |
|---|---|---|---|---|
| reference | 862/1000 | 99.5% (858/862) | 99.5% (429/431) | 99.5% (429/431) |
| base | 57/1000 | 52.6% (30/57) [39.9-65.0] | 52.6% (30/57) | n/a (none compile) |
| **08** | 715/1000 | **79.7%** (570/715) [76.6-82.5] | 52.0% (154/296) [46.3-57.7] | **99.3%** (416/419) [97.9-99.8] |

**Where samples fail**

| stage | compiles (`jac check`) | passes hidden tests | status counts |
|---|---|---|---|
| reference | 86.2% (862/1000) | 85.8% (858/1000) | pass 858, check_fail 138, test_fail 4 |
| base | 5.7% (57/1000) | 3.0% (30/1000) | check_fail 814, extract_fail 129, pass 30, test_fail 26, timeout 1 |
| 08 | 71.5% (715/1000) | 57.0% (570/1000) | pass 570, check_fail 283, test_fail 144, extract_fail 2, timeout 1 |

First compile error per failing answer (full `jac check` re-run; the grader
keeps only a 600-char excerpt):

- **base** (814): `E0005 Unexpected token` 458, `E0002 Missing ';'` 158,
  `E0001 Expected '{'` 67+23, `E0002 Missing '}'` 45
- **08** (283): `E0002 Missing '}'` **103**, `E0002 Missing ';'` 59,
  `E1004 may implicitly return None` 11, `E1054 no matching overload` 10,
  `E1055 no overload for __add__` 6, `E0013 keyword as parameter name` 5

**Output format against the prompt's rules**

| stage | completion: fenced (rule: no fences) | of which ```` ```python ```` | completion repeats the signature | translation: Jac `def f(...) {` | translation: `func` keyword (not Jac) |
|---|---|---|---|---|---|
| base | 41.0% (205/500) | 112 | 31.2% (156/500) | 0.0% (0/500) | 95.0% (475/500) |
| 08 | 14.2% (71/500) | 0 | 5.6% (28/500) | 95.4% (477/500) | 0.2% (1/500) |

### The base model does not know Jac

- **Translation, 0/500.** Asked for "idiomatic, statically typed Jac", base
  writes a Go/Swift-like pseudo-language (`func _cmp(kd1: any, kd2: any): int {`,
  `var x = ...`) in 95% of answers and never once the Jac form
  `def name(x: T) -> R {`; its failures are mostly `E0005 Unexpected token`.
- **Completion, 30/500.** The prefix hands it the Jac signature, so a bare body
  sometimes works. It loses most tasks on format: 41% of answers are fenced
  although the prompt forbids it (112 as ```` ```python ````, which the grader
  cannot extract) and 31% re-emit the `def` line, duplicating the prefix.

### Why 08's completion lags its translation

08 writes real Jac (95% proper `def ... {` in translation, one stray `func`),
so the gap is not Jac knowledge. It is the task shape:

- **202 completion answers fail `jac check`; 122 leave a brace open** (103 are
  reported first as `E0002 Missing '}'`). The prefix is cut at ~30% of the body,
  often inside a block, and 08 carries on as if that block were closed:

  ```
  prefix:  def _cmp(kd1: str, kd2: str) -> int {
               if kd1 == kd2 {
                   return 0;
  08:          if kd1 < kd2 { return -1; }   return 1; }     <- never closes the first `if`
  ref:     }   if kd1 < kd2 { return -1; }   return 1; }
  ```
  Another 28 re-declare the function (duplicating the prefix); 71 are fenced
  (as ```` ```jac ````, which the grader unwraps; 18 of those pass).
- **141 completion answers compile but fail the hidden tests, all
  brace-balanced.** Typically 08 closes the open loop at once and returns,
  dropping the logic the prefix was building (prefix ends `for k in paths {`,
  08 writes `} return locate; }`). Per test case, 08 still gets 42.3% of
  completion tests right, so many of these answers are partly correct.
- 143 of the 500 completion prefixes stop right after the `def ... {` line,
  a shape 08 never saw either.

08 was trained on whole-function answers only; "continue this partial function
from an arbitrary cut" never appeared in its data. Experiment 09
(`experiments/09-completion-pairs/`) continues 08 on such pairs cut from 08's own
SFT functions.

### Toolchain cross-check (jac 0.36.1, the eval's own toolchain)

Every 20th source function of `private/test.jsonl`, both task variants
(25 sources x 2 = 50 tasks):

| stage | n | success, 0.16.1 | success, 0.36.1 | same verdict |
|---|---|---|---|---|
| reference | 50 | 76.0% (38/50) | 96.0% (48/50) | 36/50 |
| base | 50 | 6.0% (3/50) | 6.0% (3/50) | 50/50 |
| 08 | 50 | 52.0% (26/50) | 54.0% (27/50) | 49/50 |

The models' verdicts barely move with the toolchain (50/50 and 49/50 agree);
the references move a lot, consistent with them being written for 0.36.1. So the
absolute ceiling is toolchain-dependent, the base-vs-08 comparison is not.

### Caveats

- One greedy sample per task (pass@1); batched generation is not bit-identical
  to single-row generation, so only aggregate rates are meaningful.
- Graded under 0.16.1, not the eval's 0.36.1 (see Method); the strict and
  reference-valid columns and the cross-check bound the effect.
- The eval manifest is still `candidate_unvalidated`.
- Base generation was interrupted at 128/1000 by a memory stall (another
  session loaded a second model on the same GPU) and resumed by id with
  identical settings, so its first 128 samples come from an earlier process.

## The eval

`evals/function/v1/` in [chess10kp/jac-data-gen](https://github.com/chess10kp/jac-data-gen)
(commit `5222202a`), built by `scripts/eval/build_function_eval.py` from
`nuprl/stack-dedup-python-testgen-starcoder-filter-v2` Python functions that
come with generated tests (min 5 tests, 100% line coverage), clustered by
Python AST shape so near-duplicates share a split. Test split: 500 source
functions x 2 task variants = **1,000 tasks**, 29,892 hidden test blocks:

- **completion (500):** the prompt shows the Jac signature plus the first
  ~30% of the body; the model writes the rest. The prompt says "Return only the
  missing Jac continuation; do not repeat the visible prefix and do not add
  Markdown fences." The grader prepends the prefix to the output as-is (one
  fenced ```` ```jac ```` block is unwrapped; anything else fenced is an
  extraction failure).
- **translation (500):** the prompt shows the Python function; the model writes
  the whole typed Jac function.

Each task carries hidden `test "tN" { assert ... }` blocks (`private/`, never
shown to the model) plus a feature contract (no Python syntax, no embedded
test blocks, no `node`/`edge`/`walker`). A sample **succeeds** when
`jac check` passes, the hidden tests pass under `jac test`, and the contract
holds.

**No leakage into 08.** 08's dataset build applied the eval's own
`denylist_ids.txt`; independently re-checked here: 0 of the eval's 700
source ids appear anywhere in 08's `sft_train.jsonl` or `candidate_pool.jsonl`.

## Method

- **Generation** (`eval/gen_function_eval.jac`): the public prompt as a single
  user turn, Qwen chat template, `mlx_lm.batch_generate` greedy, batch 32,
  `max_tokens` 1024 (longest reference is ~500 tokens). One sample per task.
  Raw text is kept unmodified, so following the prompt's format rules is part
  of the score.
- **Models:** base `models/qwen-q4` (Qwen3-Coder-30B-A3B, MLX 4-bit) and base +
  08 adapter (`experiments/08-nitin-new2-ds/adapter/`, Spectrum SFT-final,
  70.5% on holdout (a)).
- **Grading:** jac-data-gen's own `scripts/eval/eval_jac.py`, unmodified,
  `--k 1`, 120 s timeout per stage.
- **Toolchain.** jac-data-gen records `jac 0.36.1` (the new standalone
  binary). Under 0.36.1 on this machine `jac test` costs ~4 s per test block,
  hours per model for 29,892 blocks, so the primary grade uses **jaclang
  0.16.1**, the toolchain 08 was trained and data-gated against, and the
  0.36.1 cross-check covers 50 tasks.
- **Re-scoring** (`eval/rescore_tests.py`): re-assembles each sample exactly
  like the grader, re-runs `jac test` to count passed test blocks (per-test
  table, strict pass@1) and re-runs `jac check` on compile failures to record the
  full first error.
- **Stats:** 95% Wilson intervals; base vs 08 paired on the same tasks with an
  exact McNemar test.

Rerun: `experiments/08-nitin-new2-ds/eval/run_function_eval.sh` (stages
`reference base adapter`; needs the sibling `../jac-data-gen` clone and
`.jac-grader/jac-0.36.1`, both described in the script header; takes
`/tmp/jac-gpu.lock` around model loads; `CROSS_ONLY=1` re-runs just the 0.36.1
cross-check), then `.venv/bin/python experiments/08-nitin-new2-ds/eval/rescore_tests.py experiments/08-nitin-new2-ds/results/function_v1_test`
and `.venv/bin/python experiments/08-nitin-new2-ds/eval/summarize_function_eval.py experiments/08-nitin-new2-ds/results/function_v1_test`.
Wall time here: ~20 min generation per model, ~10 min grading per stage,
~19 min for the per-test rescore of all three stages.

**What is in git.** `results/function_v1_test/{base,adapter}/samples.jsonl`
(model outputs), their `tests.jsonl` (per-test counts and first compile error),
every `summary.json` and `provenance.json`. Not committed: the grader's
per-sample rows (`results.jsonl`, their error excerpts quote the eval's hidden
test asserts) and everything derived from the private reference solutions
(`reference/samples*.jsonl`, `reference/tests.jsonl`). All regenerate locally:
a stage whose samples are complete skips generation and only re-grades.
