# 08 on jac-data-gen's function eval v1 (Nitin's test cases)

**Question.** Does 08's Spectrum SFT adapter write Jac functions that pass hidden
tests, compared with the untrained base? Up to now every 08 number came from
`eval/eval_functional.jac` ("does the generated Jac compile and run"), which
never checks behavior. This eval does.

## Verdict

**08 turns a model that cannot write Jac (3.0%) into one that passes Nitin's
hidden tests on 57.0% of tasks** (+54.0 pp on the same 1,000 tasks, exact
McNemar p ≈ 5e-153: 546 tasks flip base-fail → 08-pass, 6 flip the other way).

- **Translation (Python → Jac): 83.2%**, within 2.6 pp of the reference
  solutions' own 85.8% under this grader. Base: 0.0%.
- **Completion (finish a partly written Jac function): 30.8%.** Base: 6.0%.
  This is 08's weak spot, and it is mostly one failure: when the visible prefix
  stops inside an open block, 08 continues as if the block were closed (below).
- 71.5% of 08's answers compile, close to its 70.5% "compiles and runs" on
  holdout (a); here we can also see that 80% of the compiling answers
  (570/715) are behaviorally correct.

## Results

pass@1 = jac-data-gen `task_success` (compiles + hidden tests pass + feature
contract), graded under jaclang 0.16.1. Rate (k/n) [95% Wilson CI].

| stage | all tasks | completion | translation | reference-valid tasks (858) |
|---|---|---|---|---|
| reference solutions | 85.8% (858/1000) [83.5-87.8] | 85.8% (429/500) | 85.8% (429/500) | 100% (858/858) |
| base (Qwen3-Coder-30B-A3B, 4-bit) | 3.0% (30/1000) [2.1-4.3] | 6.0% (30/500) [4.2-8.4] | 0.0% (0/500) [0.0-0.8] | 3.4% (29/858) [2.4-4.8] |
| **base + 08 adapter** | **57.0%** (570/1000) [53.9-60.0] | **30.8%** (154/500) [26.9-35.0] | **83.2%** (416/500) [79.7-86.2] | **59.3%** (509/858) [56.0-62.6] |

**Base vs 08, paired on the same tasks (exact McNemar)**

| scope | n | both pass | base only | 08 only | neither | delta | p |
|---|---|---|---|---|---|---|---|
| all tasks | 1000 | 24 | 6 | 546 | 424 | +54.0 pp | 5e-153 |
| completion | 500 | 24 | 6 | 130 | 340 | +24.8 pp | 2e-31 |
| translation | 500 | 0 | 0 | 416 | 84 | +83.2 pp | 1e-125 |
| reference-valid | 858 | 23 | 6 | 486 | 343 | +55.9 pp | 3e-135 |

**Where samples fail**

| stage | compiles (`jac check`) | passes hidden tests | status counts |
|---|---|---|---|
| reference | 86.2% (862/1000) | 85.8% (858/1000) | pass 858, check_fail 138, test_fail 4 |
| base | 5.7% (57/1000) | 3.0% (30/1000) | check_fail 814, extract_fail 129, pass 30, test_fail 26, timeout 1 |
| 08 | 71.5% (715/1000) | 57.0% (570/1000) | pass 570, check_fail 283, test_fail 144, extract_fail 2, timeout 1 |

By task, 08: completion = pass 154, check_fail 202, test_fail 141,
extract_fail 2, timeout 1; translation = pass 416, check_fail 81, test_fail 3.

**Output format against the prompt's rules**

| stage | completion: fenced (rule: no fences) | of which ```` ```python ```` | completion repeats the signature | translation: Jac `def f(...) {` | translation: `func` keyword (not Jac) |
|---|---|---|---|---|---|
| base | 41.0% (205/500) | 112 | 31.2% (156/500) | 0.0% (0/500) | 95.0% (475/500) |
| 08 | 14.2% (71/500) | 0 | 5.6% (28/500) | 95.4% (477/500) | 0.2% (1/500) |

### The base model does not know Jac

- **Translation, 0/500.** Asked for "idiomatic, statically typed Jac", base
  writes a Go/Swift-like pseudo-language (`func _cmp(kd1: any, kd2: any): int {`,
  `var x = ...`) in 95% of answers and never once the Jac form
  `def name(x: T) -> R {`. 393 of its 495 check failures are
  `E0005 Unexpected token`, 80 more `E0030 Unexpected semicolon at module level`.
- **Completion, 30/500.** The prefix hands it the Jac signature, so a bare body
  sometimes works. It loses most tasks on format: 41% of answers are fenced
  although the prompt forbids it (112 as ```` ```python ````, which the grader
  cannot extract) and 31% re-emit the `def` line, duplicating the prefix.

### Why 08's completion lags its translation

08 writes real Jac (95% proper `def ... {` in translation, one stray `func`),
so the gap is not Jac knowledge. It is the task shape:

- **202 completion answers fail `jac check`; 122 of them leave a brace open.**
  The prefix is cut at ~30% of the body, often inside a block, and 08 carries on
  as if that block were already closed:

  ```
  prefix:  def _cmp(kd1: str, kd2: str) -> int {
               if kd1 == kd2 {
                   return 0;
  08:          if kd1 < kd2 { return -1; }   return 1; }     <- never closes the first `if`
  ref:     }   if kd1 < kd2 { return -1; }   return 1; }
  ```
  Another 28 re-declare the function (duplicating the prefix); 71 are fenced
  (as ```` ```jac ````, which the grader unwraps; 18 of those pass).
- **141 completion answers compile but fail the hidden tests, all brace-balanced.**
  Typically 08 closes the open loop at once and returns, dropping the logic the
  prefix was building (prefix ends `for k in paths {`, 08 writes
  `} return locate; }`).

08 was trained on whole-function answers only; "continue this partial function
from an arbitrary cut" never appeared in its data. If completion matters,
prefix-continuation pairs made from the existing SFT rows are the obvious lever.

### Reference ceiling

142 of the 1,000 references fail under 0.16.1: 138 at `jac check` (63x
`E1004 declared return type ... but may implicitly return None`, 10x
`E1002 Cannot return Any, expected bool`, 47 with the error cut off by the
grader's 600-char excerpt), 4 fail their own hidden tests. These are 0.16.1
type-checker strictness, not wrong answers, so scores are also given on the 858
reference-valid tasks; the verdict does not change (08 59.3% vs base 3.4%).

### Toolchain cross-check (jac 0.36.1, the eval's own toolchain)

| stage | n | success, 0.16.1 | success, 0.36.1 | same verdict |
|---|---|---|---|---|
| reference | 50 | 82.0% (41/50) | 94.0% (47/50) | 38/50 |
| base | 50 | 10.0% (5/50) | 10.0% (5/50) | 50/50 |
| 08 | 50 | 30.0% (15/50) | 30.0% (15/50) | 48/50 |

The subset is every 20th row of `private/test.jsonl`, which alternates
completion/translation per source function, so **it contains only completion
tasks; translation was not cross-checked.** On those, the models' verdicts
barely move with the toolchain (50/50 and 48/50 agree, identical rates). The
references move more: 0.36.1 accepts 9 that 0.16.1 rejects and rejects 3 that
0.16.1 accepts, so absolute ceilings are toolchain-dependent while the
base-vs-08 comparison is not.

### Caveats

- One greedy sample per task (pass@1); batched generation is not bit-identical
  to single-row generation, so only aggregate rates are meaningful.
- Graded under 0.16.1, not the eval's 0.36.1 (see Method); the cross-check
  covers completion only.
- The eval manifest is still `candidate_unvalidated`; the reference-valid
  column is the guard against broken tasks.
- Base generation was interrupted at 128/1000 by a memory stall (another
  session loaded a second model on the same GPU) and resumed by id with
  identical settings, so its first 128 samples come from an earlier process.

## The eval

`evals/function/v1/` in [chess10kp/jac-data-gen](https://github.com/chess10kp/jac-data-gen)
(commit `5222202a`), built by `scripts/eval/build_function_eval.py` from
`nuprl/stack-dedup-python-testgen-starcoder-filter-v2` Python functions that
come with generated tests (min 5 tests, 100% line coverage), clustered by
Python AST shape so near-duplicates share a split. Test split: 500 source
functions x 2 task variants = **1,000 tasks**:

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
holds. Score = pass@1 of that.

**No leakage into 08.** 08's dataset build applied the eval's own
`denylist_ids.txt`; independently re-checked here: 0 of the eval's 700
source ids appear anywhere in 08's `sft_train.jsonl` or `candidate_pool.jsonl`.

## Method

- **Generation** (`eval/gen_function_eval.jac`): the public prompt as a single
  user turn, Qwen chat template, `mlx_lm.batch_generate` greedy, batch 32,
  `max_tokens` 1024 (longest reference is ~500 tokens). Same mechanism as
  `eval_functional.jac`. One sample per task. Raw text is kept unmodified, so
  following the prompt's format rules is part of the score.
- **Models:** base `models/qwen-q4` (Qwen3-Coder-30B-A3B, MLX 4-bit) and base +
  08 adapter (`experiments/08-nitin-new2-ds/adapter/`, Spectrum SFT-final,
  70.5% on holdout (a)).
- **Grading:** jac-data-gen's own `scripts/eval/eval_jac.py`, unmodified,
  `--k 1`, 120 s timeout per stage.
- **Toolchain.** jac-data-gen records `jac 0.36.1` (the new standalone
  binary). Under 0.36.1 on this machine `jac test` costs ~4 s per test block
  (one reference task: 69 test items, 70 s with 18 xdist workers, >150 s with
  one), which puts a full grade at hours per model and made every sample time
  out at 8 grader threads. So the primary grade uses **jaclang 0.16.1**, the
  toolchain 08 was trained and data-gated against, and every stage is re-graded
  under 0.36.1 on a fixed every-20th-task subset (50 tasks; completion-only, see
  the cross-check above).
- **Reference ceiling.** The manifest status is `candidate_unvalidated` (the
  references were never confirmed against their own tests). The reference
  solutions are graded first; model scores are reported both on all tasks and
  on the **reference-valid** tasks (whose reference passes under the same
  grader).
- **Stats:** 95% Wilson intervals; base vs adapter paired on the same tasks
  with an exact McNemar test.

Rerun: `experiments/08-nitin-new2-ds/eval/run_function_eval.sh` (stages
`reference base adapter`; needs the sibling `../jac-data-gen` clone and
`.jac-grader/jac-0.36.1`, both described in the script header; takes
`/tmp/jac-gpu.lock` around model loads), then
`.venv/bin/python experiments/08-nitin-new2-ds/eval/summarize_function_eval.py experiments/08-nitin-new2-ds/results/function_v1_test`.
Wall time here: ~20 min generation per model, ~10 min grading per stage, plus
~20 min for the reference stage's 0.36.1 subset.

**What is in git.** `results/function_v1_test/{base,adapter}/samples.jsonl`
(the model outputs), every `summary.json` and `provenance.json`. Not
committed: the per-sample grader rows (`results.jsonl`), because their error
excerpts quote the eval's hidden test asserts (111 rows for 08), and
`reference/samples*.jsonl`, which are the private reference solutions. Both
regenerate locally: a stage whose samples are complete skips generation and
only re-grades.
