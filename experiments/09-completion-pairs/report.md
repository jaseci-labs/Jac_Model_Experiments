# 09-completion-pairs — report

**Question.** 08 knows Jac (83% on jac-data-gen translation) but scores only
31% on *completion*: when a prompt shows a partly written function cut inside an
open block, 08 continues as if the block were closed. Does continuing 08 on
"finish this partial function" pairs fix that without costing translation or
holdout (a)?

**Answer: yes on completion, no measurable cost on translation.**
Holdout (a) also holds (71.1% vs 08's 70.5%), so all three parts of the
pre-set success bar are met.

| success bar (set before training) | 08 | 09 | result |
|---|---|---|---|
| completion pass@1 clearly above 08 (paired McNemar p<0.05) | 30.8% | **47.6%** | **+16.8 pp, p = 2e-14 — met** |
| translation within noise of 08 | 83.2% | 81.8% | −1.4 pp, p = 0.23 — met |
| holdout (a) within noise of 70.5% | 70.5% (603/855) | 71.1% (608/855) | +0.6 pp, p ≈ 0.8 — met |

## What 09 is

- **Init:** 08's final Spectrum adapter (read-only `INIT_ADAPTER`), same 16
  Spectrum blocks, same LoRA shape (rank 16, scale 2.0, dropout 0.05).
- **Data** (`scaffold/build_completion_pairs.py`, seeded): 3,089 completion
  pairs cut from 08's own SFT answers that are a `def f(...) {` function — 25%
  cut right after the def line (the prompt ends at `{`, like 143/500 eval
  prompts), the rest at a random 20–60% of the body (42.5% of all pairs land
  inside an open inner block). The answer is the raw remainder, no fences; the
  prompt is one of 6 instruction wordings written for 09, never the eval's
  sentence. Plus 1,500 replay rows of 08's own SFT data. Valid: 539 pairs + 500
  replay. Every pair reconstructs its source function exactly.
- **Leakage:** 0 of the eval's 1,404 denylisted/eval source ids in 09's data; 0
  prompts contain the eval sentence; 0 of the 700 eval completion prefixes
  appear verbatim.
- **Training:** 3,000 iters at batch 1 (~0.65 epoch; ~2,000 pairs + ~1,000
  replay rows seen), lr 1e-5 cosine (warmup 300 → 1e-6), max_seq_length 3072,
  prompt-masked. 1 h 22 min on the M5 Pro (0.63 it/s); peak memory 28.2 GB. Config:
  `train/configs/sft_spectrum.yaml`; design: `docs/spec.md`.
- The logged validation loss (0.06–0.76, no trend) is not informative:
  `val_batches: 8` and mlx_lm draws a new random permutation each time, so every
  point is 8 random rows.

## Results on jac-data-gen function eval v1 (test split, 1,000 tasks)

Same eval, prompts, generation settings and grader as
[08's report](../08-nitin-new2-ds/function-eval-report.md): greedy pass@1,
jac-data-gen's own `eval_jac.py` under jaclang 0.16.1, hidden `test` blocks.
Base and reference rows are 08's (shared).

**pass@1, paired against 08 on the same tasks (exact McNemar)**

| verdict | scope | 08 | **09** | both pass | 08 only | 09 only | delta | p |
|---|---|---|---|---|---|---|---|---|
| grader | all 1,000 | 57.0% | **64.7%** | 532 | 38 | 115 | +7.7 pp | 3e-10 |
| grader | completion | 30.8% | **47.6%** | 132 | 22 | 106 | **+16.8 pp** | 2e-14 |
| grader | translation | 83.2% | 81.8% | 400 | 16 | 9 | −1.4 pp | 0.23 |
| strict | all 1,000 | 54.9% | **62.0%** | 513 | 36 | 107 | +7.1 pp | 2e-9 |
| strict | completion | 27.6% | **43.0%** | 117 | 21 | 98 | **+15.4 pp** | 4e-13 |
| strict | translation | 82.2% | 81.0% | 396 | 15 | 9 | −1.2 pp | 0.31 |

Strict = grader pass AND every hidden test passes on re-run (jaclang 0.16.1's
`jac test` exits 0 when a test raises instead of failing an assert; see 08's
report). 09 has 27 such grader false-passes, 08 had 21.

**Per hidden test case** (29,892 `test` blocks, partial credit)

| model | all | completion | translation | among answers that compile |
|---|---|---|---|---|
| reference solutions | 86.5% | 86.5% | 86.5% | 98.7% |
| base | 4.5% | 9.1% | 0.0% | 77.4% |
| 08 | 62.9% | 42.3% | 83.6% | 88.3% |
| **09** | **74.0%** | **66.0%** | 82.0% | 87.0% |

**Compile and test rates**

| model | compiles | all tests pass, of compiling | completion: compiles | completion: tests pass, of compiling |
|---|---|---|---|---|
| 08 | 71.5% (715) | 79.7% | 296/500 | 52.0% (154/296) |
| **09** | **84.4% (844)** | 76.7% | **430/500** | 55.3% (238/430) |

## What changed in completion

| completion failure mode (500 tasks) | 08 | 09 |
|---|---|---|
| fails `jac check` | 202 | **70** |
| … of which leaves a brace open | 122 | **16** |
| fenced although the prompt forbids it | 71 | **0** |
| re-declares the function (repeats the prefix) | 28 | **0** |
| compiles but fails hidden tests | 141 | 192 |
| pass, prompt ends right after the `def` line (143 tasks) | 42 | **74** |
| pass, prompt cut inside the body (357 tasks) | 112 | **164** |

09 learned the task shape: it closes the blocks the prefix left open, never
fences, never repeats the signature. The remaining completion losses are mostly
behavioral (192 compile but fail tests, vs 141) — more answers now reach the
test stage, and the model has to guess the missing ~70% of an unfamiliar
function's logic from a signature, docstring and a few lines. Per test case its
completion answers are right 66% of the time (08: 42%).

**Translation** moved within noise: 16 tasks 08 passed now fail (14 at
`jac check`, e.g. `E0013 'obj' is a keyword and cannot be used as a parameter
name`), 9 tasks 08 failed now pass. 09 now fences 37 of its translation answers
(allowed: the grader unwraps one ```` ```jac ```` block).

**Toolchain cross-check** (jac 0.36.1, 25 sources x both task types): 09 26/50
vs 08 27/50 — no difference at n=50; 09's verdicts agree with 0.16.1 on 46/50.

## Holdout (a) (shared 855-row code-graded holdout, "compiles and runs")

Same harness as every earlier number (`eval/eval_functional.jac`, "compiles
and runs", 855 code-graded rows of 1,428). 08's per-row verdicts were not kept,
so the test is an unpaired two-proportion z-test (p ≈ 0.8).

| category / gate | 08 | **09** |
|---|---|---|
| conversion / behavioral | 96% (312/322) | 97% (315/322) |
| code_gen / behavioral | 53% (46/86) | 48% (42/86) |
| code_gen / compile_only | 30% (70/227) | 32% (73/227) |
| trajectory / behavioral | 70% (49/70) | 72% (51/70) |
| trajectory / compile_only | 84% (96/113) | 85% (97/113) |
| debug / behavioral | 81% (27/33) | 81% (27/33) |
| debug / compile_only | 100% (3/3) | 100% (3/3) |
| migration / behavioral | 0% (0/1) | 0% (0/1) |
| **OVERALL** | **70.5% (603/855)** | **71.1% (608/855)** |

No category moves beyond a handful of rows; the completion training did not
cost 08's general Jac ability. Log: `results/holdout_a_final.txt`,
`results/metrics_functional.jsonl`.

## Reproduce

```bash
# data (already built; deterministic, refuses to overwrite without FORCE=1)
python3 experiments/09-completion-pairs/scaffold/build_completion_pairs.py
# train: dry run, then the real run (INIT_ADAPTER defaults to 08's adapter, read-only)
experiments/09-completion-pairs/train/run_sft_spectrum.sh
CONFIRM_FULL_RUN=1 experiments/09-completion-pairs/train/run_sft_spectrum.sh
# REQUIRED before any eval: make adapter_config.json cover the 16 Spectrum blocks
.venv/bin/jac run experiments/09-completion-pairs/train/spectrum/adapter_config_fix.jac \
  --adapter-dir experiments/09-completion-pairs/adapter \
  --spectrum-layers experiments/09-completion-pairs/train/spectrum/spectrum_layers.json
# function eval + per-test rescore + paired comparison with 08
experiments/09-completion-pairs/eval/run_function_eval.sh adapter
.venv/bin/python experiments/08-nitin-new2-ds/eval/rescore_tests.py experiments/09-completion-pairs/results/function_v1_test adapter
.venv/bin/python experiments/09-completion-pairs/eval/compare_function_eval.py \
  experiments/08-nitin-new2-ds/results/function_v1_test/adapter experiments/09-completion-pairs/results/function_v1_test/adapter
# holdout (a)
JAC_EVAL_MODEL=models/qwen-q4 JAC_EVAL_ADAPTER=experiments/09-completion-pairs/adapter \
  JAC_HOLDOUT=experiments/09-completion-pairs/dataset/holdout_a_shared855.jsonl \
  .venv/bin/jac run experiments/09-completion-pairs/eval/eval_functional.jac
```
Take `/tmp/jac-gpu.lock` (shared with Jac Model Studio agents) around every
model load; only one model fits comfortably on the 48 GB machine.
