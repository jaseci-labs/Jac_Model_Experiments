# SFT→DPO on CPT-v2 — Hypothesis Probe Results

**Question:** Does CPT-v2 (rejected on its own Track A/B eval) provide real
downstream value once SFT/DPO trains on top of it, independent of the CPT-v2
rejection verdict?

**Setup:** `models/qwen-q4` + CPT-v2 adapter (rank16/layers16) -> SFT LoRA
(fresh 85/15 split of `sft_train.jsonl`, 8100/1428, resumed from CPT-v2,
8200 iters/~1 epoch) -> fuse -> DPO LoRA (fresh 85/15 split of
`dpo_train.jsonl`, 654/115, 250 iters/~0.38 epoch, rank16/layers16 override,
beta=0.1, lr=1e-6, fresh adapter on the fused SFT model).

**Results (functional pass rate, "runs" = jac-compiles-and-executes, code-graded
rows only, n=855 out of the 1428-row holdout — the remainder (573 rows) are
`prose_lexical` rows with no code output to grade):**

| Stage | Overall runs % | n |
|---|---|---|
| CPT-v2 base (no SFT) | **47%** (404/855) | 855 |
| +SFT (final, 8200 iters) | **72%** (621/855) | 855 |
| +SFT+DPO (final, 250 iters) | **12%** (105/855) | 855 |

Per-category breakdown (`results/images/comparison_by_category.png`):

| Category | base | +SFT | +SFT+DPO |
|---|---|---|---|
| code_gen | 9.6% | 64.5% | 0.3% |
| conversion | 76.4% | 86.6% | 0.9% |
| debug | 69.4% | 86.1% | 66.7% |
| migration | 100.0% | 100.0% | 0.0% |
| trajectory | 55.7% | 59.0% | 42.1% |

The DPO collapse is not uniform — it wipes out `code_gen`, `conversion`, and
`migration` almost entirely, while `debug` and `trajectory` degrade but don't
fully collapse. Whatever DPO overfit onto, it generalized worst to the
categories that make up most of the eval set (`code_gen` alone is 313/855
rows, `conversion` is 322/855 — together ~74% of the graded set), which is
why the overall number craters as hard as it does.

**DPO preference accuracy (training diagnostics, `results/dpo/images/*.png`,
raw log `results/dpo/train.log`):** train accuracy pinned at **1.000** with
train loss collapsing toward ~0 (0.05 → 0.009 by iter 10-20 windows) almost
immediately, while validation loss rose monotonically the entire run
(0.705 → 2.000) and validation accuracy sat at **~0.000** (worse than the 0.5
chance baseline) for nearly every logged validation step, with validation
margin trending negative throughout. This is a textbook DPO-overoptimization
signature: the policy memorized the training pairs' chosen/rejected ordering
perfectly while losing all ability to generalize the preference to held-out
pairs.

**Verdict:** The +SFT step is a clear, large, real win — CPT-v2 base sits at
47% functional pass rate, and SFT training on top of it (using the same
CPT-v2 adapter as a starting point) drives that to 72%, a +25-point absolute
improvement that also shows up consistently across almost every category.
That answers the underlying hypothesis question on its own: CPT-v2, despite
being rejected on its own Track A/B eval, is not dead weight — SFT on top of
it still produces a substantially better model than doing nothing.

DPO on top of that SFT model, however, is an unambiguous, catastrophic
regression as configured here: 72% → 12%, well below even the untrained
CPT-v2 base (47%). This is not a training-metric artifact or a measurement
bug — it is independently corroborated by DPO's own training diagnostics
(perfect train accuracy + near-zero train loss + rising validation loss +
near-zero validation accuracy = classic overoptimization) and by the
functional-eval collapse hitting hardest in exactly the categories with the
most training mass. The 250-iteration, beta=0.1, lr=1e-6, rank16/layers16,
654-row-filtered-dataset DPO run used here should **not** be used — it
destroys the SFT gains rather than sharpening them.

This does not mean DPO is unfixable for this task, only that this particular
configuration overfit hard and fast. Plausible mitigations for a follow-up,
none of them tried here: far fewer iterations (validation loss was already
rising by iter 20 of each logged segment), a higher beta (less aggressive
preference pressure), a smaller learning rate, or early stopping gated on
validation margin/accuracy instead of running to a fixed iteration count.
As run, though, the honest result is: SFT-on-CPT-v2 works, DPO-on-top-of-that
does not.

**Graphs:** `results/sft/images/*.png` (loss/lr/tokens/functional pass rate
over SFT training), `results/dpo/images/*.png` (loss/accuracy/margin/rewards
over DPO training), `results/images/comparison_overall.png` (3-way overall bar,
CPT-v2 base 47% / +SFT 72% / +SFT+DPO 12%), `results/images/comparison_by_category.png`
(same 3-way comparison broken out per category).
