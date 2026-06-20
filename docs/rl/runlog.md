# RL Run Log — what was tried, what happened

A running record of the actual GRPO runs, kept so failures aren't silently
overwritten. Numbers are real (local MLX, 48GB). See [`strat.md`](strat.md) for
the design.

---

## Attempt 1 — direct GRPO, no warm-start (FAILED, flat)

**Setup.** `jac-qwen3coder` (Qwen3-Coder-30B-A3B, SFT+DPO fused → q4). Three GRPO
runs, no warm-start. Holdout 7 tasks, train 24.

| run | config | holdout pass | train pass | KL | result |
|---|---|---|---|---|---|
| 1 | LR 1e-6, 200 it, group 6/comp 512 | — | — | — | **OOM at iter 2** (Metal). Fixed: group4/comp256/seq1280 → peak ~38GB. |
| 1b | LR 1e-6, 200 it (safe cfg) | 0% | 0% | 0.0 | flat; greedy outputs byte-identical to base |
| 2 | LR 1e-5, 300 it, temp 1.1 | 0% | 0% | 0.0 | flat; identical |
| 3 | LR 1e-5, 300 it, temp 1.2 + **dense reward** | 0% | 0% | 0.0 | reward σ now >0 and loss>0 (0.093), but eval still identical |

**Root cause (from the reward logs).**
1. **Reward groups collapsed to σ=0.000.** At 0% base pass, nearly every rollout
   in a group scored the same low value → GRPO advantage `(r−mean)/σ = 0` → zero
   gradient *regardless of LR*. This is why 1e-6 and 1e-5 gave the identical null
   result. (`KL Divergence: 0.000000000000` every step.)
2. Adding a **dense body-similarity reward term** (run 3) fixed the variance
   (σ rose to 0.01–0.15, loss became nonzero) — real gradient flowed — but the
   updates were too small to change *greedy* decoding, because the base is far
   from correct (0% pass) and small RL nudges don't cross to runnable-correct
   code. No success signal to amplify.

**Conclusion.** The pipeline runs clean end-to-end; the failure is a
task/capability problem: **RL cannot bootstrap a skill the base has 0% of.** The
base must first reach pass > 0 (warm-start) before GRPO has anything to optimize.
Also, exact-stdout-match eval is blind to partial progress (0→0).

**Diagnostic numbers (base, greedy):** holdout run 14.3% / pass 0% ; train run
16.7% / pass 0%. Reward μ ~0.06–0.31 throughout.

---

## Changes made after Attempt 1

1. **OOM fix** — `run_grpo.sh` defaults → `GROUP_SIZE=4`, `MAX_COMPLETION=256`,
   `MAX_SEQ=1280` (measured peak ~38GB, ~17–23 s/iter). `group6/comp512` OOMs.
2. **Dense reward shaping** — `reward_logic.jac`: added a body-similarity term
   (`0.15 · difflib(completion, gold)`); rebalanced to
   `0.25 compile + 0.25 run + 0.25 output + 0.10 idiom + 0.15 sim`. Breaks the
   σ=0 zero-advantage trap (sidecar gold bodies in `dataset/rl/refbodies/`).
3. **Sensitive eval** — `eval_rl.jac`: added `avg-osim` (continuous output
   similarity over all tasks) and `near-pass` (osim ≥ 0.9), so partial gains are
   visible where exact-pass stays 0.
4. **Supervised warm-start** — `build_sft_gold.jac`: builds SFT pairs from the
   gold reference bodies and feeds `run_rft.sh`'s SFT+fuse. At 0% base pass,
   rejection sampling keeps nothing, so warm-start is supervised on gold → lifts
   pass > 0 → GRPO gets a real success signal. Applied to **all three** models
   (incl. the jac-trained one, which is still 0% pass on *these* tasks).

---

## Attempt 2 — gold-SFT warm-start → GRPO (in progress)

**Recipe (per model):** gold-SFT warm-start (`run_rft.sh`, 200 it) → eval-warm →
GRPO on the warmed base (LR 2e-5, 300 it, temp 1.2, dense reward) → eval, with
the sensitive metrics, on holdout + train.

Results table — filled as runs land:

| model | metric | base | +warm | +warm+grpo |
|---|---|---|---|---|
| jac-qwen3coder | holdout pass / near / osim | 0% / — / — | _ | _ |
| jac-qwen3coder | train pass / near / osim | 0% / — / — | _ | _ |
| qwen3coder | … | | | |
| qwen36 | … | | | |

_(updated when Attempt 2 completes; this file is the record of whether the
changes actually moved the numbers.)_
