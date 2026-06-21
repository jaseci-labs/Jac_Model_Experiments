# RL Results — GRPO + warm-start + STaR on Jac (local MLX, 48GB)

Summary of both RL passes. Full narrative: [`runlog.md`](runlog.md) and
[`../documentation/00-rl-phase.md`](../documentation/00-rl-phase.md). Pass = exact
stdout match after body-completion.

## Phase 1 (31 tasks, holdout 7) — GRPO

| model | base | +warm | +grpo |
|---|---|---|---|
| jac-qwen3coder (MoE, SFT+DPO) | 14.3% | near-ceiling | 14.3% |
| qwen3coder (MoE, fresh) | 0% | **14.3%** | 14.3% |
| qwen36 (dense 27B) | 0% | OOM | OOM |

- Root cause of early flatness: a **splice bug** (full-unit completions nested into
  the inner hole), fixed with `unwrap_unit`.
- **Warm-start (gold-SFT) is the win:** fresh model 0% → 14.3%.
- **LoRA-GRPO changed nothing** (KL≈0).

## Phase 2.5 (51 tasks, holdout 12) — pass@k + STaR + 35B-A3B

**pass@8 re-measure (existing adapters):** pass@1 == pass@8 == 14.3% for all of
{jac base, jac grpo, q3c warm, q3c grpo}. **GRPO's null result is real, not a
greedy blind spot.**

**STaR (iterate warm-start); pass@1 greedy / pass@4 sampled:**

| model | round 0 | round 1 | round 2 |
|---|---|---|---|
| qwen3coder | 16.7% / 16.7% | 16.7% / **25.0%** | 16.7% / 16.7% |
| jac-qwen3coder | 16.7% / 16.7% | 16.7% / 16.7% | 16.7% / 16.7% |
| qwen36 (35B-A3B) | base 8.3% — untrainable | | |

- STaR's loop works (adds 2–4 correct samples/round). qwen3coder round 1 flickered
  to pass@4 **25%** — faint, real, didn't hold. Greedy never moved off the ~16.7%
  SFT floor.
- **qwen36 / all of Qwen3.6 is inference-only on 48 GB:** dense 27B (all-active) and
  MoE 35B-A3B (256 experts resident) both OOM SFT at iter 1. Only the 30B-A3B
  (fewer experts) trains.

## Bottom line (brutally honest)

The supervised levers move models here (SFT 0→94% function conversion in phase 1;
gold warm-start 0→14.3% holdout). **RL does not:** GRPO added nothing (pass@8
confirmed), STaR added only a noisy flicker, and the bigger Qwen3.6 models can't be
trained on this hardware. The harness, reward, pass@k eval, warm-start, STaR, 51
tasks, and the 35B-A3B swap are all built and proven — the science result is that
**LoRA-RL on a 30B-class model doesn't beat supervised finetuning on this task /
scale / hardware.**

## What would change the result
- A task set the base solves >0% broadly (line-level/partial grading) → real RL
  gradient. Current exact-match at 51 hard tasks gives an ~8–17% floor.
- Full-finetune RL (not LoRA) and/or more VRAM than 48 GB.
- More tasks (100s) for generalization beyond a 12-task holdout.
