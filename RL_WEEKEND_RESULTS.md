# RL Weekend Results (GRPO + warm-start + STaR on Jac)

Local MLX, 48 GB. Jun 20–21 2026. Goal: RL-with-verifiable-rewards on Jac
body-completion — Jac compiler/runtime grades each generation (compile + run +
output-match + idiom + body-similarity); optimize the model against it. No learned
reward model. Method: GRPO (LoRA, `mlx-lm-lora`). Full detail in
`docs/rl/runlog.md`, `docs/rl/RL_RESULTS.md`, `docs/documentation/00-rl-phase.md`.

Pass = exact stdout match after splicing model completion into task template.

## Result in one line

Supervised levers (SFT, DPO, gold warm-start) move models here. **RL does not.**
GRPO added nothing (confirmed by pass@8), STaR added a faint flicker that didn't
hold, and the bigger Qwen3.6 models can't be trained on 48 GB. Harness/reward/eval
all correct and proven on real 30B — the science result is that LoRA-RL on a
30B-class model doesn't beat SFT at this task difficulty / scale / hardware.

## Numbers

Phase 1 — 31 tasks, holdout 7 (pass %, exact stdout):

| model | base | +warm | +grpo |
|---|---|---|---|
| jac-qwen3coder (MoE 30B-A3B, SFT+DPO) | 14.3% | near-ceiling | 14.3% |
| qwen3coder (MoE 30B-A3B, fresh) | 0% | **14.3%** | 14.3% |
| qwen36 (dense 27B) | 0% | OOM | OOM |

Phase 2.5 — 51 tasks, holdout 12. pass@8 re-measure of existing adapters:
pass@1 == pass@8 == 14.3% for all of {jac base, jac grpo, q3c warm, q3c grpo}.

STaR (iterate warm-start; pass@1 greedy / pass@4 sampled):

| model | round 0 | round 1 | round 2 |
|---|---|---|---|
| qwen3coder | 16.7% / 16.7% | 16.7% / **25.0%** | 16.7% / 16.7% |
| jac-qwen3coder | 16.7% / 16.7% | 16.7% / 16.7% | 16.7% / 16.7% |
| qwen36 (35B-A3B) | base 8.3% — untrainable | — | — |

## Failures and why

1. **OOM at iter 2 (Metal).** First GRPO config group6/comp512 blew 48 GB.
   Fix: defaults → group4/comp256/seq1280 (~38 GB peak). Not a real blocker.

2. **σ=0 zero-advantage trap (Attempt 1, flat).** At ~0% base pass, every rollout
   in a group scored the same low value → GRPO advantage `(r−mean)/σ = 0` → zero
   gradient regardless of LR. KL = 0.0 every step. RL cannot bootstrap a skill the
   base has 0% of. Fix attempt: dense body-similarity reward term → σ rose, loss
   nonzero — gradient flowed but greedy eval still identical (suspicious → led to #3).

3. **THE real bug — broken splice (faked Attempts 1–2 entirely).** Models emit the
   WHOLE unit `can walk_day with Day entry { <body> }`, but the template hole sat
   *inside* the unit `can ... { __HOLE__ }`. Splicing full unit into inner hole →
   nested broken Jac `can {...can {...}...}` → never ran. So base, warm (SFT loss
   0.006!), and all GRPO adapters scored identically because the splice — not the
   model — was broken. Every "RL is flat" conclusion before this measured nested
   garbage. Fix: `unwrap_unit` in `extract_jac` (reward_logic.jac + eval_rl.jac) —
   unwrap a single enclosing unit to its inner block before splicing. After fix,
   runs == pass.

4. **LoRA-GRPO doesn't move greedy output (Attempts 3–4, the core null result).**
   With splice fixed and real reward variance (σ up to 0.11, loss 0.02–0.05),
   +grpo eval was still byte-identical to base/warm, KL ≈ 0, on every model. At
   feasible LR (≤1e-5) / 300 iters, LoRA-GRPO barely perturbs a 30B's argmax
   decoding. pass@8 re-measure confirmed pass@1 == pass@8 → the null is real, not a
   greedy blind spot. GRPO is a dead end here.

5. **Dense Qwen3.6-27B untrainable on 48 GB.** Warm-start SFT OOM'd at iter 1
   across every config (16/8/4 layers, batch 1, seq 2048→768, grad-checkpoint). A
   dense model activates all 27B params/token → activation memory exceeds 48 GB
   regardless of LoRA rank. Inference fits, training doesn't. Base-eval only.

6. **35B-A3B swap didn't fix training either.** Swapped dense 27B → Qwen3.6-35B-A3B
   (MoE) expecting it to train like the 30B-A3B. It still OOMs SFT at iter 1: the
   256-expert MoE keeps all experts resident (~18 GB q4) and backward through them
   exceeds 48 GB. Same wall as the dense, different reason (expert count vs
   all-active). Whole Qwen3.6 line is inference-only on this box; only the
   fewer-expert 30B-A3B (Qwen3-Coder) trains.

7. **STaR flicker didn't hold.** Loop works (adds 2–4 exact-correct samples/round).
   qwen3coder round 1 flickered to pass@4 = 25% (3/12) — faint, real — but didn't
   persist and greedy (pass@1) never left the ~16.7% SFT floor.

## What actually worked

- **Warm-start (gold-SFT) where the base has room:** fresh qwen3coder holdout
  0% → 14.3%. The one measured RL-adjacent win. The jac-trained base was already
  near-gold so warm-start barely moved it.
- Harness, reward, pass@k eval, warm-start, STaR loop, 51-task set, 35B-A3B swap —
  all built and validated on real 30B (no mocks).

## Root reasons RL underperformed

- **Coarse metric:** exact full-stdout match at hard tasks gives an ~8–17% floor
  and almost no positive gradient. Misses are tiny Jac slips (`Missing ';'`,
  `here.jid` vs `jid(here)`) that all-or-nothing scoring rejects.
- **LoRA + feasible LR too weak** to shift a 30B's greedy decoding via RL.
- **Hardware:** 48 GB trains only the 30B-A3B (MoE, 3B active, ~38 GB peak);
  every dense/large-MoE Qwen3.6 OOMs.

## What would change it

- Partial/line-level grading so the base scores >0% broadly → real RL gradient.
- Full-finetune RL (not LoRA) and/or >48 GB VRAM.
- 100s of tasks for generalization beyond a 12-task holdout.
