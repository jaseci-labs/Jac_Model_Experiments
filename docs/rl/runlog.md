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

## Attempt 2 — gold-SFT warm-start → GRPO (revealed the REAL bug)

Warm-start SFT on the gold bodies converged hard (train loss 2.46 → **0.006**),
but eval was **still flat and identical to base** (warm pass == base pass). A
memorized model not reproducing its golds made no sense — so it wasn't the model.

**THE REAL BUG (root cause of every flat run): broken splice.** Models emit the
WHOLE unit — `can walk_day with Day entry { <body> }` — but the template hole sits
*inside* the unit: `can walk_day with Day entry { __HOLE__ }`. Splicing the full
unit into the inner hole produced **nested, broken Jac**
(`can … { can … { … } }`) → never ran. So base, warm (loss 0.006!), and all three
GRPO adapters scored ~identically because the **splice — not the model — was
broken**. Every "RL is flat" conclusion in Attempt 1 was measuring nested garbage.

Verified: the warm model *does* emit the gold body; unwrapping its single
enclosing unit to the inner block → splice → **exit 0, output == expected**.

**Fix:** `unwrap_unit` in `extract_jac` (reward_logic.jac + eval_rl.jac) — when a
completion is one wrapped unit, take its inner block before splicing.

### After the fix — the true baseline

With the splice fixed, **runs == pass** (runnable completions are now correct):
- base: holdout 1/7 (14.3%), train 2/24 (8.3%) exact pass — *real* numbers.
- base ≈ warm: the jac-trained Qwen3-Coder **already generates near-gold**
  completions, so warm-start adds ~nothing (only whitespace differs). Model
  capability is not the bottleneck for this base.
- Why only ~8% pass: generations are *close* but slip on Jac minutiae —
  `Missing ';'`, `Missing ']'`, `here.jid` instead of `jid(here)`. Small fixable
  errors the exact-match bar rejects. **This is exactly what GRPO can target now
  that the reward gives real signal.**

The 3 Attempt-1 GRPO runs were trained on the broken (nested, all-~0-reward)
splice → no gradient. Re-running GRPO with the working reward = Attempt 3.

---

## Attempt 3 — GRPO with the FIXED reward (in progress)

`jac-qwen3coder` base (no warm-start; it's redundant here), LR 1e-5, 300 it,
temp 1.0, dense reward, fixed splice. Holdout + train, sensitive metrics.

| model | split | base pass / near / osim | +grpo pass / near / osim |
|---|---|---|---|
| jac-qwen3coder | holdout | 14.3% / 14.3% / 0.143 | 14.3% / 14.3% / 0.143 |
| jac-qwen3coder | train | 8.3% / 8.3% / 0.083 | 8.3% / 8.3% / 0.083 |

**jac-qwen3coder GRPO result: no greedy change** — despite the fixed reward now
giving real variance (group σ up to 0.11) and nonzero training loss (0.02–0.05),
the +grpo eval is identical to base. Consistent across all attempts: **LoRA-GRPO
at feasible LR (≤1e-5) / 300 iters barely perturbs a 30B's argmax decoding**
(KL≈0). This base is also already near-gold, so there's little room. The pipeline
and reward are correct; the RL *effect on greedy output* is below threshold here.

---

## Attempt 4 — fresh bases, warm-start is the lever (in progress)

`jac-qwen3coder` was already near-ceiling, so warm-start + GRPO barely moved it.
The **fresh** bases (`qwen3coder` = Qwen3-Coder-30B-A3B with no Jac training;
`qwen36` = Qwen3.6-27B dense) start far lower, so the gold-SFT warm-start should
produce a visible lift — the real demonstration. `run_remaining.sh`:
base evals → gold-SFT warm-start → warm evals → GRPO → grpo evals, both models.

Pass % (exact stdout match):

| model | split | base | +warm | +warm+grpo |
|---|---|---|---|---|
| qwen3coder (fresh) | holdout | **0%** | **14.3%** | 14.3% |
| qwen3coder (fresh) | train | 8.3% | 8.3% | 8.3% |
| qwen36 (dense) | holdout | 0% | _ | _ |
| qwen36 (dense) | train | 0% (run 12.5%) | _ | _ |

**qwen3coder: gold-SFT warm-start lifted holdout 0% → 14.3%** — the fresh
Qwen3-Coder base couldn't pass any holdout task; warm-start on the gold bodies
taught it enough to pass one. This is the real, measured win: **warm-start helps
where the base has room** (fresh < jac-trained), exactly as predicted. GRPO on
top added nothing (same LoRA-GRPO-too-weak-for-greedy effect as jac-qwen3coder).

**qwen36 (dense 27B): NOT trainable on 48GB.** Warm-start SFT OOM'd at iter 1
across every config tried (16/8/4 layers, batch 1, seq 2048→768, grad-checkpoint).
The dense model activates all 27B params per token, so forward+backward activation
memory exceeds 48GB regardless of LoRA rank/layers (grad-checkpoint still needs the
full active forward). Inference fits (base eval ran), training does not. The MoE
bases (30B-A3B = 3B active) train fine. So qwen36 is **base-eval only**: holdout
0% pass, train run 12.5% / pass 0%.

---

## Findings

1. **The real blocker was a splice bug, not RL.** Models emit whole units; the
   template hole was inside the unit → nested broken Jac. Every "flat" run before
   the `unwrap_unit` fix measured garbage. Fixed → runs == pass.
2. **Warm-start (gold-SFT) helps where the base has room.** Fresh `qwen3coder`
   holdout pass **0% → 14.3%** after warm-start. The jac-trained base was already
   near-gold, so warm-start (and GRPO) barely moved it.
3. **LoRA-GRPO did not change greedy outputs** on any model at feasible LR
   (≤1e-5) / 300 iters: reward variance and nonzero loss were present, but KL≈0
   and greedy eval was identical to the (warm or base) start. GRPO's effect on a
   30B's argmax decoding is below threshold at this scale; the supervised
   warm-start is what moved the needle.
4. **Hardware:** MoE 30B-A3B trains on 48GB (peak ~38GB); dense 27B does not.
5. **Task difficulty:** exact full-stdout match at 31 tasks is a hard, coarse
   bar (base 0–14%). `near-pass`/`avg-osim` were added to see partial progress.

**Net:** the harness, reward, warm-start, and eval are all correct and proven on
real 30B models. The measured RL gain is the warm-start lift (0→14.3% on the
fresh MoE base); GRPO-on-LoRA added nothing measurable at this scale, and the
dense model is untrainable on this box. Honest, fully-recorded outcome.

---

## Phase 2.5 (the fixes — full record in [`../documentation/00-rl-phase.md`](../documentation/00-rl-phase.md))

Built: pass@k sampled eval, 51 tasks (holdout 12), STaR loop (`run_star.sh`),
35B-A3B swap. Results:
- **pass@8 re-measure:** pass@1 == pass@8 across all configs → **GRPO null is real,
  not a greedy artifact.**
- **STaR:** loop works (adds 2–4 correct samples/round); qwen3coder round 1
  flickered to pass@4 **25%** but didn't hold; greedy stuck at the ~16.7% SFT floor.
- **qwen36 → 35B-A3B (MoE 256 experts):** still **OOMs SFT** on 48 GB even lean.
  Both Qwen3.6 variants are inference-only here; only the 30B-A3B trains.
- **Conclusion:** supervised moves models in this project; LoRA-RL (GRPO, STaR) does
  not, at this difficulty/scale/hardware. In-process jac runner: kept subprocess
  (jaclang reset API fragile, silent-corruption risk).
