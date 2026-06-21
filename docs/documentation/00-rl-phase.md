# Phase 2 — Reinforcement Learning (GRPO on Jac): the full, honest record

*Everything that happened in the RL phase, brutally honest — what we built, every
attempt including the failures, the bug that faked most of the failures, the real
numbers, what worked and what didn't, and the forward plan. Single source of truth;
the per-run detail is in [`../rl/runlog.md`](../rl/runlog.md), the strategy in
[`../rl/strat.md`](../rl/strat.md).*

---

## The idea

After SFT+DPO ([`01-sft-dpo-phase.md`](01-sft-dpo-phase.md)) the natural next step
is RL with a **verifiable reward**: the Jac compiler + runtime grade every
generation for free, so optimize the model directly against "does it compile, run,
and produce the right output" — no learned reward model, no human grading. This is
RL-with-verifiable-rewards (RLVR), the regime that works for math/code.

**Method:** GRPO (group-relative policy optimization, via `mlx-lm-lora`), LoRA,
local MLX, 48 GB. **Reward** (`rl/reward_logic.jac`): splice the model's completion
into a task template, `jac run` it, score
`0.25 compile + 0.25 run + 0.25 output-match + 0.10 idiom + 0.15 body-similarity`.
**Tasks:** body-completion over the real open-source `this_is_jac` corpus — hand-
authored "drivers" that mask one unit's body and run deterministically (31 built).
**Models:** `jac-qwen3coder` (our SFT+DPO model), `qwen3coder` (fresh
Qwen3-Coder-30B-A3B), `qwen36` (Qwen3.6, dense 27B → later swapped).

## The scaffolding (all built, all proven on real 30B)

Repo split into `sft_dpo/` (phase 1) and `rl/` (this phase). The `rl/` harness is
all-Jac except a 5-line Python reward shim the trainer's loader requires:
- `build_tasks.jac` (drivers → GRPO data + templates + gold refbodies),
  `build_rl_splits.jac` (train/valid/holdout), `build_sft_gold.jac` (warm-start set)
- `reward_logic.jac` (the reward), `eval_rl.jac` (holdout scorer),
  `rft_sample.jac` (rejection sampling), `run_grpo.sh` / `run_rft.sh` (runners)
- 31 determinism-verified drivers, jac-native test harness

Every seam was validated on a real 30B (model load, generation, reward scoring,
GRPO step, adapter fuse). No part is mocked.

## What actually happened (every attempt)

### Attempt 1 — direct GRPO, no warm-start → FLAT
Three runs on `jac-qwen3coder` (LR 1e-6, 1e-5, dense reward). All flat: holdout
pass 0%, train pass 0%, greedy outputs byte-identical to base. First run also
**OOM'd at iter 2** (group6/comp512) → fixed defaults to group4/comp256/seq1280
(~38 GB peak).
- Reward logs showed group reward **σ=0.000** constantly: at ~0% base success
  nearly every rollout scored the same low value → GRPO advantage `(r-mean)/σ = 0`
  → zero gradient *regardless of LR*. `KL Divergence: 0.0` every step.
- Added a **dense body-similarity reward term** → σ rose, loss became nonzero
  (real gradient) — but eval still identical. Suspicious.

### Attempt 2 — gold-SFT warm-start → revealed THE bug
Warm-start SFT on the gold bodies converged hard (loss 2.46 → **0.006**) yet eval
was *still* identical to base. A memorized model not reproducing its golds made no
sense — so it wasn't the model.

**THE REAL BUG (root cause of every flat run): a broken splice.** Models emit the
WHOLE unit — `can walk_day with Day entry { <body> }` — but the template hole sat
*inside* the unit: `can walk_day ... { __HOLE__ }`. Splicing the full unit into the
inner hole produced **nested, broken Jac** (`can {...can {...}...}`) → never ran.
So base, warm (loss 0.006!), and all three GRPO adapters scored identically because
the **splice, not the model, was broken**. Every "RL is flat" conclusion before
this was measuring nested garbage.

**Fix:** `unwrap_unit` in `extract_jac` — when a completion is one wrapped unit,
take its inner block before splicing. Verified: warm model emits the gold body →
unwrap → splice → exit 0, output == expected.

### After the fix — the true baseline
With the splice fixed, **runs == pass** (runnable completions are now correct).
- `jac-qwen3coder` base: holdout 14.3% / train 8.3% pass — real numbers.
- The jac-trained base **already generates near-gold** completions, so warm-start
  adds almost nothing (only whitespace differs). Its misses are small Jac slips —
  `Missing ';'`, `Missing ']'`, `here.jid` instead of `jid(here)`.

### Attempt 3 — GRPO with the working reward → still no greedy change
Re-ran GRPO on `jac-qwen3coder` (LR 1e-5, 300 it, fixed reward). Reward now had
real variance (σ up to 0.11), loss nonzero (0.02–0.05) — gradient flowing — but
**+grpo eval identical to base** and KL≈0. Consistent finding: **LoRA-GRPO at
feasible LR/iters does not move a 30B's argmax (greedy) decoding.**

### Attempt 4 — fresh bases (where warm-start has room)
- **`qwen3coder` (fresh):** base holdout **0%** → **14.3%** after gold-SFT
  warm-start. The real, measured win — the fresh base couldn't pass any holdout;
  warm-start taught it one. GRPO on top added nothing.
- **`qwen36` (dense 27B):** **untrainable on 48 GB.** Warm-start SFT OOM'd at iter 1
  across every config (16/8/4 layers, batch 1, seq 2048→768, grad-checkpoint). A
  dense model activates all 27B params per token → activation memory exceeds 48 GB
  regardless of LoRA size. Inference fits; training doesn't. Base-eval only:
  holdout 0%, train run 12.5% / pass 0%.

## The numbers (pass % = exact stdout match; holdout 7, train 24)

| model | base ho/tr | +warm ho/tr | +grpo ho/tr |
|---|---|---|---|
| jac-qwen3coder (MoE, SFT+DPO) | 14.3% / 8.3% | near-ceiling | 14.3% / 8.3% |
| qwen3coder (MoE, fresh) | 0% / 8.3% | **14.3%** / 8.3% | 14.3% / 8.3% |
| qwen36 (dense 27B) | 0% / 0% | OOM | OOM |

## Honest findings

1. **The real blocker was a splice bug, not RL.** Most of the "RL doesn't work"
   story was a measurement artifact. Fixed.
2. **Warm-start (gold-SFT) helps where the base has room** — fresh model 0% → 14.3%.
3. **LoRA-GRPO changed no greedy outputs** at feasible LR (≤1e-5)/300 iters on any
   model. Reward variance + nonzero loss were present, but KL≈0 and greedy eval was
   identical. The supervised warm-start did the work, not the RL.
4. **Hardware:** MoE 30B-A3B trains on 48 GB (~38 GB peak); dense 27B does not.
5. **The metric is coarse:** exact full-stdout match at 31 tasks (base 0–14%) gives
   almost no positive signal. `near-pass` / `avg-osim` were added to see partials.

**Net:** harness, reward, warm-start, and eval are all correct and proven on real
30B models. The measured RL gain is the warm-start lift; GRPO-on-LoRA added nothing
measurable at this scale, and the dense model is untrainable on this box. Modest,
fully-recorded, nothing oversold.

---

## Forward plan (Phase 2.5 — approved)

The honest results point at specific fixes. In order:

1. **Model swap:** drop the untrainable dense 27B for **`Qwen/Qwen3.6-35B-A3B`**
   (MoE, 3B active → trains like the 30B-A3B). The dense was the wrong pick for a
   48 GB box.
2. **pass@k / sampled eval:** greedy eval can't see RL gains (RL shifts the
   distribution; argmax may not flip). Add sampled pass@k (`JAC_EVAL_K`) and
   **re-measure the existing adapters** — GRPO may have helped under the greedy
   blind spot.
3. **Grow tasks 31 → 50:** more (and a real holdout) for denser signal +
   generalization.
4. **STaR warm-start loop** (`rl/run_star.sh`): iterate the thing that works —
   SFT on gold → sample → keep passing → SFT again → repeat. Self-improving.
5. **Graded scoring tiers:** exact / near / runs, so the model gets credit for
   getting close (more gradient than all-or-nothing).
6. **In-process jac runner** (optional, gated): a warm worker pool that resets Jac
   machine state per snippet, to kill the subprocess import tax — only if it passes
   a 100× determinism gate (else stay on subprocess; the persistent-root +
   archetype-name-collision trap is why we use subprocess today).

Then run all 3 with pass@1 + pass@8 + tiers, and decide on heavier RL
(full-finetune, GRPO variants) only after seeing that data — not before.

*This document and [`../rl/runlog.md`](../rl/runlog.md) are updated with the new
numbers as Phase 2.5 runs land.*
