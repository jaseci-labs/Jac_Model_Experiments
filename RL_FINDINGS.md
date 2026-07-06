# RL Findings — Jac Code Generation

Everything we know about making a 30B coder model write idiomatic, compiler-correct
Jac — the definitions, the findings, the deployable answer. Model under study:
`jac-qwen3coder` (Qwen3-Coder-30B-A3B, already SFT+DPO'd on Jac), MLX on 48GB.

## TL;DR

**The model is capable; the real problem is a closeable *syntax* gap, not a capability wall.**

- **best-of-k + the jac compiler as verifier ships ~94%** on meaningful pure-function
  tasks, **today, zero training** — sample k, keep the one that compiles+runs; the
  compiler is a perfect picker.
- **SFT works:** greedy **39% → 61%** (peak rung-20), and the lift **holds at a bigger,
  fresher holdout (n=32)** and **generalizes to unseen tasks** — it isn't memorizing.
- **The gap is syntax, not logic:** if the model's Jac compiles, it's almost always
  exactly right. Greedy lands on an almost-valid variant; sampling reaches the answer.
- **Conversion (python→jac) is the best task framing** — base 64% greedy, conv-SFT
  **73% greedy / 82% best-of-k**.
- **GRPO ≈ SFT** (no extra lift); raw-GRPO from the base moves nothing. The fresh
  (non-Jac) model is a dead end.
- **The one real gap: free-form NL generation** — both models are tuned for the task
  format and fail arbitrary "write a function that…" prompts.

---

## Legend

| Term | Meaning |
|---|---|
| **pass@1 / greedy** | one deterministic best guess; % of holdout that is byte-exact correct. The headline. |
| **pass@8 / sampled** | 8 sampled tries; pass if **any** compiles+runs+matches. The *reachable ceiling*. |
| **best-of-k (deploy)** | sample k, return the first that the jac **compiler** accepts — no gold peek. The shippable number. |
| **syntax gap** | pass@8 − pass@1. Correct-but-not-greedy capability the model already has. |
| **holdout** | fixed tasks never trained on; the generalization set (n=16–32 across runs). |
| **mem** | train-recall — eval on the rung's own training tasks. Gauges overfitting. |
| **rung** | how many tasks we trained on in one ladder step: 5, 20, all. Each is a superset of the last. |
| **hole-fill task** | a real `this_is_jac` program with one function body blanked; the model fills it. |
| **conversion task** | "translate this Python function to Jac" — same grader, richer/unambiguous spec. |
| **synthetic task** | a fresh, authored deterministic Jac function (not from `this_is_jac`) — for holdout power + novelty. |
| **osim** | output similarity (`difflib` ratio) between produced and gold stdout (0–1). |
| **SFT / GRPO / DPO / LoRA** | supervised FT / the RL method / preference-tuning / low-rank (48GB-forced) adapter. |
| **distillation / expert-iteration** | train on a stronger teacher's / compiler-verified correct answers — adds capability the student can't self-produce. |
| **boundary** | the best the model can do *with* sampling (pass@k ceiling). |
| **the two models** | `qwen3coder` (fresh) · `jac-qwen3coder` (SFT+DPO'd on Jac — the capable one). |

---

## Results — the full run (jac-qwen3coder, pure-fn holdout n=18)

| cell | greedy pass@1 | oracle pass@8 | best-of-k deploy |
|---|---|---|---|
| base | 38.9% | 72.2% | **72.2%** |
| SFT rung-5 | 55.6% | **83.3%** | — |
| **SFT rung-20** | **61.1%** ← peak greedy | 72.2% | — |
| SFT rung-all | 55.6% | 72.2% | **77.8%** |
| SFT + GRPO (rall) | 55.6% | 77.8% | 77.8% |
| raw-GRPO control | 38.9% | 72.2% | — |

- **SFT closes the greedy gap** — 39%→61% at the sweet spot (rung-20). The knowledge is
  reachable (67% sampled); SFT makes it greedy-default.
- **best-of-k deploy 72%→78%** after SFT; deploy == oracle throughout (compiler is a perfect picker).
- **GRPO adds nothing over SFT**; **raw-GRPO from base = base** — GRPO needs the SFT warm-start.
- **Sweet spot is rung-20, not "all":** rung-all drops one task (`lib_log`) to task
  interference from the harder graph walkers. More tasks ≠ better.

---

## Findings

Each: **what we saw → what it means.**

### F1 — True base accuracy is 33–39% greedy; the jac model is the one to use
Fresh 33%, `jac-qwen3coder` 39% greedy. → the fresh model can't write Jac (see F3); all
work is on `jac-qwen3coder`.

### F2 — Huge sampling gap: 39% greedy → 67% sampled (+28pp)
The correct Jac is *reachable* — the model produces it among 8 samples — but greedy
decoding lands on an almost-valid variant that won't compile. → the target isn't "make
the model smarter," it's "make the already-reachable answer the greedy default." Far easier.

### F3 — The fresh model has *no* gap (33% → 33%)
pass@1 == pass@8 for the fresh model: sampling finds nothing it doesn't already emit
greedily → it can't write Jac syntax at all. **Dead end; use `jac-qwen3coder`.**

### F4 — Failures are compile-fails, not wrong answers
`osim`/norm/pass/runs all move together: if the model's Jac **runs**, it's almost always
**exact**; when it fails, it **didn't compile**. → (a) exact-stdout is a fine metric here;
(b) SFT teaching syntax moves greedy a lot; (c) **the jac compiler is a perfect free verifier.**

### F5 — Conversion (python→jac) beats hole-fill by +11pp
On identical functions: conversion 56% vs hole-fill 44% greedy — the Python is an
unambiguous spec + a single transferable skill. → better task design is a real lever.
(Conversion only sources pure functions; the graph-walker idiom needs its own task.)

### F6 — SFT works and generalizes; GRPO ≈ SFT
SFT lifts greedy 39%→61% (F-run) and 34%→44% on the bigger fresh holdout, **generalizing
to unseen tasks** (not memorizing). GRPO with a working reward reinforces the model's own
correct samples but lands at the same accuracy as SFT — no extra lift at this scale/LoRA.

---

## Full program — all levers (jac-qwen3coder)

| lever | result |
|---|---|
| **best-of-k + compiler verifier** | **the universal win** — 72–78% (pure-fn), 65% (graph walkers), **~94% (clean)**, 89% at k=32. Works on every holdout; deploy == oracle. |
| **SFT** | greedy +22pp (39→61) on pure-fn, +25pp (44→69) clean, +9pp on the bigger fresh holdout. Sweet spot rung-20. |
| **conversion framing** | the peak — base 64% → SFT **73% greedy / 82% best-of-k**. |
| **GRPO** | ≈ SFT (no extra lift); raw-GRPO from base = nothing. Not the lever at 48GB/LoRA. |
| **fresh model** | dead (gap 0 — can't write Jac). |
| **graph-idiom (OSP walkers)** | best-of-k works (53→65% w/ SFT); greedy doesn't improve — SFT moves *sampling* on hard tasks, not the greedy default. |
| **free-form NL prompts** | both models 0/3 — the one real gap; would need SFT on NL→jac. |

**The deployable answer:** **conversion prompting + SFT (rung-20) + best-of-k with the
jac compiler as verifier → ~82% on pure functions, ~94% on the clean set.** Shipped as
`rl/generate.py`.

### Extensions (tightening the result)
- **Drop the 2 junk tasks** (`lib_fstring_rules`/`lib_innerstring_rules` — verbatim-regex
  memorization, not real Jac) → the true best-of-k ceiling is **93.8%** (they masked it at 72%).
- **Exploit vs explore:** SFT sharpens toward greedy (best single-shot) but slightly
  narrows sampling diversity (base best-of-k 94% ≥ SFT 88%). Want one shot → SFT; want k
  samples → base + best-of-k.
- **Bigger, fresher holdout (n=32, +16 synthetic):** the SFT lift holds and generalizes
  to novel tasks; fresh tasks are harder (base 25% vs 44% on familiar) → the numbers are
  honest, not optimistic.

---

## Shipped

- **`rl/generate.py`** — the best-of-k Jac generator (sample k, return the first the jac
  compiler accepts). Live-verified.
- **Studio RL section** — the story (11→94% journey), the ladder, best-of-k across 5
  holdouts, k-scaling, and a live **GENERATE JAC** panel. Backed by `get_rl_corrected()`
  → `resultspub/rl/corrected_summary.json` (assembled by `rl/make_summary.py`).
- **Studio Data pipeline** — RL datasets browsable by use: SFT gold · GRPO reward tasks ·
  conversion · synthetic · splits.
- **Graphs** — `resultspub/rl/` (journey, ladder, all-holdouts, k-scale, follow-up).
- **32 synthetic tasks** — `rl/drivers/syn_*.jac`, ingested by `rl/build_tasks.jac`.
- **Raw record** — `docs/rl/raw/`, `results/corrected_*.jsonl`.

---

## Open leads (not yet done)

- **NL→jac SFT** — close the free-form gap (the one real weakness).
- **Bigger generated dataset** — n is still small (±8–12pp noise); synthetic families +
  distillation would tighten every number and give more headroom.
- **Full fine-tune** (cloud / >48GB) — the only untested way to check if GRPO can beat SFT
  once LoRA capacity is off the table.

---

# APPENDIX A — How far we came

The headline number moved a long way. This is the arc and the one measurement lesson
behind it, kept so the progress is legible.

### The journey (holdout accuracy)

| stage | accuracy | what changed |
|---|---|---|
| earliest read | **11%** | first measurement (see the lesson below) |
| true base greedy | **33–39%** | measured correctly |
| + SFT (greedy) | **61%** | training closes the syntax gap |
| + best-of-k | **78%** | compiler picks the right sample |
| + drop junk tasks | **94%** | on meaningful pure-function tasks |

`resultspub/rl/corrected_journey.png` renders this; the Studio RL section shows it live.

### The one measurement lesson

The earliest ~11% numbers were an **eval artifact**, not the model. The grader's
body-extractor (`extract_jac`/`unwrap_unit`) grabbed the driver *docstring* whenever the
model echoed the whole driver file, splicing garbage into the hole → auto-fail —
undercounting accuracy **~3.5–4×**. The GRPO reward shared the same extractor, so early
RL trained on bad signal. Fixing it (name-targeted, brace-matched extraction — commit
`8164ee2`) revealed the true picture: the model was always ~4× more capable than the raw
number showed, and SFT does move it.

Takeaway: **verify the grader before trusting a flat result.** A null can be a broken
ruler. Everything in this document is post-fix.

### Before → after (pure-fn holdout, jac-qwen3coder)

| metric | earliest read | now |
|---|---|---|
| base greedy | 11% | **39%** |
| SFT greedy | "no change" | **61%** |
| best-of-k deploy | (not measured) | **78–94%** |
| verdict | "RL is a dead end, models can't do Jac" | **capable model + a shipped ~80–94% generator** |

The full pre-fix write-up (the original ladder findings F1–F8, verbatim) is archived at
[`docs/rl/RL_FINDINGS_v1_invalidated.md`](docs/rl/RL_FINDINGS_v1_invalidated.md) — kept
for the reasoning trail, superseded by everything above.

*References: Yue et al. 2504.13837 (RL sharpens sampling, doesn't expand the boundary);
ProRL 2505.24864 (prolonged RL can expand, but needs full-FT + 1000s of tasks); Spurious
Rewards 2506.10947. Full notes: `docs/rl/references.md`.*
