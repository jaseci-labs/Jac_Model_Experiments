# Jac Coding Agent — Attempts at a Model That's Really Good at Jac

The goal: a coding agent for **Jac** (Jaseci Labs) — what Claude Code is for Python.
Generate, debug, explain, and convert to **idiomatic, compiler-correct Jac**, deployed
via the Jac MCP inside coding assistants. Quality bar: **compiles + runs + idiomatic**,
not "Jac-looking." No real Jac corpus exists to scrape, so every attempt below trains on
**100% synthetic, compiler-validated data**.

This repo is organized as a series of **attempts**, each in its own folder, each building
on what the last one learned:

| Attempt | Method | Status | Headline result |
|---|---|---|---|
| **[`model-experiments/01-sft-dpo/`](model-experiments/01-sft-dpo/)** | Supervised finetuning + DPO | done | stock model **0%** runnable Jac → **94%** after one LoRA pass |
| **[`model-experiments/02-rl-grpo/`](model-experiments/02-rl-grpo/)** | RL (GRPO) on top of attempt 1's model | done | best-of-k + compiler-as-verifier ships **~94%**; GRPO ≈ SFT, no extra lift |
| **[`model-experiments/03-cpt-only/`](model-experiments/03-cpt-only/)** | Continual pretraining (CPT) on raw Jac/OSP docs, before SFT | done, **rejected** | CPT-v2 0/3 acceptance gates cleared — fabricates plausible wrong-domain syntax instead of admitting uncertainty; a structural limit of next-token CPT on doc prose, not a bad run |
| **[`model-experiments/04-cpt-sft/`](model-experiments/04-cpt-sft/)** | SFT (+DPO) on top of the rejected CPT-v2 checkpoint vs. a fresh base, identical recipe both arms | done | CPT-v2's real base-stage edge (**+37pp**) is statistically absorbed by SFT (**+2.8pp**, p≈0.20) — but survives *structurally* in the SFT adapter's own weight geometry |
| **[`model-experiments/05-cpt-sft-grpo/`](model-experiments/05-cpt-sft-grpo/)** | GRPO on top of both attempt-4 arms — closes the loop on the original 4-stage CPT architecture | docs scaffolded, not yet built | does CPT change GRPO's ceiling, not just SFT's? |

Shared across all attempts, at repo root: `models/` (base + merged checkpoints,
gitignored), `docs/` (repo-wide strategy + the adapter-hyperparameter registry),
`jms/` (Jac Model Studio, JMS — the app that reads results from every attempt),
`papers/` (reference papers). The attempts themselves live under
`model-experiments/`; `model-experiments/02-rl-grpo/dataset/this_is_jac/` is the
real Jac codebase RL mines tasks from.

---

## Table of contents

- [What Jac is (and why models fail at it)](#what-jac-is-and-why-models-fail-at-it)
- [Attempt 1 — SFT + DPO](#attempt-1--sft--dpo)
- [Attempt 2 — RL / GRPO](#attempt-2--rl--grpo)
- [Attempt 3 — CPT (rejected)](#attempt-3--cpt-rejected)
- [Attempt 4 — SFT/DPO on CPT-v2 vs. fresh](#attempt-4--sftdpo-on-cpt-v2-vs-fresh)
- [Attempt 5 — GRPO, closing the loop](#attempt-5--grpo-closing-the-loop)
- [Repository layout](#repository-layout)
- [Environment](#environment)
- [Documentation map](#documentation-map)
- [Glossary](#glossary)

---

## What Jac is (and why models fail at it)

Jac is a programming language built **on top of Python** with a **data-spatial /
object-spatial** model (OSP): computation is expressed with **nodes, edges, walkers,
and abilities** instead of plain functions and classes. It compiles to Python and
interops with the ecosystem, but its idioms are distinct enough that a model trained
on Python/JS/C has a **very weak prior** on correct Jac.

| Jac construct | Role | Python analogue |
|---|---|---|
| `walker` | a traversal agent that moves through the graph | (no direct equivalent) |
| `node` / `edge` | graph primitives — data + typed connections | object + reference |
| `can … with <Node> entry` | an **ability** — event-triggered behavior | method (sort of) |
| `def` | a plain method | method |
| `obj` | preferred data archetype | `class` |
| `with entry` | module entry block | `if __name__ == "__main__"` |
| `spawn` / `++>` / `visit [-->]` / `disengage` | launch a walker / create edge / traverse / stop | — |
| `has` | typed field declaration | typed attribute |

A non-finetuned model produces Python-shaped code that *looks* plausible but is
syntactically or semantically wrong Jac. Closing that gap — cheaply, verifiably — is
the whole project. Every attempt shares one non-negotiable rule: **the gate is `jac
run`, never `jac check`** — "correct" means compiles, executes, and its output matches
recorded behavioral test cases, not just that the type-checker is happy (idiomatic
Jac is often untyped-but-runnable, and `jac check` over-rejects it).

---

## Attempt 1 — SFT + DPO

**[`model-experiments/01-sft-dpo/`](model-experiments/01-sft-dpo/)** — the first attempt: prove that supervised finetuning
on synthetic, compiler-validated data can take a model from zero to mostly-correct
Jac, then use DPO to push the *idiomatic* (not just correct) style on top.

### The idea

Three anchors substitute for a real-data distribution:

1. **Jac grammar** = the distribution anchor — every construct must appear in the data.
2. **Jac compiler + cross-compiled tests** = an unlimited oracle — rejection sampling
   is free, and a **behavioral test pass** is the real gate, not mere compilation.
3. **Python** = the proxy distribution — translate validated Python → idiomatic Jac
   (the **MultiPL-T** methodology).

Data pipeline: mine runnable functions from `Vezora/Tested-22k-Python-Alpaca` →
transpile (`jac py2jac`) with a jac-run gate for volume (`sft_auto.jsonl`, 1500) → hand
/ agentically-written idiomatic examples including graph-tier node/edge/walker tasks
(`sft.jsonl`, 147) → DPO pairs of idiomatic (chosen) vs. transpiled Python-shaped
(rejected) versions of the same function (`dpo.jsonl`, 147). Everything is written in
Jac itself — see [`model-experiments/01-sft-dpo/sft_dpo/jacgen/`](model-experiments/01-sft-dpo/sft_dpo/jacgen/) (24
modules: generate, validate, dedup, decontaminate, split, eval harness).

### Results

Base model: **Qwen3-Coder-30B-A3B-Instruct** (chosen after a 6-model bake-off — see
below). Measured on a decontaminated, disjoint holdout.

| stage | function-tier test-pass (n=150) | graph-tier correct (n=13) |
|---|---|---|
| **base** (stock model) | **0%** | **0%** |
| **SFT** | **94%** | 46% |
| **DPO** | 93% | **61%**, 100% of correct outputs idiomatic |

- **Function tier:** a stock model produces essentially zero runnable Jac; one LoRA-SFT
  pass takes it to 93–94% behaviorally correct. On pure functions the model learns to
  **transpile** (Python-shaped but correct) — there's no idiom headroom to push on;
  `factorial` written idiomatically *is* the mechanical transpile.
- **Graph tier** is where idiom actually diverges from transpile. SFT gets Qwen to 46%
  correct (mostly already idiomatic); **DPO lifts correctness to 61% and makes 100% of
  correct outputs idiomatic**, pulling transpile-similarity down from 0.457 toward the
  0.26 idiomatic reference.
- **Base-model bake-off:** before committing the full generation budget, the same
  SFT+DPO treatment ran on 5 more same-size candidates (Qwen3-30B-Instruct, gpt-oss-20b,
  DeepSeek-Coder-V2-Lite, Qwen2.5-Coder-14B, Ling-Coder-lite) to confirm Qwen3-Coder was
  the right base to invest in. **Verdict: kept Qwen3-Coder** — no candidate beat it on
  behavioral pass-% beyond run-to-run noise, and its DPO graph score (61%) was the best
  of any DPO-capable model. Full matrix →
  [`model-experiments/01-sft-dpo/docs/initmodelchoice/2026-06-26-sft-dpo-bakeoff-results.md`](model-experiments/01-sft-dpo/docs/initmodelchoice/2026-06-26-sft-dpo-bakeoff-results.md).

Full results, all 16 training graphs, side-by-side model comparison →
**[`model-experiments/01-sft-dpo/resultspub/initmodelchoice/RESULTS.md`](model-experiments/01-sft-dpo/resultspub/initmodelchoice/RESULTS.md)**.

### Run it

```bash
./setup_env.sh && source .venv/bin/activate
./model-experiments/01-sft-dpo/sft_dpo/check.sh                                                # type + behavioral gate, non-destructive
./model-experiments/01-sft-dpo/sft_dpo/run_probe.sh Qwen/Qwen3-Coder-30B-A3B-Instruct qwen      # quantize → base eval → train → fuse → finetuned eval
./model-experiments/01-sft-dpo/sft_dpo/run_dpo.sh qwen                                          # DPO stage on top of the SFT adapter
```

Full docs → operator runbook
[`model-experiments/01-sft-dpo/sft_dpo/process.md`](model-experiments/01-sft-dpo/sft_dpo/process.md), architecture handoff
[`model-experiments/01-sft-dpo/docs/sft_dpo/modeltesting/HANDOFF.md`](model-experiments/01-sft-dpo/docs/sft_dpo/modeltesting/HANDOFF.md),
pipeline reference [`model-experiments/01-sft-dpo/sft_dpo/jacgen/README.md`](model-experiments/01-sft-dpo/sft_dpo/jacgen/README.md).

---

## Attempt 2 — RL / GRPO

**[`model-experiments/02-rl-grpo/`](model-experiments/02-rl-grpo/)** — starting from attempt 1's SFT+DPO'd model
(`jac-qwen3coder`), the second attempt asked whether **RL (GRPO)** could push
correctness further, using the Jac compiler itself as a free, verifiable reward (no
learned reward model). Full story with every number and every bug:
**[`model-experiments/02-rl-grpo/RL_FINDINGS.md`](model-experiments/02-rl-grpo/RL_FINDINGS.md)**.

### The headline

> **The model was already capable; the real problem was a closeable *syntax* gap, not
> a capability wall — and for three weeks a measurement bug made it look like neither
> of those things was true.**

- **best-of-k + the Jac compiler as verifier ships ~94%** on meaningful pure-function
  tasks, **today, zero extra training** — sample k completions, keep the first one
  that compiles and runs; the compiler is a perfect picker since compiles ⟹ almost
  always exactly right.
- **SFT works:** greedy pass@1 **39% → 61%** (peak at 20 training examples), and the
  lift holds on a bigger, fresher holdout and generalizes to unseen tasks.
- **GRPO ≈ SFT** — adds no measurable lift once SFT has already moved greedy decoding
  close to the model's own sampling ceiling. Raw GRPO from a fresh (non-Jac-trained)
  base moves nothing at all — RL can't bootstrap a skill the base model has zero of.
- **The one real gap:** free-form natural-language prompts (no starter code) — both
  models score 0/3, since neither was trained on that input distribution.

### Why the numbers moved so much: three eras

The measured headline number went **14% → 11% → 39% → 61% → 78% → 94%** over about
two weeks. Most of that motion was not the model improving — it was three rounds of
fixing *how the eval measured it*.

1. **Era 1 (Jun 20–21) — weekend GRPO, flat at 14.3%.** Built a real
   compiler/runtime-verified GRPO reward on MLX LoRA. Hit and fixed three real bugs
   along the way: a Metal OOM (config, not fundamental), the **σ=0 trap** (a GRPO
   group with 0% pass rate has zero reward variance → zero gradient at any learning
   rate — fixed with a similarity-based reward term that's non-zero even for failing
   completions), and a **splice bug** (the model's output was being nested inside an
   already-enclosing unit before compiling, so *everything* looked broken regardless
   of the model). After fixing all three, the real result was: LoRA-GRPO barely moves
   a 30B model's greedy output at a feasible learning rate. Verdict at the time
   (correct, for this attempt): "supervised levers move the model; RL doesn't — yet."
2. **Era 2 (Jun 25–28) — a proper 30-cell SFT/GRPO ladder, still flat.** A leak-free
   ladder (train-N ∈ {1,3,5,10,20,all} × {base, SFT, SFT+GRPO, raw-GRPO, tuned-GRPO} ×
   2 models) came back **exactly flat** in every cell, on three different corpora.
   Declared "RL is a dead end" (v1 verdict, Jun 28) — a suspiciously clean flat line
   that was actually the tell something was wrong with the measurement, not the model.
3. **Era 3 (Jul 1–2) — the correction.** The eval script and the GRPO reward shared
   one extraction helper. When the model echoed back its entire surrounding driver
   file (common, otherwise harmless), that helper grabbed the driver's **docstring**
   instead of the model's actual answer — an auto-fail baked into every measurement,
   a clean **~3.5× undercount** with nothing to do with model capability. Worse: since
   the *reward* used the same buggy helper, Era 1 and 2's RL runs had also been
   **trained** against a partially garbage signal the whole time. Fixed in one commit;
   re-measured on the same holdout: 11.1% → **38.9%** for the SFT+DPO'd model.

**Takeaway carried forward:** verify the grader before trusting a null result. A flat,
convincing-looking null can be a broken ruler, not a finding.

### Corrected results (post-fix, trustworthy)

Pure-function holdout, `jac-qwen3coder` (already SFT+DPO'd from attempt 1):

| cell | greedy pass@1 | oracle pass@8 | note |
|---|---|---|---|
| base | 38.9% | 72.2% | true floor once measured correctly, not zero |
| SFT rung-5 | 55.6% | **83.3%** | a small, low-conflict sample already teaches most of the syntax |
| **SFT rung-20** | **61.1%** (peak) | 72.2% | sweet spot — enough coverage, not yet enough cross-task conflict |
| SFT rung-all | 55.6% | 77.8% | **task interference** — a bigger, more varied mix regresses an already-learned task |
| SFT + GRPO | 55.6% | 77.8% | flat vs. SFT alone |
| raw-GRPO (fresh base) | 38.9% | 72.2% | equals base exactly — GRPO alone can't manufacture syntax the base doesn't have |

Deployable numbers, no further training needed — sample k, return the first the
compiler accepts:

| task family | best-of-k accuracy |
|---|---|
| conversion tasks | **82%** (peak) |
| pure functions | ~78% (94% on the cleanest subset) |
| graph-walker (OSP idiom) | 65% — the acknowledged weak spot |
| free-form NL prompts | 0% — untested gap, don't ship this path |

**Why the syntax gap is closeable for free:** failures are almost always
*compile*-fails (a missing `;`, `here.jid` vs `jid(here)`), not wrong logic — when the
model's Jac runs at all, it's almost always exactly right. That tight coupling is what
makes the compiler a perfect, zero-cost verifier: no learned reward model or ground
truth needed at inference time, just sample-and-check.

Shipped: **[`model-experiments/02-rl-grpo/rl/generate.py`](model-experiments/02-rl-grpo/rl/generate.py)** — the best-of-k
generator; the live Studio RL section (11%→94% journey, ladder, k-scaling, a GENERATE
JAC panel), backed by
[`model-experiments/02-rl-grpo/resultspub/rl/corrected_summary.json`](model-experiments/02-rl-grpo/resultspub/rl/corrected_summary.json).
Graphs → [`model-experiments/02-rl-grpo/resultspub/rl/`](model-experiments/02-rl-grpo/resultspub/rl/).

### Run it

```bash
jac run model-experiments/02-rl-grpo/rl/build_tasks.jac           # this_is_jac/ drivers -> tasks + templates
jac run model-experiments/02-rl-grpo/rl/build_rl_splits.jac       # fixed holdout + trainpool
jac run model-experiments/02-rl-grpo/rl/run_ladder.jac            # DRY: prints the plan, runs nothing heavy
JAC_LADDER_GO=1 jac run model-experiments/02-rl-grpo/rl/run_ladder.jac   # execute the ladder (hours per cell)
jac run model-experiments/02-rl-grpo/rl/show_ladder.jac           # pivot results into a curve table
```

Full pipeline reference (reward design, warm-start, the recommended
compute-smart execution order, gotchas) → **[`model-experiments/02-rl-grpo/rl/README.md`](model-experiments/02-rl-grpo/rl/README.md)**.

---

## Attempt 3 — CPT (rejected)

**[`model-experiments/03-cpt-only/`](model-experiments/03-cpt-only/)** — attempts 1–2 closed
with SFT fixing syntax but problem-pass flat around 40%, and GRPO never beating SFT on greedy
pass@1. Hypothesis: the shared ceiling is syntax-bound because the model never had a domain-
adaptation stage to learn Jac/OSP *semantics* — SFT few-shot examples teach pattern-matching,
not the underlying concept. Locked architecture: `base → +CPT → +CPT+SFT/DPO →
+CPT+SFT/DPO+GRPO`, LoRA continual-pretrain on raw Jac docs/OSP paper/blogs, mixed with
general-code rehearsal for catastrophic-forgetting insurance.

- **CPT-v1** (2026-07-14): trained clean (loss down, no OOM, CF-check 16/16), but **NULL** —
  byte-identical MCQ concept-recognition scores before/after (18/20 both, same 2 wrong). Moved
  free-form generation *vocabulary* (real Jac keywords appear) but not *understanding*.
- **CPT-v2** (2026-07-20): redesigned corpus (dropped code entirely, shrunk rehearsal to ~10%),
  added a curation pass and a 12-epoch checkpoint-loop past CPT-v1's cosine-schedule ceiling.
  Trained clean to the full 12-leg budget. **REJECTED — 0 of 3 acceptance gates cleared**:
  Track A (cosine-to-oracle) margin vs. base +0.008 (need ≥0.03), vs. CPT-v1 +0.0007
  (statistically indistinguishable from noise); Track B (blind pairwise judge) — the oracle
  beat CPT-v2 in 91/100 judgments (need ≥50% win-or-tie).
- **Root cause** (`model-experiments/03-cpt-only/docs/cpt-2/analysis.md`): not a bad run — corpus
  dilution, instrument mismatch, and LoRA rank-16 capacity were all ruled out. The model
  **fabricates plausible, fluent, wrong-domain syntax** (invents Jac constructs, misidentifies
  the framework as Next.js/Neo4j/Cypher) instead of admitting uncertainty — a structural limit
  of next-token CPT on doc prose, not something a 3rd attempt would fix.
  **Recommendation: skip further CPT, go straight to SFT/DPO on the base model.**

Full story → **[`model-experiments/03-cpt-only/docs/cpt-2/analysis.md`](model-experiments/03-cpt-only/docs/cpt-2/analysis.md)**
(root cause), [`results.md`](model-experiments/03-cpt-only/docs/cpt-2/results.md) (acceptance gates), [`design.md`](model-experiments/03-cpt-only/docs/cpt-2/design.md) (the locked architecture).

---

## Attempt 4 — SFT/DPO on CPT-v2 vs. fresh

**[`model-experiments/04-cpt-sft/`](model-experiments/04-cpt-sft/)** — CPT-v2 was rejected on
its own instrument, but the question of whether it still helps once real task-specific training
lands on top of it was worth an independent, differently-instrumented answer rather than
treating attempt 3 as fully closed. Two arms — `sft_cptv2_probe/` (CPT-v2 adapter under the
hood) and `sft_fresh_probe/` (unmodified base) — trained with the byte-identical SFT/DPO recipe
and dataset/holdout, differing in exactly one variable.

| Stage | fresh (no CPT) | CPT-v2 | Δ |
|---|---|---|---|
| base (no SFT) | 10.5% (90/855) | 47.3% (404/855) | **+37.0pp** |
| +SFT (8200 iters) | 69.8% (597/855) | 72.6% (621/855) | +2.8pp (p≈0.20, not significant) |
| +SFT+DPO (corrected, best ckpt) | 69.8% (597/855, ties SFT) | 71.7% (613/855) | ~0pp |

- **CPT-v2 gives a real, large base-stage head start** (+37pp) — but that advantage does **not**
  survive as a statistically confirmed edge once SFT trains on top. A fresh base put through
  the identical SFT recipe performs indistinguishably from the CPT-v2 arm at every post-training
  stage.
- **A measurement bug, found and fixed:** the first DPO pass showed both arms collapsing to
  ~12% — traced to `mlx_lm.fuse` silently dropping ~15% of the packed 4-bit weight deltas when
  re-quantizing the SFT adapter, so every DPO run had been training on an effectively un-SFT'd
  base. Fixed (fuse-free, `--resume-adapter-file`-seeded); corrected DPO caps out at SFT parity
  — doesn't beat it, and regresses ~8pp if run to its full 250-iter budget past the early-step
  peak.
- **A structural finding SFT's functional numbers don't show:** a q_proj LoRA singular-value
  probe (`lora_svd_qproj.py`) found the CPT-v2-base SFT adapter's rank-concentration and
  dominant-direction magnitude still track the standalone CPT-v2 adapter far more closely than
  they track the fresh-base SFT adapter — trained adapters are genuinely rank-1-concentrated
  (stable rank ~3-6 of a 16-dim budget), and CPT-v2's fingerprint survives SFT *structurally*
  even though it's statistically invisible *behaviorally*.

Full results → **[`model-experiments/04-cpt-sft/RESULTS.md`](model-experiments/04-cpt-sft/RESULTS.md)**, deep-dive
phase-by-phase (including the fuse bug and the SVD finding) →
**[`model-experiments/04-cpt-sft/docs/reports/2026-07-cpt-vs-fresh-comparison.md`](model-experiments/04-cpt-sft/docs/reports/2026-07-cpt-vs-fresh-comparison.md)**.

---

## Attempt 5 — GRPO, closing the loop

**[`model-experiments/05-cpt-sft-grpo/`](model-experiments/05-cpt-sft-grpo/)** — attempt 3's
original locked architecture was `base → +CPT → +CPT+SFT/DPO → +CPT+SFT/DPO+GRPO`; attempt 4
covered base/SFT/DPO for both arms but left the GRPO stage unrun. Attempt 5 runs it: does CPT
change GRPO's ceiling (not just SFT's, which absorbed it to statistical noise), and does
attempt 2's "GRPO ≡ SFT everywhere" null (confirmed 3× across two corpora) hold on a harness
upgrade — multi-source corpus (adds task-mining from the other 16 repos in attempt 3's
already-vetted 17-org code corpus) and Type-B AST-equivalence grading (replacing exact-stdout),
extending `02-rl-grpo/rl/` rather than rebuilding it.

4 GRPO training lines (fresh/CPT-v2 arms × warm-started-from-SFT/cold-control) × 6 rungs
(1/3/5/10/20/all) = 24 training runs, lineage-preserving (`--resume-adapter-file`, never
`mlx_lm.fuse` — the bug attempt 4 found and fixed). **Status: docs scaffolded, harness build and
training not started.** Design + full runbook →
**[`model-experiments/05-cpt-sft-grpo/docs/`](model-experiments/05-cpt-sft-grpo/docs/)**
(`README.md` → `strat.md` → `spec.md` → `workflow.md`, plus an in-depth `dataset/spec.md` +
`dataset/workflow.md` for the corpus-mining pipeline).

---

## Repository layout

| Path | What |
|---|---|
| `model-experiments/` | parent dir for all five attempts (see rows below) |
| `model-experiments/01-sft-dpo/` | attempt 1 — code, dataset, adapters, results, docs (see [above](#attempt-1--sft--dpo)) |
| `model-experiments/02-rl-grpo/` | attempt 2 — code, dataset, adapters, results, docs, the RL slide deck (see [above](#attempt-2--rl--grpo)); harness (`rl/`) is reused/extended by attempt 5 |
| `model-experiments/03-cpt-only/` | attempt 3 — CPT-v1/v2 continual-pretrain, adapters, docs (see [above](#attempt-3--cpt-rejected)); **rejected**, but its checkpoint feeds attempt 4's CPT-v2 arm |
| `model-experiments/04-cpt-sft/` | attempt 4 — SFT/DPO on CPT-v2 vs. fresh base, code, dataset, adapters, results, docs (see [above](#attempt-4--sftdpo-on-cpt-v2-vs-fresh)) |
| `model-experiments/05-cpt-sft-grpo/` | attempt 5 — GRPO on both attempt-4 arms, docs scaffolded (see [above](#attempt-5--grpo-closing-the-loop)) |
| `models/` *(gitignored)* | base + merged/fused checkpoints, shared across attempts — each later attempt finetunes an earlier one's output |
| `results/` | JMS scratch space only (`_builder`, `_evals`) — per-attempt run outputs live inside each `model-experiments/0N-.../results/` |
| `docs/` | repo-wide: `training_configs/` (hyperparameter registry for every adapter, incl. deleted ones — see `docs/ARTIFACT_LOG.md`), `wholestack/` (end-to-end strategy spanning both attempts) |
| `jms/` | **Jac Model Studio (JMS)** — the app that visualizes/drives all of this (dataset browser, GENERATE panel, RL section, builder jobs) |
| `model-experiments/02-rl-grpo/dataset/this_is_jac/` | the real open-source Jac codebase attempt 2 mines RL tasks from |
| `context.md` | durable project framing (what Jac is, the goal, fixed constraints) |
| `papers/` | reference papers (MultiPL-T, WizardCoder, Magicoder, SelfCodeAlign, DeepSeek-Coder, CodeDPO, Magpie) |
| `setup_env.sh` | one-time venv + toolchain install (jaclang, mlx-lm, mlx-lm-lora, matplotlib) |

---

## Environment

**Anaconda was removed on purpose — do not reinstall it.** The project runs on a venv
over Homebrew `python3.14`:

```bash
./setup_env.sh                 # python3 -m venv .venv + pip install jaclang mlx-lm mlx-lm-lora matplotlib
source .venv/bin/activate      # puts jac + mlx_lm.* on PATH
```

- `jaclang` **0.16.0** (strict `Any` handling — Python-interop calls return `Any`,
  rejected in typed positions; cast at the boundary).
- `mlx-lm` (`mlx_lm.convert` / `lora` / `fuse` / `generate`).
- `mlx-lm-lora` **2.1.0** (DPO + GRPO — mlx-lm has no native support for either).
- `matplotlib` (PNG graphs), `caffeinate` (macOS built-in; keeps long runs awake).

You need **~50–60 GB free disk per model** (download + quantize). Everything runs on a
single Apple-Silicon Mac, 48 GB unified memory — the hard ceiling every experiment
design in both attempts had to respect.

---

## Documentation map

**Repo-wide**
| Doc | What |
|---|---|
| [`context.md`](context.md) | durable project framing — what Jac is, the goal, fixed constraints |
| [`docs/wholestack/strat.md`](docs/wholestack/strat.md) | end-to-end strategy spanning data gen → finetune → eval |
| [`docs/ARTIFACT_LOG.md`](docs/ARTIFACT_LOG.md) | record of every model/adapter, how to recreate any deleted one |
| [`docs/training_configs/`](docs/training_configs/) | hyperparameter JSON for every adapter trained across both attempts |

**Attempt 1 — SFT + DPO**
| Doc | What |
|---|---|
| [`model-experiments/01-sft-dpo/sft_dpo/process.md`](model-experiments/01-sft-dpo/sft_dpo/process.md) | operator runbook — setup → check → run, pause/resume, timings |
| [`model-experiments/01-sft-dpo/docs/sft_dpo/modeltesting/HANDOFF.md`](model-experiments/01-sft-dpo/docs/sft_dpo/modeltesting/HANDOFF.md) | single source of truth — architecture, every module, every gotcha |
| [`model-experiments/01-sft-dpo/docs/initmodelchoice/2026-06-26-sft-dpo-bakeoff-results.md`](model-experiments/01-sft-dpo/docs/initmodelchoice/2026-06-26-sft-dpo-bakeoff-results.md) | 6-model base bake-off, the keep-Qwen3-Coder verdict |
| [`model-experiments/01-sft-dpo/docs/initmodelchoice/strat.md`](model-experiments/01-sft-dpo/docs/initmodelchoice/strat.md) | the 12 data-generation recipes (R1–R12) |
| [`model-experiments/01-sft-dpo/resultspub/initmodelchoice/RESULTS.md`](model-experiments/01-sft-dpo/resultspub/initmodelchoice/RESULTS.md) | full measured results + all 16 training graphs |
| [`model-experiments/01-sft-dpo/sft_dpo/jacgen/README.md`](model-experiments/01-sft-dpo/sft_dpo/jacgen/README.md) | module-by-module pipeline reference (24 modules) |

**Attempt 2 — RL / GRPO**
| Doc | What |
|---|---|
| [`model-experiments/02-rl-grpo/RL_FINDINGS.md`](model-experiments/02-rl-grpo/RL_FINDINGS.md) | the full story — every era, every bug, every corrected number |
| [`model-experiments/02-rl-grpo/rl/README.md`](model-experiments/02-rl-grpo/rl/README.md) | pipeline reference — reward design, warm-start, ladder execution, gotchas |
| [`model-experiments/02-rl-grpo/docs/rl/00-overview.md`](model-experiments/02-rl-grpo/docs/rl/00-overview.md) / [`01-design.md`](model-experiments/02-rl-grpo/docs/rl/01-design.md) | design docs written before the ladder was built |
| [`model-experiments/02-rl-grpo/docs/rl/RL_WEEKEND_RESULTS.md`](model-experiments/02-rl-grpo/docs/rl/RL_WEEKEND_RESULTS.md) | original Era-1 write-up, verbatim |
| [`model-experiments/02-rl-grpo/docs/rl/references.md`](model-experiments/02-rl-grpo/docs/rl/references.md) | cited RL literature (Yue et al., ProRL, Spurious Rewards) |
| [`model-experiments/02-rl-grpo/resultspub/rl/README.md`](model-experiments/02-rl-grpo/resultspub/rl/README.md) | index of the published (corrected) graphs |
| [`model-experiments/02-rl-grpo/presentation/main.pdf`](model-experiments/02-rl-grpo/presentation/main.pdf) | slide deck ([source](model-experiments/02-rl-grpo/presentation/main.tex)) |

**Attempt 3 — CPT (rejected)**
| Doc | What |
|---|---|
| [`model-experiments/03-cpt-only/docs/cpt-2/design.md`](model-experiments/03-cpt-only/docs/cpt-2/design.md) | the locked 4-stage architecture (`base→+CPT→+CPT+SFT/DPO→+CPT+SFT/DPO+GRPO`) |
| [`model-experiments/03-cpt-only/docs/cpt-2/results.md`](model-experiments/03-cpt-only/docs/cpt-2/results.md) | CPT-v2 acceptance gates — 0/3 cleared |
| [`model-experiments/03-cpt-only/docs/cpt-2/analysis.md`](model-experiments/03-cpt-only/docs/cpt-2/analysis.md) | root cause — structural limit of next-token CPT on doc prose, not a bad run |
| [`model-experiments/03-cpt-only/docs/cpt-1/`](model-experiments/03-cpt-only/docs/cpt-1/) | CPT-v1 design + null result (MCQ concept-recognition byte-identical before/after) |

**Attempt 4 — SFT/DPO on CPT-v2 vs. fresh**
| Doc | What |
|---|---|
| [`model-experiments/04-cpt-sft/RESULTS.md`](model-experiments/04-cpt-sft/RESULTS.md) | consolidated bottom line — the one doc to share |
| [`model-experiments/04-cpt-sft/docs/reports/2026-07-cpt-vs-fresh-comparison.md`](model-experiments/04-cpt-sft/docs/reports/2026-07-cpt-vs-fresh-comparison.md) | full phase-by-phase deep dive — base/SFT/DPO comparison, the `mlx_lm.fuse` bug + fix, the q_proj SVD structural finding |
| [`model-experiments/04-cpt-sft/docs/README.md`](model-experiments/04-cpt-sft/docs/README.md) | docs index — spec, workflow, datagen |
| [`model-experiments/04-cpt-sft/lora_svd_qproj.py`](model-experiments/04-cpt-sft/lora_svd_qproj.py) | the q_proj LoRA singular-value probe behind the structural finding |

**Attempt 5 — GRPO, closing the loop**
| Doc | What |
|---|---|
| [`model-experiments/05-cpt-sft-grpo/docs/README.md`](model-experiments/05-cpt-sft-grpo/docs/README.md) | docs index — start here |
| [`model-experiments/05-cpt-sft-grpo/docs/strat.md`](model-experiments/05-cpt-sft-grpo/docs/strat.md) | the *why* — research questions, falsifiable hypotheses, carried scars from attempt 2 |
| [`model-experiments/05-cpt-sft-grpo/docs/spec.md`](model-experiments/05-cpt-sft-grpo/docs/spec.md) | umbrella design — training matrix, lineage-preservation mechanics, eval design |
| [`model-experiments/05-cpt-sft-grpo/docs/workflow.md`](model-experiments/05-cpt-sft-grpo/docs/workflow.md) | phased runbook |
| [`model-experiments/05-cpt-sft-grpo/docs/dataset/`](model-experiments/05-cpt-sft-grpo/docs/dataset/) | multi-source corpus mining + Type-B AST-equivalence grading design |

---

## Glossary

| Term | Meaning |
|---|---|
| **SFT** | supervised finetuning — train on input→output pairs |
| **DPO** | direct preference optimization — train on (chosen vs rejected) pairs to push toward a preferred style |
| **CPT** | continual pretraining — further next-token training on raw domain text (docs/papers/blogs) before SFT, aimed at domain *semantics* rather than task-specific behavior; rejected in attempt 3 (see [above](#attempt-3--cpt-rejected)) |
| **GRPO** | group-relative policy optimization — the RL method used in attempt 2; sample a group of rollouts per prompt, advantage = `(reward − group mean) / group σ` |
| **LoRA** | low-rank adapter finetuning — cheap, small, fusable into the base weights; the only way a 30B model trains at all on 48GB |
| **MLX** | Apple's array/ML framework for Apple Silicon; `mlx-lm` / `mlx-lm-lora` run train/infer locally |
| **py2jac** | `jac` subcommand that mechanically transpiles Python → Jac (Python-shaped output) |
| **transpile-similarity** | ROUGE-L of model output vs `py2jac` of the same Python — high = Python-shaped, low = rewritten/idiomatic |
| **idiom headroom** | how much an idiomatic answer can diverge from a mechanical transpile (large for graph/OSP tasks, ~zero for pure functions) |
| **cross-compiled test-pass** | the primary metric throughout: converted Jac compiles, runs, and its output matches the recorded behavioral cases |
| **holdout** | unseen, decontaminated eval set the model never trained on |
| **pass@1 / greedy** | one deterministic (temperature-0) answer — the headline "what does the model default to" number |
| **pass@k / oracle** | sample k times, pass if *any* sample is correct — the reachable ceiling with an oracle picking the best sample |
| **best-of-k (deploy)** | sample k, return the first candidate the **Jac compiler** accepts — no gold answer peeked at; the number you'd actually ship |
| **σ=0 trap** | when every rollout in a GRPO group scores identically, the advantage divides by zero variance → zero gradient regardless of learning rate; RL can't bootstrap a skill the base model has none of |
| **task interference** | adding more/harder/more-varied training data regresses a task already learned — a small LoRA adapter running out of capacity to hold multiple skills at once |
| **OSP** | object-spatial programming — Jac's node/edge/walker/visit model, with no Python equivalent |
