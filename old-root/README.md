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
| **[`01-sft-dpo/`](01-sft-dpo/)** | Supervised finetuning + DPO | done | stock model **0%** runnable Jac → **94%** after one LoRA pass |
| **[`02-rl-grpo/`](02-rl-grpo/)** | RL (GRPO) on top of attempt 1's model | done | best-of-k + compiler-as-verifier ships **~94%**; GRPO ≈ SFT, no extra lift |
| **[`03-new/`](03-new/)** | TBD | just started | seeded with [`03-new/rui.md`](03-new/rui.md) |

Shared across all attempts, at repo root: `models/` (base + merged checkpoints,
gitignored), `docs/` (repo-wide strategy + the adapter-hyperparameter registry),
`studio/` (the Jac Model Studio app, which reads results from every attempt),
`this_is_jac/` (the real Jac codebase RL mines tasks from), `papers/` (reference papers).

---

## Table of contents

- [What Jac is (and why models fail at it)](#what-jac-is-and-why-models-fail-at-it)
- [Attempt 1 — SFT + DPO](#attempt-1--sft--dpo)
- [Attempt 2 — RL / GRPO](#attempt-2--rl--grpo)
- [Attempt 3 — next](#attempt-3--next)
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

**[`01-sft-dpo/`](01-sft-dpo/)** — the first attempt: prove that supervised finetuning
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
Jac itself — see [`01-sft-dpo/sft_dpo/jacgen/`](01-sft-dpo/sft_dpo/jacgen/) (24
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
  [`01-sft-dpo/docs/initmodelchoice/2026-06-26-sft-dpo-bakeoff-results.md`](01-sft-dpo/docs/initmodelchoice/2026-06-26-sft-dpo-bakeoff-results.md).

Full results, all 16 training graphs, side-by-side model comparison →
**[`01-sft-dpo/resultspub/initmodelchoice/RESULTS.md`](01-sft-dpo/resultspub/initmodelchoice/RESULTS.md)**.

### Run it

```bash
./setup_env.sh && source .venv/bin/activate
./01-sft-dpo/sft_dpo/check.sh                                                # type + behavioral gate, non-destructive
./01-sft-dpo/sft_dpo/run_probe.sh Qwen/Qwen3-Coder-30B-A3B-Instruct qwen      # quantize → base eval → train → fuse → finetuned eval
./01-sft-dpo/sft_dpo/run_dpo.sh qwen                                          # DPO stage on top of the SFT adapter
```

Full docs → operator runbook
[`01-sft-dpo/sft_dpo/process.md`](01-sft-dpo/sft_dpo/process.md), architecture handoff
[`01-sft-dpo/docs/sft_dpo/modeltesting/HANDOFF.md`](01-sft-dpo/docs/sft_dpo/modeltesting/HANDOFF.md),
pipeline reference [`01-sft-dpo/sft_dpo/jacgen/README.md`](01-sft-dpo/sft_dpo/jacgen/README.md).

---

## Attempt 2 — RL / GRPO

**[`02-rl-grpo/`](02-rl-grpo/)** — starting from attempt 1's SFT+DPO'd model
(`jac-qwen3coder`), the second attempt asked whether **RL (GRPO)** could push
correctness further, using the Jac compiler itself as a free, verifiable reward (no
learned reward model). Full story with every number and every bug:
**[`02-rl-grpo/RL_FINDINGS.md`](02-rl-grpo/RL_FINDINGS.md)**.

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

Shipped: **[`02-rl-grpo/rl/generate.py`](02-rl-grpo/rl/generate.py)** — the best-of-k
generator; the live Studio RL section (11%→94% journey, ladder, k-scaling, a GENERATE
JAC panel), backed by
[`02-rl-grpo/resultspub/rl/corrected_summary.json`](02-rl-grpo/resultspub/rl/corrected_summary.json).
Graphs → [`02-rl-grpo/resultspub/rl/`](02-rl-grpo/resultspub/rl/).

### Run it

```bash
jac run 02-rl-grpo/rl/build_tasks.jac           # this_is_jac/ drivers -> tasks + templates
jac run 02-rl-grpo/rl/build_rl_splits.jac       # fixed holdout + trainpool
jac run 02-rl-grpo/rl/run_ladder.jac            # DRY: prints the plan, runs nothing heavy
JAC_LADDER_GO=1 jac run 02-rl-grpo/rl/run_ladder.jac   # execute the ladder (hours per cell)
jac run 02-rl-grpo/rl/show_ladder.jac           # pivot results into a curve table
```

Full pipeline reference (reward design, warm-start, the recommended
compute-smart execution order, gotchas) → **[`02-rl-grpo/rl/README.md`](02-rl-grpo/rl/README.md)**.

---

## Attempt 3 — next

**[`03-new/`](03-new/)** — not started yet. Seeded with
[`03-new/rui.md`](03-new/rui.md).

---

## Repository layout

| Path | What |
|---|---|
| `01-sft-dpo/` | attempt 1 — code, dataset, adapters, results, docs (see [above](#attempt-1--sft--dpo)) |
| `02-rl-grpo/` | attempt 2 — code, dataset, adapters, results, docs, the RL slide deck (see [above](#attempt-2--rl--grpo)) |
| `03-new/` | attempt 3 — just `rui.md` so far |
| `models/` *(gitignored)* | base + merged/fused checkpoints, shared across attempts — attempt 2 finetunes attempt 1's output |
| `results/` | studio scratch space only (`_builder`, `_evals`) — per-attempt run outputs live inside `01-sft-dpo/results/` and `02-rl-grpo/results/` |
| `docs/` | repo-wide: `training_configs/` (hyperparameter registry for every adapter, incl. deleted ones — see `docs/ARTIFACT_LOG.md`), `wholestack/` (end-to-end strategy spanning both attempts) |
| `studio/` | **Jac Model Studio** — the app that visualizes/drives all of this (dataset browser, GENERATE panel, RL section, builder jobs) |
| `this_is_jac/` | the real open-source Jac codebase attempt 2 mines RL tasks from |
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
| [`01-sft-dpo/sft_dpo/process.md`](01-sft-dpo/sft_dpo/process.md) | operator runbook — setup → check → run, pause/resume, timings |
| [`01-sft-dpo/docs/sft_dpo/modeltesting/HANDOFF.md`](01-sft-dpo/docs/sft_dpo/modeltesting/HANDOFF.md) | single source of truth — architecture, every module, every gotcha |
| [`01-sft-dpo/docs/initmodelchoice/2026-06-26-sft-dpo-bakeoff-results.md`](01-sft-dpo/docs/initmodelchoice/2026-06-26-sft-dpo-bakeoff-results.md) | 6-model base bake-off, the keep-Qwen3-Coder verdict |
| [`01-sft-dpo/docs/initmodelchoice/strat.md`](01-sft-dpo/docs/initmodelchoice/strat.md) | the 12 data-generation recipes (R1–R12) |
| [`01-sft-dpo/resultspub/initmodelchoice/RESULTS.md`](01-sft-dpo/resultspub/initmodelchoice/RESULTS.md) | full measured results + all 16 training graphs |
| [`01-sft-dpo/sft_dpo/jacgen/README.md`](01-sft-dpo/sft_dpo/jacgen/README.md) | module-by-module pipeline reference (24 modules) |

**Attempt 2 — RL / GRPO**
| Doc | What |
|---|---|
| [`02-rl-grpo/RL_FINDINGS.md`](02-rl-grpo/RL_FINDINGS.md) | the full story — every era, every bug, every corrected number |
| [`02-rl-grpo/rl/README.md`](02-rl-grpo/rl/README.md) | pipeline reference — reward design, warm-start, ladder execution, gotchas |
| [`02-rl-grpo/docs/rl/00-overview.md`](02-rl-grpo/docs/rl/00-overview.md) / [`01-design.md`](02-rl-grpo/docs/rl/01-design.md) | design docs written before the ladder was built |
| [`02-rl-grpo/docs/rl/RL_WEEKEND_RESULTS.md`](02-rl-grpo/docs/rl/RL_WEEKEND_RESULTS.md) | original Era-1 write-up, verbatim |
| [`02-rl-grpo/docs/rl/references.md`](02-rl-grpo/docs/rl/references.md) | cited RL literature (Yue et al., ProRL, Spurious Rewards) |
| [`02-rl-grpo/resultspub/rl/README.md`](02-rl-grpo/resultspub/rl/README.md) | index of the published (corrected) graphs |
| [`02-rl-grpo/presentation/main.pdf`](02-rl-grpo/presentation/main.pdf) | slide deck ([source](02-rl-grpo/presentation/main.tex)) |

---

## Glossary

| Term | Meaning |
|---|---|
| **SFT** | supervised finetuning — train on input→output pairs |
| **DPO** | direct preference optimization — train on (chosen vs rejected) pairs to push toward a preferred style |
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
