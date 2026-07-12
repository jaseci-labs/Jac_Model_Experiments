# Jac Coding Agent — Project Context

**Confidential — Jaseci Labs.** Persistent background for AI-assisted sessions.
Operational detail lives in [`01-sft-dpo/sft_dpo/process.md`](01-sft-dpo/sft_dpo/process.md) and [`docs/`](docs/); this
is the durable framing only.

---

## What Jac is

Jac is a programming language built on Python (Jaseci Labs), centered on a
**data-spatial / object-spatial model**: computation expressed with **nodes,
edges, walkers, and abilities** rather than plain functions/classes. It compiles
to Python and interops with the ecosystem, but its idioms are distinct enough
that models trained on Python/JS/C have very weak priors on correct Jac.

Distinct constructs: `walker` (traversal agents), `node`/`edge` (graph
primitives), `can ... with <Node> entry` (event abilities) vs `def` (methods),
`obj` (preferred over `class`), `with entry`, `spawn` / `++>` / `visit [-->]` /
`disengage`, `has` typed fields, archetype inheritance. A model not finetuned on
Jac produces Python-shaped code that looks plausible but is syntactically or
semantically wrong — the core justification for this project.

## Goal

A coding agent for Jac (what Claude Code is for Python): generate, debug,
explain, and convert to **idiomatic, compiler-correct** Jac — not
"Jac-looking" code. Deployed via Jac MCP in coding assistants. Quality bar =
compiles + runs + idiomatic.

## Base model

**Qwen3-Coder-30B-A3B-Instruct** — selected empirically by the 7-model SFT+DPO
bake-off (see `01-sft-dpo/docs/initmodelchoice/2026-06-26-sft-dpo-bakeoff-results.md`; no
candidate beat it above noise). Small-MoE (~3B active), Q4-fits the 48 GB M5
Pro for local MLX LoRA.

## Data strategy (100% synthetic)

No real Jac corpus exists. Three anchors substitute for real-data distribution:
1. **Jac grammar** = the distribution anchor (every construct must appear).
2. **Jac compiler + cross-compiled tests** = the unlimited oracle (rejection
   sampling is free; behavioral test pass is the real gate, not just compilation).
3. **Python** = the proxy distribution (translate validated Python → idiomatic Jac;
   MultiPL-T methodology).

Generation recipes (R1–R12: coverage matrix, Python↔Jac parallel corpus,
adversarial DPO negatives, bug-synthesis, persona/evol-instruct, self-distill,
multi-turn, reasoning traces, doc-grounded, OSS-Instruct, Magpie) are documented
in [`01-sft-dpo/docs/initmodelchoice/strat.md`](01-sft-dpo/docs/initmodelchoice/strat.md). Verification order:
compiler gate → cross-compiled tests → idiom judge → sampled manual review.

## Current state

SFT+DPO phase done: 1647 SFT / 147 DPO examples (git-tracked under `01-sft-dpo/dataset/`),
fn conversion 0%→94%, graph conversion 46%→61% (see `README.md` for the full
tables). RL/GRPO phase done and written up in `02-rl-grpo/RL_FINDINGS.md` — SFT moves
greedy accuracy (39%→61% at rung-20), GRPO adds nothing; deployable recipe =
SFT + best-of-k with the Jac compiler as verifier (`02-rl-grpo/rl/generate.py`, ~78–82%).
The pipeline (mine + generate + dedup + decontaminate + split + train/eval
harness) is all in Jac under [`01-sft-dpo/sft_dpo/jacgen/`](01-sft-dpo/sft_dpo/jacgen/). See
`01-sft-dpo/sft_dpo/process.md` to run the probe,
[`01-sft-dpo/docs/sft_dpo/modeltesting/HANDOFF.md`](01-sft-dpo/docs/sft_dpo/modeltesting/HANDOFF.md) for the full
handoff, and [`01-sft-dpo/docs/sft_dpo/modeltesting/`](01-sft-dpo/docs/sft_dpo/modeltesting/) for strategy/evaluation.

## Fixed constraints

- Target language: **Jac** (agent is Jac-specific).
- Compiler/behavioral validation: hard requirement for all training code.
- Finetuning: LoRA (MLX local / Unsloth cloud); Instruct variant.
- Data regime: 100% synthetic.
- Quality target: compiler-correct, idiomatic Jac — not approximate.

## Key references

| Resource | Location |
|---|---|
| Run the probe | `01-sft-dpo/sft_dpo/process.md` |
| Data generation strategy (12 recipes) | `01-sft-dpo/docs/initmodelchoice/strat.md` |
| Whole-stack strategy | `docs/wholestack/strat.md` |
| Model testing | `01-sft-dpo/docs/sft_dpo/modeltesting/strategy.md`, `evaluation.md`, `mini_probe.md` |
| RL findings | `02-rl-grpo/RL_FINDINGS.md` (authoritative), artifacts in `docs/ARTIFACT_LOG.md` |
| Pipeline code | `01-sft-dpo/sft_dpo/jacgen/` (+ its `README.md`) |
| Research papers | `papers/` (MultiPL-T, WizardCoder, Magicoder, SelfCodeAlign, DeepSeek-Coder, CodeDPO, Magpie) |
