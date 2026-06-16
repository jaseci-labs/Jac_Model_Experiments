# Jac Coding Agent — Project Context

**Confidential — Jaseci Labs.** Persistent background for AI-assisted sessions.
Operational detail lives in [`sft_dpo/process.md`](sft_dpo/process.md) and [`docs/`](docs/); this
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

Finetune target candidates (model-testing phase): **Gemma 4 26B A4B** (primary)
and **Qwen3-Coder-30B-A3B** (fallback). Both small-MoE (~3B active), Q4-fit the
48 GB M5 Pro for local MLX LoRA. DeepSeek-V3-Lite was dropped; Kimi K2.x is too
large for the hardware. LoRA via MLX (Apple Silicon) / Unsloth (cloud A100).

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
in [`docs/datagenstrat/strat.md`](docs/datagenstrat/strat.md). Verification order:
compiler gate → cross-compiled tests → idiom judge → sampled manual review.

## Current state

The model-testing phase is built and runnable. The pipeline (mine + generate +
dedup + decontaminate + split + train/eval harness) is all in Jac under
[`sft_dpo/jacgen/`](sft_dpo/jacgen/). Conversion dataset (gitignored under
`dataset/`): 1616 SFT (116 idiomatic core + 1500 py2jac volume), 60 DPO, 150
decontaminated eval holdout. See `sft_dpo/process.md` to run the probe,
[`sft_dpo/docs/modeltesting/HANDOFF.md`](sft_dpo/docs/modeltesting/HANDOFF.md) for the full
handoff, and [`sft_dpo/docs/modeltesting/`](sft_dpo/docs/modeltesting/) for strategy/evaluation.

## Fixed constraints

- Target language: **Jac** (agent is Jac-specific).
- Compiler/behavioral validation: hard requirement for all training code.
- Finetuning: LoRA (MLX local / Unsloth cloud); Instruct variant.
- Data regime: 100% synthetic.
- Quality target: compiler-correct, idiomatic Jac — not approximate.

## Key references

| Resource | Location |
|---|---|
| Run the probe | `sft_dpo/process.md` |
| Data generation strategy (12 recipes) | `docs/datagenstrat/strat.md` |
| Whole-stack strategy | `docs/wholestack/strat.md` |
| Model testing | `sft_dpo/docs/modeltesting/strategy.md`, `evaluation.md`, `mini_probe.md` |
| Pipeline code | `sft_dpo/jacgen/` (+ its `README.md`) |
| Research papers | `papers/` (MultiPL-T, WizardCoder, Magicoder, SelfCodeAlign, DeepSeek-Coder, CodeDPO, Magpie) |
