# Phase 1 — Supervised Finetuning (SFT + DPO): the first finetuning experience

*The complete record of the first finetuning phase: teaching small open models to
write idiomatic, compiler-correct Jac from 100% synthetic data, on a single 48 GB
Apple-Silicon Mac. This worked. Sourced from `sft_dpo/docs/modeltesting/`,
`sft_dpo/process.md`, `docs/initmodelchoice/`, `results/RESULTS.md`.*

---

## The problem

Jac is a data-spatial language (walkers, nodes, edges, abilities) built on Python.
Models trained on Python/JS have weak priors on correct Jac — they produce
Python-shaped code that looks plausible but is syntactically or semantically wrong.
No real Jac corpus exists, so all training data is synthetic.

**Goal:** a coding model that generates idiomatic, compiler-correct Jac, not
"Jac-looking" code. Quality bar = compiles + runs + idiomatic.

## Data strategy (100% synthetic)

Three anchors substitute for a real-data distribution:
1. **Jac grammar** = the distribution target (every construct must appear).
2. **Jac compiler + cross-compiled tests** = the unlimited oracle. Rejection
   sampling is free; behavioral test pass is the real gate, not just compilation.
3. **Python** = the proxy distribution (translate validated Python → idiomatic Jac,
   MultiPL-T methodology).

The pipeline (all written in Jac, under `sft_dpo/jacgen/`) mines + generates +
gates + dedups + decontaminates + splits. Every example is gated by `jac run`
(exit 0) AND a behavioral check (output matches across multiple test cases) — never
by `jac check`, which over-rejects untyped-but-runnable Jac.

### Data tiers produced
- **Idiomatic core** (`sft.jsonl`, ~140 incl. 24 graph) — true graph-spatial Jac
  (walkers, nodes, edges, abilities). Authored agentically (Claude + jac-mcp). The
  quality bar.
- **Volume tier** (`sft_auto.jsonl`, 1500) — transpiled corpus functions
  (`jac py2jac`), compiler + behaviorally validated. Python-shaped Jac; breadth.
- **DPO** (`dpo.jsonl`, 60) — idiomatic (chosen) vs Python-shaped (rejected); teaches
  de-Python-ification.

Total SFT ≈ 1640. Manifest `sft_train.jsonl` ≈ 560 (idiomatic + stride-sampled
transpile at 1:3). Holdout: 150 conversion (decontaminated, 14-gram) + 10 graph.

## Models

- **Qwen** — `Qwen/Qwen3-Coder-30B-A3B-Instruct` (30B MoE, ~3B active)
- **Gemma** — `google/gemma-4-26b-a4b-it` (26B MoE, ~4B active)

Both small-MoE so ~3-4B active params fit a 48 GB M-series for local MLX LoRA.

## Training

`sft_dpo/run_probe.sh` (SFT) → `sft_dpo/run_dpo.sh` (DPO). Config
`sft_dpo/configs/lora.yaml`: LoRA r16, 600 iters, lr 2e-5, batch 2.

- **SFT:** quantize base → q4/q8, LoRA-finetune on the conversion data, fuse.
- **DPO:** the SFT model is 94% behavioral but ~99% of its correct outputs are
  Python-shaped (transpile-similarity 0.968). DPO on the de-Python-ification pairs
  (chosen=idiomatic, rejected=transpile) pushes it toward idiomatic Jac. LoRA-DPO
  with the reference left as the frozen base, so only one 30B weight set sits in
  RAM (a separate reference copy would OOM 48 GB).

## Results (measured, base vs finetuned, held-out + decontaminated)

| metric | Qwen3-Coder-30B-A3B | Gemma-4-26B-A4B |
|---|---|---|
| function holdout — base | 0% | 0% |
| function holdout — finetuned | **94%** | 93% |
| graph holdout — SFT correct | **46%** | 15% |
| graph holdout — DPO correct | **61%** | 15% |
| graph — of-correct idiomatic (SFT→DPO) | 83% → **100%** | 100%* |
| graph — transpile-similarity (SFT→DPO) | 0.457 → **0.338** | 0.667 (flat) |
| graph — constructs/output (SFT→DPO) | 4.5 → **6.75** | 0.0 |

**Read:** the finetune took function-conversion from **0% → 94%**. DPO raised
graph-task correctness 46% → 61% AND made the correct outputs fully idiomatic
(transpile-similarity dropped, construct count rose). Qwen clearly beat Gemma on
the graph-spatial (the hard, distinctly-Jac) tier. This is the success the RL phase
was meant to build on.

## Gotchas (recorded so they aren't re-hit)

- **Live-eval OOM:** running a holdout eval per checkpoint *during* training loads a
  second full model in-process → OOM/deadlock on 48 GB for a 30B. Default off; watch
  val loss instead (free, same model). Base-vs-finetuned is measured separately.
- **Seed truncation:** `seed_conversion.jac` TRUNCATES `sft.jsonl` back to the 32
  seeds — never run it after the idiomatic_batch* appends. See HANDOFF.
- **Parse vs type-check:** `eval_probe.jac` / `idiom_eval.jac` import mlx_lm; the
  type-checker crashes on mlx types, so they're parse-checked (`jac check -p`) and
  trusted via `jac run`.
- **The gate is behavioral, not `jac check`:** `jac check` over-rejects
  untyped-but-runnable Jac; correctness = `jac run` exit 0 + output match.

## Artifacts

- Pipeline: `sft_dpo/jacgen/` (~24 Jac modules). Runbook: `sft_dpo/process.md`.
  Handoff: `sft_dpo/docs/modeltesting/HANDOFF.md`.
- Adapters: `adapters/{qwen,gemma}-{probe,dpo}`. Models:
  `models/{qwen,gemma}-jac-{fused,dpo-fused}-q8`. Charts: `sft_dpo/resultsft/`.

**Bottom line:** the SFT+DPO phase succeeded — a model that had never seen Jac
reached 94% function conversion and 61% idiomatic graph-task correctness, all from
synthetic data, locally. The RL phase ([`00-rl-phase.md`](00-rl-phase.md)) tried to
push further and is a more mixed story.
