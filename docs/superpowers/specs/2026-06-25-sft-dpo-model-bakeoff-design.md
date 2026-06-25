# SFT+DPO Model Bake-off: Qwen3-Coder vs Same-Size MoE Coders

**Date:** 2026-06-25
**Status:** design — awaiting review

## Objective

Settle, once and for all, which base model learns Jac best from our synthetic
SFT+DPO data — comparing the incumbent **Qwen3-Coder-30B-A3B** against same-size
coder models that fit a 48GB Apple-Silicon Mac under MLX. One controlled
variable: the base model. Everything else (dataset, LoRA config, eval suite,
hardware) held constant.

This re-runs the exact treatment Qwen3-Coder already got (`run_probe.sh` +
`run_dpo.sh`) on each candidate, then ranks by the same metrics.

## Why this, why now

Qwen3-Coder-30B-A3B is the current pick (94% behavioral, 100% idiomatic after
DPO). Before committing the full 300k–500k generation budget to it, prove no
same-size model does better. Cost of testing ≈ 1 week of Mac time; cost of
choosing wrong ≈ the whole budget.

## Hardware constraint (hard gate: NO OOM)

- Mac M5 Pro, 48GB unified RAM, MLX only.
- Train Q4 + LoRA; eval Q8 (fuse adapter into Q8). Proven path from Qwen3-Coder.
- **Kimi K2 dropped:** 1T-param MoE (32B active), ~500GB+ at Q4 — cannot load.
  No small Kimi variant exists. Out of scope for this hardware.
- Any model whose Q4 train state or Q8 eval weights would exceed RAM is
  excluded up front. Dense ≥27B already OOM'd on warm-start → dense capped at 14B.

## Candidate pool (locked)

All MoE except one dense control. Every entry fits Q4-train / Q8-eval with
headroom on 48GB.

| # | Model | HF id (verify exact tag) | total/active | type | short-name | role |
|---|---|---|---|---|---|---|
| baseline | Qwen3-Coder-30B-A3B | `Qwen/Qwen3-Coder-30B-A3B` | 30B / 3B | MoE | `qwen` | incumbent — already measured (94% / 100%), not re-run |
| 1 | Qwen3-30B-A3B-Instruct | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 30B / 3B | MoE | `qwen3i` | sibling — isolates value of coder-specific pretrain |
| 2 | gpt-oss-20b | `openai/gpt-oss-20b` | 21B / 3.6B | MoE | `gptoss` | different lineage, Apache-2.0 |
| 3 | DeepSeek-Coder-V2-Lite-Instruct | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | 16B / 2.4B | MoE | `dscoder` | coder MoE |
| 4 | Ling-Coder-Lite | `inclusionAI/Ling-Coder-lite` | 16.8B / 2.75B | MoE | `ling` | coder MoE, native SFT→DPO recipe |
| 5 | Qwen2.5-Coder-14B-Instruct | `Qwen/Qwen2.5-Coder-14B-Instruct` | 14B dense | dense | `qwen25c` | dense-vs-MoE LoRA-learnability control |

**Excluded for OOM:** GLM-4.5-Air (106B-A12B), Llama-4-Scout (109B-A17B), any
dense ≥27B. **Gemma3-26B-A4B** already tested — reuse its numbers, do not re-run.

## Protocol

Per candidate, sequential (each saturates RAM — no parallel runs):

1. `./run_probe.sh <hf-id> <short-name>` — quantize Q4+Q8, base eval, LoRA SFT
   (resumable), learning curve, fused finetuned eval. Output: behavioral pass-%.
2. `./run_dpo.sh <short-name>` — LoRA-DPO on R3 idiom pairs, behavior re-eval +
   idiom eval. Output: avg transpile-similarity (lower = more idiomatic) +
   behavior-holds check.
3. Record into the comparison matrix.

**Config control:** identical `sft_dpo/configs/lora.yaml` for all (rank 16,
scale 2.0, 600 iters, lr 2e-5, batch 2, max-seq 2048). `num_layers` adjusted
**only** where a model's depth makes 16 invalid. DPO identical too (8 layers,
beta 0.1, lr 1e-6, 200 iters, grad-checkpoint, max-seq 384, reference=frozen
base — single weight set in RAM).

**Data control:** same `dataset/mlx` (SFT) and `dataset/mlx_dpo` (DPO) splits
for every model. No regeneration.

**Gate:** none. Full SFT+DPO on every model regardless of SFT result (complete
matrix — user wants the definitive comparison, not an early-drop screen).

## Memory budget per candidate (the no-OOM proof)

| Model | Q4 train weights | + LoRA/opt/act | Q8 eval weights | verdict |
|---|---|---|---|---|
| qwen3i (30B-A3B) | ~16GB | ~24GB total | ~32GB | proven (qwen ran) |
| gptoss (21B) | ~12GB | ~20GB | ~21GB | comfortable |
| dscoder (16B) | ~8GB | ~16GB | ~16GB | comfortable |
| ling (16.8B) | ~8GB | ~16GB | ~17GB | comfortable |
| qwen25c (14B dense) | ~7GB | ~15GB | ~14GB | comfortable |

All under the 48GB ceiling with the same margins the proven qwen run used.

## Metrics & decision

Primary, objective (no judge needed):
- **SFT behavioral pass-%** on the decontaminated holdout (`eval_probe.jac`) —
  base vs finetuned. This is the headline number.
- **DPO idiom gain** (`idiom_eval.jac`): avg transpile-similarity must DROP
  (more idiomatic) while behavioral pass-% HOLDS.

Secondary: training stability (val-loss curve), inference tok/s, license.

**Winner:** highest finetuned behavioral pass-% with idiom-gain holding behavior,
compared head-to-head against Qwen3-Coder's 94% / 100%. A model only displaces
Qwen3-Coder if it beats it on behavioral-% by more than run-to-run noise AND
matches/beats idiom. Ties → keep Qwen3-Coder (incumbent, Apache-2.0, proven).

## Risks

- **gpt-oss-20b MXFP4-native quant** — MLX convert/LoRA/DPO path is the one
  unknown. Dry-run conversion FIRST; if it blocks, drop gptoss and note it.
- **HF id / tag drift** — verify each exact repo id + revision on Hugging Face
  before download (table tags are best-known, not guaranteed).
- **Per-arch `num_layers`** — MLX LoRA may reject 16 layers on shallower models;
  fall back to that model's depth. Log any deviation from the shared config.
- **Disk** — ~50-60GB per model (Q4+Q8+fused). 5 models ≈ 300GB. Clean up
  fused models between runs if disk-bound.

## Deliverables

- Per-model `results/<short-name>/` (base.txt, finetuned.txt, learning_curve.png)
  and `results/<short-name>/dpo/` (finetuned.txt, idiom.txt).
- One comparison report: matrix of all 6 (5 new + incumbent) across behavioral-%,
  idiom avg-sim, train stability, tok/s, license — with the final recommendation.
- `make_comparison.py` extended if needed to tabulate all candidates.

## Out of scope

- No RL (GRPO/STaR — already shown not to beat supervised).
- No dataset regeneration or new task categories.
- No cloud/CUDA port (would be a separate effort if Kimi-class models ever wanted).
