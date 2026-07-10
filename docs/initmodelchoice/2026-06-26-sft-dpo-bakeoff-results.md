# SFT+DPO Model Bake-off — Results & Recommendation

**Date:** 2026-06-26
**Spec:** `docs/archive/superpowers/specs/2026-06-25-sft-dpo-model-bakeoff-design.md`
**Plan:** `docs/archive/superpowers/plans/2026-06-26-sft-dpo-model-bakeoff.md`
**Status:** complete

## TL;DR

**Keep the incumbent, Qwen3-Coder-30B-A3B.** No candidate beats it on behavioral
pass-% by more than run-to-run noise, and on the graph holdout (the real idiom
tell) the incumbent's DPO behavioral score (61%) is the best of any DPO-capable
model. The strongest challenger is **Qwen3-30B-A3B-Instruct (`qwen3i`)** — a
statistical tie on behavior that writes *more idiomatic* graph Jac after DPO
(0.223 vs 0.338) — but a tie keeps the incumbent (Apache-2.0, coder-pretrained,
already proven on the full pipeline).

## Method

Each candidate got the identical treatment Qwen3-Coder already had: `run_probe.sh`
(quantize Q4+Q8, base eval, LoRA SFT 600 iters, learning curve, fused eval) +
`run_dpo.sh` (LoRA-DPO 200 iters, behavior re-eval, idiom eval), plus — added per
request — a **graph holdout** tier (node/edge/walker idiom, 13 tasks) at base/SFT/DPO.
One controlled variable: the base model. Identical `lora.yaml`, dataset, hardware
(M5 Pro 48GB, MLX). Behavioral = cross-compiled test-pass on the holdout; idiom =
avg transpile-similarity (**lower = more idiomatic**, 0.26 ≈ ideal).

## Results

### Function holdout (150 tasks)

| model | base | SFT | DPO | SFT sim | DPO sim | idiom gain | tok/s | license |
|---|---|---|---|---|---|---|---|---|
| **Qwen3-Coder-30B-A3B** (incumbent) | 0% | 94% | 93% | 0.968 | 0.959 | +0.008 | 63 | Apache-2.0 |
| gpt-oss-20b | 0% | 92% | NA* | 0.973 | NA* | — | 80 | Apache-2.0 |
| Qwen3-30B-A3B-Instruct | 0% | **95%** | 94% | 0.964 | 0.958 | +0.006 | 68 | Apache-2.0 |
| DeepSeek-Coder-V2-Lite | 2% | 94% | 94% | 0.968 | 0.967 | +0.001 | 58 | DeepSeek |
| Qwen2.5-Coder-14B | 0% | 94% | 93% | 0.964 | 0.964 | +0.000 | 16 | Apache-2.0 |
| Ling-Coder-lite | — | DROPPED† | — | — | — | — | — | MIT |

### Graph holdout (13 tasks — node/edge/walker idiom)

| model | base | SFT | DPO | SFT sim | DPO sim | idiom gain |
|---|---|---|---|---|---|---|
| **Qwen3-Coder-30B-A3B** (incumbent) | 0% | 46% | **61%** | 0.457 | 0.338 | +0.119 |
| gpt-oss-20b | 0% | 61% | NA* | **0.210** | NA* | — |
| Qwen3-30B-A3B-Instruct | 0% | 53% | 53% | 0.558 | **0.223** | +0.335 |
| DeepSeek-Coder-V2-Lite | 0% | 15% | 23% | 0.707 | 0.546 | +0.160 |
| Qwen2.5-Coder-14B | 0% | 38% | 23% | 0.444 | 0.232 | +0.212 |
| Ling-Coder-lite | — | DROPPED† | — | — | — | — |

\* gpt-oss DPO not possible — see deviation below. † ling dropped — see below.

Comparison graphs: `results/comparison/` — `learning_curve_compare.png`,
`train_loss_compare.png`, `val_loss_compare.png`, `accuracy_compare.png`,
`idiom_compare.png`, `graph_accuracy_compare.png`, `graph_idiom_compare.png`.

## Per-candidate read

**Qwen3-30B-A3B-Instruct (`qwen3i`) — strongest challenger, but a tie.**
Best function SFT of the field (95%, vs incumbent 94% — a 1-task-of-150 edge, inside
noise). Function DPO 94 ≈ incumbent 93. On graph it ties on behavior (53% SFT/DPO; on a
13-task set, 46/53/61% are all 6–8/13, i.e. ±1 task = noise) and *wins idiom* — DPO drives
graph similarity from 0.558→**0.223**, the biggest idiom gain of any model (+0.335) and
more idiomatic than the incumbent's 0.338. Verdict: matches behavior, beats graph idiom —
but does **not** beat behavior beyond noise, so it does not clear the displacement bar.

**gpt-oss-20b (`gptoss`) — excellent learner, undeployable on this pipeline.**
SFT function 92% and graph **61%** with graph idiom **0.21** (the most idiomatic graph Jac
in the field). But its MXFP4 weights break the standard MLX path: Q8 quantization produces
0% (garbage generation) and `mlx_lm.fuse` corrupts the model at any precision. Scored
SFT-only via an unfused Q4+adapter eval (documented, conservative). **DPO impossible** —
`run_dpo` must fuse SFT→base first, and the fuse breaks it. Promising but not shippable
without new MLX MXFP4 support. (`results/gptoss/DEVIATION.md`.)

**DeepSeek-Coder-V2-Lite (`dscoder`) — strong functions, weak graphs.**
Function 94/94 (ties the field) but graph collapses to 15%→23% with the least-idiomatic
output (0.707/0.546). Learns to transpile, not to think graph-spatially. Not competitive on
the dimension that matters most for Jac.

**Qwen2.5-Coder-14B (`qwen25c`) — dense control, confirms MoE advantage.**
Function 94/93 on par, but graph **regressed** under DPO (38%→23%: DPO traded behavior for
idiom, 0.444→0.232). And as a 14B *dense* model it runs at **16 tok/s — ~4× slower** than the
MoE candidates (58–80). No reason to take a slower model that's worse on graph.

**Ling-Coder-lite (`ling`) — DROPPED.**
BailingMoE architecture. Standard convert failed (`rotary_emb.inv_freq` strict-load reject);
salvaged with a relaxed convert, but the model then loads into a generic class and generates
**runaway output (512 tokens, no EOS) — 0% across all six checkpoints**. Not a learning
failure, an integration failure: BailingMoE isn't supported by the installed `mlx_lm`.
(`results/ling/DROPPED.md`.)

## Deviations & incidents (full disclosure)

- **gpt-oss MXFP4** — Q8 quant + `mlx_lm.fuse` both broken; scored SFT-only on Q4+adapter,
  no DPO. The spec pre-authorized this drop path.
- **qwen3i DPO save OOM** — DPO trained cleanly (loss 0.62→0.03) but the 30B end-of-run
  full-model save OOM'd on 48GB. The LoRA adapter persisted; recovered by evaluating the
  adapter unfused on the Q8 SFT base. Numbers are valid.
- **ling** — dropped (above).
- **num_layers** — no model required the fallback; all kept the shared 16.

## Decision

Rule (from spec): *a candidate displaces Qwen3-Coder only if it beats behavioral pass-% by
more than run-to-run noise AND matches/beats idiom on both tiers; ties keep the incumbent
(Apache-2.0, proven).*

- Function behavioral: all survivors 92–95% — a wash within noise.
- Graph behavioral: incumbent DPO **61%** is the best of any DPO-capable model.
- Idiom: `qwen3i` beats the incumbent on graph idiom (0.223 vs 0.338); function idiom is a tie.

No candidate beats the incumbent on **behavioral** beyond noise. `qwen3i` ties behavior and
wins graph idiom, but "ties keep the incumbent."

## Recommendation

**Keep Qwen3-Coder-30B-A3B.** Commit the 300k–500k generation budget to it.

The bake-off did its job: it proved no same-size model decisively beats the incumbent, so the
incumbent is confirmed rather than assumed. If a future reason ever favors switching off the
coder line, **`qwen3i` is the one fallback worth it** — statistically equal on behavior, more
idiomatic on graphs after DPO, same Apache-2.0, same 30B-A3B footprint. `gpt-oss` is the
model to revisit if/when MLX gains working MXFP4 fuse+quant (its SFT idiom is the best in the
field). Everything else is dominated.
