# Python→Jac Conversion Probe

*A small single-category finetuning probe that runs before the full 5,000-example model comparison.*

| | |
|---|---|
| **Objective** | Measure how each candidate model responds to Jac finetuning on one highly-verifiable category |
| **Probe category** | Python→Jac conversion only |
| **Candidates** | Gemma 4 26B A4B, Qwen3-Coder-30B-A3B, DeepSeek-V3-Lite |
| **Dataset** | ~1,500 conversion SFT pairs + ~300-500 DPO pairs |
| **Objective type** | SFT, then a small DPO stage |
| **Hardware / framework** | Mac M5 Pro, 48GB, MLX (Q4 train, Q8 eval) |
| **Primary metric** | Cross-compiled test pass rate (objective) |

---

## Purpose

The full model comparison ([`strategy.md`](strategy.md)) finetunes three models on a 5,000-example multi-category dataset and scores them with a weighted decision matrix. That comparison mixes five categories and three training objectives, which is expensive and confounds many variables. Before paying that cost, the probe answers one isolated question: **which base model learns Jac best from finetuning?**

The probe runs *before* the full comparison and gates which models advance. A model that cannot meaningfully improve on the most-verifiable category is dropped, so the full comparison only spends compute on models that can plausibly learn Jac.

## Why conversion is the probe category

Conversion (Python → idiomatic Jac) is the ideal probe because it has an **objective** correctness signal. Following the MultiPL-T / Recipe 2 methodology (see [`../datagenstrat/strat.md`](../datagenstrat/strat.md)), tests are written in Python where LLMs are reliable, compiled to Jac with a deterministic rule-based test compiler, and run against the translation. Model ranking therefore depends on compiler + behavioral test outcomes, not on a subjective judge. No other category gives this clean a read on raw learning ability.

Conversion also directly stresses the hardest thing about Jac: producing idiomatic, graph-spatial Jac instead of Python-with-Jac-syntax. If a model can be finetuned to convert well, it is a strong candidate for the full pipeline.

## Probe dataset (~1,500 SFT pairs, conversion only)

Generate with the existing Recipe 2 pipeline, restricted to conversion:

- **Source pool:** filtered Python functions — docstring present, Pyright type-check passing, returns a value, no TODO/FIXME, no HumanEval/MBPP overlap, LLM-generated tests at ≥90% line coverage.
- **Translation:** 50–100 candidate Jac translations per function at temperature 0.8, with inferred Python types injected into the prompt.
- **Gate:** cross-compiled test pass is a hard gate. Keep candidates that compile and pass; deduplicate within candidates via ROUGE-L (threshold 0.6).
- **Quality uniformity:** all translations generated with **Claude Max** so data quality is constant and the only variable under test is the base model.
- **Difficulty spread:** ~30% atomic / 40% idiomatic / 30% composed.
- **Construct coverage:** bias source selection toward graph / data-modeling / traversal Python so translations exercise core Jac constructs (node, edge, walker, ability, type annotations) rather than collapsing to plain functions.

**Small DPO set (~300–500 pairs):** chosen = idiomatic, test-passing Jac; rejected = Python-style Jac (a Recipe 3 adversarial negative, or a compile/test failure). This tests whether negative signal moves idiom quality — an early read on the de-Python-ification hypothesis before the full pipeline commits 40k–80k DPO pairs.

## Eval holdout (built BEFORE training)

- ~150–200 held-out Python→Jac tasks, never trained on, each with a reference Jac solution and cross-compiled tests.
- This is a conversion-only subset mirroring **Area 3** of [`evaluation.md`](evaluation.md).
- Decontaminate train vs eval (14-gram overlap) before training, per the quality-control rules in [`../wholestack/strat.md`](../wholestack/strat.md). The holdout must exist before any training data is generated so decontamination is meaningful.

## Training configuration (identical across all three models)

Only the base model differs. Everything else is held constant to isolate model capability.

| Setting | Value |
|---|---|
| Quantization (training) | Q4 |
| Quantization (evaluation) | Q8 |
| Adapter | LoRA, rank 16, alpha 32, dropout 0.05 |
| SFT learning rate | 2e-5, cosine schedule |
| SFT data | ~1,500 pairs, 90/10 train/val split, ~3 epochs |
| DPO stage | ~300–500 pairs, lr 1e-5, beta 0.1, 1–2 epochs, from the SFT adapter |
| Checkpoints | every ~250 steps (for learning-curve analysis) |
| Seeds | 42 and 123 (2 runs per model for variance) |
| Sequencing | sequential, one model at a time (memory) |
| Per-architecture | LoRA target modules and native chat template per model, as in [`strategy.md`](strategy.md) and [`evaluation.md`](evaluation.md) |

The Q4-train / Q8-eval split and LoRA sizing match the memory budget in [`strategy.md`](strategy.md); the probe does not introduce a different training regime.

## Evaluation (objective-first)

Report the **base-vs-finetuned delta** per model on the holdout:

- **Cross-compiled test pass rate — PRIMARY.** Objective behavioral correctness. This is the number that decides the probe.
- **Compiler pass rate.** Necessary-but-insufficient gate.
- **Construct diversity score.** Is the model using Jac constructs, or producing Python-shaped code that happens to compile?
- **Idiom adherence score (secondary).** LLM judge vs `skills.md`. Used only to differentiate models that tie on objective metrics.
- **DPO effect.** Idiom score and Python-ism rate before vs after the DPO stage.
- **Learning curve.** From the per-250-step checkpoints: slope, and whether the model is still improving at the end (headroom for the full 300k dataset) or plateaued early (lower ceiling).
- **Token efficiency.** Tokens-to-correct, consistent with the token-accounting metadata.

Metric definitions are reused verbatim from [`evaluation.md`](evaluation.md); this probe does not redefine them, it uses the conversion subset with a reduced metric set.

## Decision criteria

- The model with the **largest base→finetuned jump in cross-compiled test pass rate** learns Jac conversion best.
- A model **still rising at the end** of training has more headroom for the full dataset.
- A model that **responds strongly to the DPO stage** is early support for the de-Python-ification hypothesis.
- A model that **barely moves** on conversion is dropped before the full 5,000-example comparison.

If the top models are within noise on the primary metric, advance them all to the full comparison rather than forcing a decision here — the probe's job is to eliminate clear losers cheaply, not to make the final call.

## Timeline

~3–5 days:

| Day | Activity |
|---|---|
| Day 1 | Generate ~1,500 conversion SFT pairs + small DPO set via Claude Max with cross-compiled test validation. Build the ~150–200 task holdout first; decontaminate. |
| Day 2–4 | Sequentially SFT + DPO each of the three models (Q4 LoRA), checkpointing every ~250 steps. |
| Day 5 | Merge adapters to Q8, run the conversion eval on base and finetuned, analyze deltas and learning curves, decide which models advance. |

Far cheaper than the full multi-category comparison.

## Relationship to the full comparison

The probe does **not** replace the 5,000-example comparison in [`strategy.md`](strategy.md). It is a pre-step:

1. **Conversion probe** (this doc) — cheap, one category, objective metric → drop clear losers.
2. **Full 5k comparison** ([`strategy.md`](strategy.md)) — five categories, full decision matrix → final model selection.

The data pipeline, recipes, verification, and quality controls are unchanged; the probe only narrows which models reach the full comparison.
