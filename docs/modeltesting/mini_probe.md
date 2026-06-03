# Mini Conversion Probe — readiness

Everything needed to run the SFT conversion probe **except** the ML itself. The
data is packaged and the eval scoring loop is proven; training is one command
and scoring is one command once `mlx-lm` and a model are present.

## What is prebuilt (data-prep, done)

| Artifact | Path |
|---|---|
| Balanced training manifest (1:3 idiom:transpile, 464) | `dataset/conversion/sft_train.jsonl` |
| mlx-lm splits (messages format) | `dataset/mlx/train.jsonl` (417) + `valid.jsonl` (47) |
| Decontaminated eval holdout (150, behavioral cases) | `dataset/eval_holdout/conversion.jsonl` |
| DPO pairs (compile-gated, 85) | `dataset/conversion/dpo.jsonl` |
| LoRA config (scaffold) | `configs/lora.yaml` |
| Eval harness (model-agnostic) | `srccurrent/jacgen/eval_probe.jac` |
| Runner (scaffold) | `run_probe.sh` |

The eval scoring loop is validated with a `jac py2jac` stand-in (no LLM):
`jac run srccurrent/jacgen/eval_probe.jac` (default `JAC_EVAL_MODE=py2jac`).

## What is NOT done (needs your resources — no ML happened here)

Environment note: anaconda was intentionally removed. Install these into whatever
Python you standardize on (must be on PATH):

- `pip install jaclang` — the `jac` CLI the whole pipeline shells out to
- `pip install mlx-lm` — training + generation (`mlx_lm.*` console scripts)
- `pip install matplotlib` — PNG graphs (the ASCII dashboard needs nothing)
- download + quantize a model (~50-60 GB each): Gemma 4 26B A4B / Qwen3-Coder-30B-A3B
- the actual LoRA training run and the base-vs-finetuned eval

## Live metrics + graphs

`run_probe.sh` does: quantize → base eval → **30-iter dry-run** (bail check) →
full train (tee to `results/<name>-train.log`) with a live loop that, every ~60s,
runs a **50-task adapter eval** (no fuse) appending a learning-curve point to
`results/<name>-metrics.jsonl` and redraws the **ASCII dashboard**
(`dashboard.jac`). After training: fuse → full 150 eval → **PNG graphs**
(`plot_metrics.jac`).

- `dashboard.jac` (zero-dep): train/val loss, LR, tokens/sec, and the holdout
  test-pass learning curve — live in the terminal.
- `plot_metrics.jac` (matplotlib): the same as `results/*.png`; the learning
  curve (`results/learning_curve.png`) is the one that tells you if it's learning
  Jac, not just lowering loss.
- Config intervals: `configs/lora.yaml` reports train loss every 10 steps, val
  loss every 50, checkpoints every 100.

## Run order (later)

```bash
pip install mlx-lm
# rebuild splits if data changed:
jac run srccurrent/jacgen/build_splits.jac
# set configs/lora.yaml `model:` to your Q4 path, then:
./run_probe.sh Qwen/Qwen3-Coder-30B-A3B qwen
```

`run_probe.sh` quantizes → base eval → LoRA SFT → fuse → finetuned eval, writing
`results/<name>-base.txt` and `results/<name>-finetuned.txt`. The eval harness
reads `JAC_EVAL_MODE` / `JAC_EVAL_MODEL` from the environment (no source edits).

## Metrics

`eval_probe.jac` reports, over the 150 holdout tasks:
- **compiler pass rate** — generated Jac type-checks (`jac check`)
- **cross-compiled test pass rate** — generated Jac runs and matches the recorded
  behavioral `test_cases` (the probe's primary, objective metric)

## Notes / limits

- **SFT-first.** DPO is a later stage; mlx-lm native DPO support is unconfirmed.
  `dpo.jsonl` is ready for whatever DPO path is chosen.
- The eval calls the converted function by its original `func_name`; a model that
  renames it scores as wrong (mechanical but fair).
- Holdout carries behavioral cases, not reference Jac — sufficient for the
  primary metric; an idiom-judge over outputs can be added later.
- `configs/lora.yaml` keys vary by mlx-lm version — verify against your install.
