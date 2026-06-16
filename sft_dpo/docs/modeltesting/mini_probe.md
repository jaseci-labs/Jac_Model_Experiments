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
| DPO pairs (compile-gated, 60) | `dataset/conversion/dpo.jsonl` |
| LoRA config (scaffold) | `configs/lora.yaml` |
| Eval harness (model-agnostic) | `srccurrent/jacgen/eval_probe.jac` |
| Runner (scaffold) | `run_probe.sh` |

The eval scoring loop is validated with a `jac py2jac` stand-in (no LLM):
`jac run srccurrent/jacgen/eval_probe.jac` (default `JAC_EVAL_MODE=py2jac`).

## What is NOT done (needs your resources — no ML happened here)

Environment (anaconda intentionally removed — use a project venv):

```bash
./setup_env.sh                 # python3 -m venv .venv + pip install jaclang mlx-lm matplotlib
source .venv/bin/activate      # jac + mlx_lm on PATH
./check.sh                     # jac check (19) + eval_probe parse + non-destructive jac run audit
```

`run_probe.sh` and `check.sh` auto-prepend `.venv/bin` to PATH, so the pipeline's
internal `jac`/`mlx_lm` subprocess calls resolve without activation.

**On `jac check`:** 19 modules pass the **full** type-checker; `eval_probe.jac` is
**parse-checked only** (`jac check -p`) because jaclang 0.16.0's type-checker
*crashes* resolving `mlx_lm`'s model types (internal bug) — so `eval_probe.jac`
lazy-imports `mlx_lm` *inside* its functions, never at top level. Dynamic
Python-interop (`json.loads`/`subprocess`/regex/matplotlib return `Any`, which
jaclang 0.16.0 won't assign to typed vars) is handled with `str()`/`list()`/
`dict()`/`int()` casts at the boundary; a few genuinely-untypeable stdlib calls
(`inspect.signature`, `signal`, matplotlib stubs) carry `# jac:ignore[...]`.
`check.sh` runs full `jac check` (19) + parse-check (eval_probe) + a **sampled,
non-destructive** `jac run` re-validation (`verify_dataset.jac`, the real gate —
behavioral, how every dataset example was validated). It does **not** mutate the
dataset. (Earlier it ran `seed_conversion.jac`, which truncated `sft.jsonl` to 32
every invocation — fixed.)

Still needed for the actual probe: download + quantize a model (~50-60 GB each)
and run `./run_probe.sh <model> <name>`. Full handoff with all gotchas:
`docs/modeltesting/HANDOFF.md`.

## Live metrics + graphs

`run_probe.sh` does: quantize → base eval → **30-iter dry-run** (bail check) →
full train (redirected — not piped — to `results/<name>-train.log` so `$!` is the
trainer) with a live loop that, every `EVAL_EVERY`s, runs a `SUBSET`-task adapter
eval (no fuse) appending a learning-curve point to `results/<name>-metrics.jsonl`
and redraws the **ASCII dashboard** (`dashboard.jac`) + the **PNG graphs**
(`plot_metrics.jac`). After training: fuse → full 150 eval → final graphs.
The eval loads the model **once** in-process (mlx) and reuses it across the whole
subset — an earlier version reloaded the 30B model per task and was unusably slow.

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
# run (the script passes --model to override configs/lora.yaml's placeholder path):
./run_probe.sh Qwen/Qwen3-Coder-30B-A3B-Instruct qwen   # -Instruct (bare id 401s)
```

`run_probe.sh` quantizes → base eval → LoRA SFT → fuse → finetuned eval, writing
`results/<name>-base.txt` and `results/<name>-finetuned.txt`. The eval harness
reads `JAC_EVAL_MODE` / `JAC_EVAL_MODEL` from the environment (no source edits).

## Metrics

`eval_probe.jac` reports, over the 150 holdout tasks (gate is `jac run`, never
`jac check`):
- **runs rate** — generated Jac compiles and executes (`jac run` exit 0)
- **cross-compiled test pass rate** — generated Jac runs and matches the recorded
  behavioral `test_cases` (the probe's primary, objective metric)
- **generation tokens**, **eval tokens/sec** (avg `generation_tps`), and
  **tokens-to-correct** (avg generated tokens per correct conversion)

## Notes / limits

- **SFT-first.** DPO is a later stage; mlx-lm native DPO support is unconfirmed.
  `dpo.jsonl` is ready for whatever DPO path is chosen.
- The eval calls the converted function by its original `func_name`; a model that
  renames it scores as wrong (mechanical but fair).
- Holdout carries behavioral cases, not reference Jac — sufficient for the
  primary metric; an idiom-judge over outputs can be added later.
- `configs/lora.yaml` keys vary by mlx-lm version — verify against your install.
- Resume specifics vary by mlx-lm version: `run_probe.sh` assumes checkpoints named
  `NNNN_adapters.safetensors` and a `--resume-adapter-file` flag. If your mlx-lm
  uses different names/flags, adjust the discovery block. On resume the LR schedule
  restarts per-invocation (loads weights, re-warms up) — continues learning, not a
  perfectly-stitched resume.
