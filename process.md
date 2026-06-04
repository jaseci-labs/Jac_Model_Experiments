# Process — how to run the conversion probe

Start-to-finish for the mini model test: finetune one base model on the
Python→Jac conversion data and measure base-vs-finetuned on the held-out set.
Everything is built; this is the run order.

## 0. Prereqs (one time)

Anaconda was removed on purpose. Use the project venv:

```bash
./setup_env.sh                 # python3 -m venv .venv + pip install jaclang mlx-lm matplotlib
source .venv/bin/activate      # puts jac + mlx_lm.* on PATH
```

You also need disk for a model (~50–60 GB per model to download + quantize).

## 1. Sanity check the toolchain + data

```bash
./check.sh
```

Expect: `20 passed` (full `jac check`) and `32/32 re-validated` (behavior). If
that's green, jac + the pipeline work.

## 2. Confirm the data is in place

Already generated (gitignored under `dataset/`):

| File | What | Count |
|---|---|---|
| `dataset/mlx/train.jsonl` + `valid.jsonl` | training split (mlx `messages` format, 1:3 idiom:transpile) | 417 / 47 |
| `dataset/eval_holdout/conversion.jsonl` | unseen, decontaminated eval tasks (behavioral `test_cases`) | 150 |
| `dataset/conversion/dpo.jsonl` | DPO pairs (later stage) | 85 |

Rebuild only if the data changed:

```bash
jac run srccurrent/jacgen/seed_conversion.jac     # idiomatic core
jac run srccurrent/jacgen/idiomatic_batch.jac     # + batches 2/3
jac run srccurrent/jacgen/scale_conversion.jac    # transpile volume (1500)
jac run srccurrent/jacgen/build_manifest.jac      # 1:3 balanced sft_train.jsonl
jac run srccurrent/jacgen/build_splits.jac        # -> dataset/mlx/{train,valid}.jsonl
jac run srccurrent/jacgen/holdout.jac             # eval holdout (decontaminated)
jac run srccurrent/jacgen/dataset_stats.jac       # composition report
```

## 3. Run the probe

```bash
./run_probe.sh Qwen/Qwen3-Coder-30B-A3B qwen
# or:  ./run_probe.sh google/gemma-4-26b-a4b-it gemma
```

What it does, in order:
1. quantize the model → Q4 (train) + Q8 (eval)
2. **base eval** on the 150 holdout → `results/qwen-base.txt`
3. **30-iter dry-run** — bail check (loss drops, no NaN/OOM); Ctrl-C within 8s to abort
4. **train** (LoRA SFT, `configs/lora.yaml`) with a **live dashboard** every ~60s
   (train/val loss, LR, tokens/sec, and the holdout test-pass **learning curve**
   from a 50-task per-checkpoint eval)
5. **fuse** adapter → Q8
6. **finetuned eval** on the 150 holdout → `results/qwen-finetuned.txt`
7. **graphs** → `results/*.png`

Tunables (env): `EVAL_EVERY` (dashboard secs), `SUBSET` (tasks/checkpoint),
`DRY_ITERS`, `SKIP_DRY=1`.

## 4. Read the result

- **Primary metric:** cross-compiled test pass rate, base vs finetuned, in
  `results/qwen-base.txt` vs `results/qwen-finetuned.txt`. The delta is the signal:
  did finetuning on our Jac data make the model produce more correct Jac?
- **`results/learning_curve.png`** — pass-rate per checkpoint. Rising = it's
  learning Jac; flat-while-loss-drops = it's memorizing format, not idiom.
- Live view during training: the ASCII dashboard (auto-refreshed), or manually
  `JAC_TRAIN_LOG=results/qwen-train.log JAC_METRICS=results/qwen-metrics.jsonl jac run srccurrent/jacgen/dashboard.jac`.

## Notes / limits

- **SFT first.** DPO is a later stage (mlx-lm native DPO is unconfirmed);
  `dpo.jsonl` is ready when you get there.
- The eval calls the converted function by its `func_name` (or the name the model
  defines); a model that produces no callable scores as wrong (fair).
- `configs/lora.yaml` keys vary by mlx-lm version — verify against your install
  (`mlx_lm.lora --help`) before the real run.
- To compare two models, run `run_probe.sh` twice and diff the `-finetuned.txt`
  results (same data, same config — only the base model differs).
