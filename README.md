# model-experiments

Fine-tuning experiments for a Jac coding model: LoRA on
Qwen3-Coder-30B-A3B-Instruct (4-bit MLX), trained and evaluated on one Apple
Silicon Mac with 48 GB. Eight experiments so far (01 to 08). The current recipe
is plain SFT on 16 Spectrum-selected decoder blocks, scored by whether the
generated Jac compiles and runs.

- **[docs/PLAYBOOK.md](docs/PLAYBOOK.md)**: how to run a new experiment
  (scaffold, Spectrum SFT, eval, report). Start here.
- **[docs/HISTORY.md](docs/HISTORY.md)**: what 01 to 08 tried and what came of
  it. Raw reports are in [docs/history/](docs/history/).

## Layout

```
model-experiments/
  08-nitin-new2-ds/     latest experiment, kept whole; the template for new ones
    scripts/            dataset pipeline, NaN guard, eval harness (eval_functional.jac), plotting
    dataset/            holdout_a_shared855.jsonl (shared baseline), holdout (b), SFT/DPO releases + splits
    spectrum_probe/     Spectrum SFT/DPO runners + eval, layer list, trained adapter, results
    stock_probe/        stock trailing-16 control arm (scripts only)
    docs/               spec, workflow, dataset structure, reports/
  archive/              best adapters of 04, 06, 07 (adapter-only, with eval logs)
  models/qwen-q4/       base model, MLX 4-bit (gitignored, single copy)
  docs/
    PLAYBOOK.md  HISTORY.md
    history/            key reports from the deleted experiments 01 to 07
    reference/          Spectrum design, original whole-stack strategy
    blog/  presentation/
  setup_env.sh          builds ../.venv and sanity-checks 08
```

## Best models

Holdout (a) is the shared 855-row code-graded holdout
(`08-nitin-new2-ds/dataset/holdout_a_shared855.jsonl`). The score is the share
of generations that compile and run under
`08-nitin-new2-ds/scripts/eval_functional.jac`. The untrained base scores
10.5% (90/855).

| adapter | what | holdout (a) | holdout (b) |
|---|---|---|---|
| [`archive/04-cpt-sft/`](archive/04-cpt-sft/) | 04 fresh-arm Spectrum SFT (jacgen2 data) | **74.7% (639/855)** | not run |
| [`archive/06-nitin-ds-sft/`](archive/06-nitin-ds-sft/) | 06 Spectrum SFT (Nitin data v1) | 74.4% (636/855) | 97.9% (837/855), 06's own holdout |
| [`archive/07-nitin-ds-new-sft/`](archive/07-nitin-ds-new-sft/) | 07 Spectrum DPO-best, step 120 (Nitin data v2) | 73.0% (624/855) | 97.9% (837/855), 07's holdout |
| [`08-nitin-new2-ds/spectrum_probe/adapters/sft-on-nitin-spectrum/`](08-nitin-new2-ds/spectrum_probe/adapters/sft-on-nitin-spectrum/) | 08 Spectrum SFT (7-source merge) | 70.5% (603/855) | 37.7% (322/855), 07's holdout |

The differences among 04, 06 and 07 are not statistically significant. 04 is
the one to use. All four are adapter-only and need the base model. Their
`adapter_config.json` already carries the rewritten Spectrum layer list, so a
plain `mlx_lm.load(base, adapter_path=...)` loads every trained block. Details:
[archive/README.md](archive/README.md).

## Quickstart

```bash
# from the directory that contains model-experiments/
./model-experiments/setup_env.sh
.venv/bin/pip install jaclang==0.16.1 mlx==0.31.2 mlx-lm==0.31.3 mlx-lm-lora==2.1.0   # the versions every number was produced with
source .venv/bin/activate

# score the best adapter on holdout (a)  (~1 hour; nothing else may hold a model in memory)
JAC_EVAL_MODEL=model-experiments/models/qwen-q4 \
JAC_EVAL_ADAPTER=model-experiments/archive/04-cpt-sft \
JAC_HOLDOUT=model-experiments/08-nitin-new2-ds/dataset/holdout_a_shared855.jsonl \
  jac run model-experiments/08-nitin-new2-ds/scripts/eval_functional.jac
```

The scripts expect this folder to be named `model-experiments` and to sit
directly inside the workspace that holds `.venv/`. They `cd` to that
workspace and use `model-experiments/...` paths. The base model must be at
`model-experiments/models/qwen-q4`. It is not in git; see the playbook
prerequisites for how to rebuild it.
