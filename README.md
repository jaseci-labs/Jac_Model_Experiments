# model-experiments

Fine-tuning experiments for a Jac coding model: LoRA on
Qwen3-Coder-30B-A3B-Instruct (4-bit MLX), trained and evaluated on one Apple
Silicon Mac with 48 GB. Eight experiments so far (01 to 08). The current recipe
is plain SFT on 16 Spectrum-selected decoder blocks, scored by whether the
generated Jac compiles and runs.

- **[PLAYBOOK.md](PLAYBOOK.md)**: how to run a new experiment, step by step
  (scaffold, Spectrum SFT, eval, report). Start here.
- **[docs/HISTORY.md](docs/HISTORY.md)**: what 01 to 08 tried and what came of
  it.
- **[experiments/](experiments/)**: every experiment that left an adapter, one
  directory each.

## Layout

```
./                      repo root; run everything from here (scripts cd here themselves)
  PLAYBOOK.md           the step-by-step guide
  experiments/
    04-cpt-sft/  06-nitin-ds-sft/  07-nitin-ds-new-sft/
                        each: adapter/, results/, report.md  (no scripts; predate this recipe)
    08-nitin-new2-ds/   adapter/, results/, report.md, docs/, dataset/, and the pipeline itself:
      scaffold/         dataset build: pipeline.jac, copy_holdout.jac, release.jac,
                        prep_training_dirs.sh (85/15 split) + nan_guard.jac
      train/            run_sft_spectrum.sh (the recipe), spectrum/ (layer list, driver,
                        adapter_config_fix, SNR scan), configs/;
                        stock/ (trailing-16 control arm) and dpo/ (optional DPO pass)
      eval/             eval_functional.jac (the one scoring harness), eval_*.sh runners,
                        per-row dump/grade scripts, plotting
  models/qwen-q4/       base model, MLX 4-bit (gitignored, single copy)
  docs/                 HISTORY.md, history/ (reports of 01 to 05), reference/, blog/, presentation/
  setup_env.sh          builds .venv (here, at the repo root) and sanity-checks the pipeline
  .venv/                python venv (gitignored)
```

There is no `EXP` env var. Every script under an experiment's `scaffold/`,
`train/` and `eval/` derives that experiment from its own location, and `cd`s
to the repo root itself — so scripts are invoked by path, from anywhere, e.g.
`experiments/08-nitin-new2-ds/train/run_sft_spectrum.sh`. Data is read from
that experiment's `dataset/`; the adapter goes to its `adapter/`, logs and
evals to its `results/`. A new experiment gets its own copies of
`scaffold/`, `train/` and `eval/` (see [PLAYBOOK.md](PLAYBOOK.md)).

## Results

Holdout (a) is the shared 855-row code-graded holdout
(`experiments/08-nitin-new2-ds/dataset/holdout_a_shared855.jsonl`). The score is
the share of generations that compile and run under
`experiments/08-nitin-new2-ds/eval/eval_functional.jac`.
The untrained base scores 10.5% (90/855).

| experiment | adapter | holdout (a) | holdout (b) | report |
|---|---|---|---|---|
| [04-cpt-sft](experiments/04-cpt-sft/) | fresh-arm Spectrum SFT-final (jacgen2 data) | **74.7% (639/855)** | not run | [report](experiments/04-cpt-sft/report.md) |
| [06-nitin-ds-sft](experiments/06-nitin-ds-sft/) | Spectrum SFT-final (Nitin data v1) | 74.4% (636/855) | 97.9% (837/855), 06's own holdout | [report](experiments/06-nitin-ds-sft/report.md) |
| [07-nitin-ds-new-sft](experiments/07-nitin-ds-new-sft/) | Spectrum DPO-best, step 120 (Nitin data v2) | 73.0% (624/855) | 97.9% (837/855), 07's holdout | [report](experiments/07-nitin-ds-new-sft/report.md) |
| [08-nitin-new2-ds](experiments/08-nitin-new2-ds/) | Spectrum SFT-final (7-source merge) | 70.5% (603/855) | 37.7% (322/855), 07's holdout | [report](experiments/08-nitin-new2-ds/report.md) |

The differences among 04, 06 and 07 are not statistically significant. 04 is
the one to use. All four are adapter-only (`experiments/<name>/adapter/`) and
need the base model. Their `adapter_config.json` already carries the rewritten
Spectrum layer list, so a plain `mlx_lm.load(base, adapter_path=...)` loads
every trained block. Details: [experiments/README.md](experiments/README.md).

## Quickstart

```bash
# from the repo root
./setup_env.sh
source .venv/bin/activate

# score the best adapter on holdout (a)  (~1 hour; nothing else may hold a model in memory)
JAC_EVAL_MODEL=models/qwen-q4 \
JAC_EVAL_ADAPTER=experiments/04-cpt-sft/adapter \
JAC_HOLDOUT=experiments/08-nitin-new2-ds/dataset/holdout_a_shared855.jsonl \
  jac run experiments/08-nitin-new2-ds/eval/eval_functional.jac

# a new experiment, end to end (details in PLAYBOOK.md)
mkdir experiments/09-my-dataset
cp -r experiments/08-nitin-new2-ds/{scaffold,train,eval} experiments/09-my-dataset/
# ... build experiments/09-my-dataset/dataset/sft_train.jsonl, then:
experiments/09-my-dataset/scaffold/prep_training_dirs.sh
experiments/09-my-dataset/train/run_sft_spectrum.sh                      # self-test + dry run, then exits
CONFIRM_FULL_RUN=1 experiments/09-my-dataset/train/run_sft_spectrum.sh   # the ~4 h run
experiments/09-my-dataset/eval/eval_sft_spectrum.sh                      # holdout (a)
```

`setup_env.sh` pins `jaclang==0.16.1 mlx==0.31.2 mlx-lm==0.31.3 mlx-lm-lora==2.1.0`,
the versions every number was produced with. The base model must be at
`models/qwen-q4`. It is not in git; see the playbook prerequisites for how to
rebuild it.
