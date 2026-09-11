# experiments/ — one directory per experiment

Every experiment has the same shape:

```
NN-name/
  adapter/     adapters.safetensors (1.1 GB, gitignored) + adapter_config.json
  results/     eval logs (final*.txt), metrics_functional.jsonl; for 08 also train.log, plots, holdoutB/
  report.md    the experiment's final report
```

08 also keeps `dataset/` (its full dataset, including the shared holdout (a))
and `docs/` (spec, workflow, CONTEXT_BRIEF, dataset structure, corpus triage,
funnel stats). New experiments go here too; the pipeline at the repo root
(`1-scaffold/`, `2-train/`, `3-eval/`) takes `EXP=experiments/NN-name`.

The original experiment trees of 04, 06 and 07 were deleted 2026-09-11. Their
`adapter/` holds the **single best-scoring final adapter** of that experiment,
byte-identical (`cmp`) to its source, and `results/` the eval output that
produced its score. Reports of 01 to 05 that are not an experiment's final
report stay in `../docs/history/`.

All four are Spectrum-layer LoRA adapters (rank 16, scale 2.0, 16 SNR-selected
blocks out of 48 — the layer list is inside each `adapter_config.json`) on the
base **Qwen3-Coder-30B-A3B 4-bit** at `models/qwen-q4`.
No fused models — adapter only.

| dir | what | holdout (a) shared 855 | holdout (b) Nitin 855 | original training path |
|---|---|---|---|---|
| `04-cpt-sft/` | fresh-arm Spectrum SFT-final (jacgen2 data, no CPT) — beat its DPO-best 634 and DPO-final 622 | **74.7% (639/855)** | — | `04-cpt-sft/sft_fresh_probe/adapters/sft-on-fresh-spectrum/` |
| `06-nitin-ds-sft/` | Spectrum SFT-final on Nitin ds v1 (ties its DPO-final 636 on (a); SFT wins on (b) 837 vs 825) | **74.4% (636/855)** | 97.9% (837/855) | `06-nitin-ds-sft/spectrum_probe/adapters/sft-on-nitin-spectrum/` |
| `07-nitin-ds-new-sft/` | Spectrum DPO-best (step 120 of 250; `.best_step`=120, cmp-identical to `snapshots/step_0120`) on Nitin ds v2 | **73.0% (624/855)** | 97.9% (837/855) | `07-nitin-ds-new-sft/spectrum_probe/adapters/dpo-on-sft-nitin-spectrum-best/` |
| `08-nitin-new2-ds/` | Spectrum SFT-final on the 7-source merge | 70.5% (603/855) | 37.7% (322/855), 07's holdout | `08-nitin-new2-ds/spectrum_probe/adapters/sft-on-nitin-spectrum/` (pre-flatten) |

Holdout (a) = `experiments/08-nitin-new2-ds/dataset/holdout_a_shared855.jsonl` (1428 rows, 855
code-graded). Holdout (b) is each experiment's own py2jac holdout, and they are
not the same file: 07's (b) is `experiments/08-nitin-new2-ds/dataset/nitin_holdout_eval.jsonl`
(08 reused it byte-for-byte); 06's (b) was a different 855-row carve that was
deleted with 06 (its `final_holdoutB.txt` dates from 2026-08-13, before 07's
holdout existed). So 06's and 07's (b) numbers are not comparable to each other.
Score = "OVERALL runs" of `3-eval/eval_functional.jac` (compiles + executes).
Full history and context: [`../docs/HISTORY.md`](../docs/HISTORY.md).

Result files: `metrics_functional.jsonl`, `final*.txt` (full-holdout eval log;
`*_holdoutB.txt` for (b); 08 keeps its (b) run in `results/holdoutB/`).

`adapter_config.json` edits vs source: `"model"` retargeted to
`models/qwen-q4` (07's DPO config had no `model` key; one was
added). The `adapter_path` / `data` / `config` fields in 04/06 are training
provenance only (not read at load time) and still name the deleted dirs; 08's
were updated to the flattened layout.

## Load / eval (run from the repo root)

```bash
# functional eval (same harness as every reported number)
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL=models/qwen-q4 \
  JAC_EVAL_ADAPTER=experiments/06-nitin-ds-sft/adapter \
  JAC_HOLDOUT=experiments/08-nitin-new2-ds/dataset/holdout_a_shared855.jsonl \
  .venv/bin/jac run 3-eval/eval_functional.jac

# python
.venv/bin/python -c "from mlx_lm import load, generate; \
m,t = load('models/qwen-q4', adapter_path='experiments/06-nitin-ds-sft/adapter'); \
print(generate(m, t, prompt='Write a Jac walker that ...', max_tokens=256))"
```

The config already carries the rewritten Spectrum layer list (`num_layers: 48` +
`spectrum_layers`), so plain `mlx_lm.load` covers every trained block — do not
restore the original `num_layers: 16` or out-of-slice layers are silently dropped.
