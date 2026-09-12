# Pipeline check — 2026-09-11

Every 08 pipeline script was run for real on scratch inputs after the two
restructures of the day (flatten, then the move into
`experiments/08-nitin-new2-ds/{scaffold,train,eval}`). The first pass was
CPU-only; the model-loading steps ran on 2026-09-12 (see [GPU run](#gpu-run-2026-09-12)).

## Results

| Step | Result |
|---|---|
| `setup_env.sh` checks | pass: jac, mlx_lm + mlx_lm_lora, base model found, `bash -n`, `jac check` |
| `scaffold/copy_holdout.jac` → `release.jac` → `prep_training_dirs.sh` + `nan_guard.jac` | **byte-identical** to the committed 08 data: candidate_pool, sft_train (14,792), dpo_train (1,375), release.json, splits 12,570 / 2,217 SFT and 1,169 / 206 DPO; nan_guard drops 5 over-long rows |
| `scaffold/pipeline.jac` (full corpus merge) | **byte-identical** (2026-09-12 run, CPU, peak 569 MB): 16,167 pool rows, 0 holdout/denylist leaks; `candidate_pool`, `rejected/sft` (9,404) and `rejected/dpo` (224) match 08 byte for byte; `stats.json` differs only in `corpus_root` (machine path) |
| train scripts | all pass `bash -n`; `--verify-patches` passes |
| refuse-to-retrain guards | spectrum, stock and both DPO runners all refuse to retrain over a finished adapter with no run markers (tested with a fake `jac`, so nothing trained) |
| eval helpers | `plot_metrics`, `plot_progress`, `grade_reference`, `grade_eval_detail` run on a few rows; `eval_functional` ran end to end against a CPU stand-in for `mlx_lm` |
| `jac check` | all 14 pipeline `.jac` files pass (it used to crash on `eval_functional.jac`) |
| stale paths | code and configs clean after both moves |

## Bugs fixed

| # | Bug | Cause | Fix | Commit |
|---|---|---|---|---|
| 1 | Stock SFT could overwrite a finished adapter | `train/stock/run_sft.sh` had no refuse-to-retrain guard: an adapter with no markers restarted from step 0 | guard added (same as the spectrum runner) | `2d4b642` |
| 2 | Both DPO runners could overwrite a finished DPO adapter | a finished adapter with no progress file re-seeded from SFT and retrained | guards added | `f70e02a` |
| 3 | `jac check` crashed on `eval/eval_functional.jac` (`NoneType … is_instantiable_class`) | the 0.16.1 checker crashes on any `mlx_lm.load` call | that call moved into an inline Python block (the workaround `gen_eval_detail.jac` already uses); the type errors the crash had hidden are fixed | `f70e02a` |
| 4 | Step 8200 in `metrics_functional.jsonl` looked double-written (12 rows, not 6) | not a double write: the last sweep checkpoint (100-row subset) and the final full 855-row eval both carried step 8200 | rows now carry `"scope"` (full/subset) and `"limit"`; `plot_progress` drops subset rows at steps that have a full eval and infers scope for old rows; DPO runners prefer `scope=full` for the SFT baseline | `f70e02a` |
| 5 | `setup_env.sh` pointed at a JMS path that no longer exists | stale after the repo split | now `start.sh` in the sibling repo `Jac_Model_Studio/` | `1c48b7c` |
| 6 | With no adapter present, `eval_sft_sweep.sh` truncated `metrics_functional.jsonl`, ran a full base-model eval, then crashed in `mlx_lm.load`; `eval_dpo_nofuse.sh` loaded the base model before crashing the same way | neither runner checked that its adapter exists before loading | both stop at the top with `MISSING: <path>` (exit 1) before any truncation or model load. Verified: all three no-adapter runners exit in 0s with no model loaded | `c6605e7` |

## GPU run (2026-09-12)

On the flattened layout (`experiments/08-nitin-new2-ds/{scaffold,train,eval}`),
from scratch copies, one model at a time under `/tmp/jac-gpu.lock`:

| Step | Result |
|---|---|
| spectrum `--verify-layers` | **VERIFY: PASS**: 281.838M / 5054.233M trainable, 256 LoRA tensors, trailing-16 control matches |
| spectrum 30-iter dry run | pass: val loss 0.890 → 0.768, peak memory 27.1 GB, stops at the `CONFIRM_FULL_RUN` gate; the real 08 adapter is untouched (byte-identical) |
| stock 30-iter dry run | pass: val loss 0.890 → 0.766, peak memory 26.7 GB, stops at the gate |
| spectrum DPO dry run | preflight passes (`--verify-patches`, and `--verify-layers` with the SFT adapter loaded: 256 keys, 281.838M trainable); policy + reference + SFT adapter loaded, DPO mode entered, then **stopped by the memory gate before its first iteration** (0/8): free memory fell to 6% (gate is 8%), swap grew 5.8 GB |
| stock DPO dry run | skipped: same memory profile as spectrum DPO |
| no-adapter runners (`eval_sft_sweep`, `eval_dpo_spectrum`, `eval_dpo_nofuse`) | each prints `MISSING: <path>` and exits 1 before any model load (fix #6) |

**Known limit.** DPO on this box needs the other big memory users closed first
(browser, IDE, a second Claude session, any JMS server). It isn't a code bug.
Experiment 07's full DPO run completed here under lighter memory load. Don't
override the gate by running a second model alongside.

### Eval checks

| Step | Result |
|---|---|
| `eval_sft_spectrum.sh` on 10 holdout (a) rows + the step-8200 checkpoint | pass: 256 adapter keys load after the config rewrite; every row carries `scope`/`limit`. Step 8200 has a subset row (`scope=subset, limit=5`) and a full row (`scope=full`); `plot_progress` plots the full one. (The 10 rows are all UI-component tasks, hence the low absolute score; not a regression.) Swap growth 0, lowest free 45% |
| `gen_eval_detail` + `grade_eval_detail` on the same 10 rows | pass: 10/10 non-empty generations in 63s; the grade matches the eval |

Everything in the pipeline now has a real run behind it, except DPO training,
which needs the memory headroom described above.

The function eval of 08 on jac-data-gen's hidden tests is a separate report:
[`experiments/08-nitin-new2-ds/function-eval-report.md`](../../experiments/08-nitin-new2-ds/function-eval-report.md).
