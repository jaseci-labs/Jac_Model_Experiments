# Pipeline check — 2026-09-11

Every 08 pipeline script was run for real on scratch inputs after the two
restructures of the day (flatten, then the move into
`experiments/08-nitin-new2-ds/{scaffold,train,eval}`). This pass is CPU-only.
The model-loading steps are listed at the end and haven't run yet.

## Results

| Step | Result |
|---|---|
| `setup_env.sh` checks | pass: jac, mlx_lm + mlx_lm_lora, base model found, `bash -n`, `jac check` |
| `scaffold/copy_holdout.jac` → `release.jac` → `prep_training_dirs.sh` + `nan_guard.jac` | **byte-identical** to the committed 08 data: candidate_pool, sft_train (14,792), dpo_train (1,375), release.json, splits 12,570 / 2,217 SFT and 1,169 / 206 DPO; nan_guard drops 5 over-long rows |
| `scaffold/pipeline.jac` (full corpus merge) | not run yet: needs memory headroom. The corpus clone is at `../jac-data-gen` @ `c95b7563` |
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
| 5 | `setup_env.sh` pointed at a JMS path that no longer exists | stale after the repo split | now `../Jac_Model_Studio/start.sh` | `1c48b7c` |

## Still to run (needs the GPU)

- `pipeline.jac` full run from a scratch copy (`JDG_CLONE` set, then remove the
  worktree), diffed against the committed pool
- spectrum `--verify-layers` (expect `VERIFY: PASS`, 281.838M trainable)
- spectrum 30-iteration dry run and stock dry run
- 8-iteration DPO dry runs
- `eval_sft_spectrum.sh` on a small holdout subset against the 08 adapter,
  confirming the new `scope` rows on real output
- `eval_sft_sweep` and the DPO eval runners (no stock or DPO adapters exist for
  08, so they should fail clearly)
- `gen_eval_detail`

The function eval of 08 on jac-data-gen's hidden tests is a separate report:
[`experiments/08-nitin-new2-ds/function-eval-report.md`](../../experiments/08-nitin-new2-ds/function-eval-report.md).
