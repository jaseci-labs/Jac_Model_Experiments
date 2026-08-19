# 07-nitin-ds-new-sft — docs index

Phase goal: take **Nitin's NEW corpus** — the successor to the one
`06-nitin-ds-sft` used (`[TODO: confirm exact source repo/commit]`) — turn it
into an SFT + DPO dataset, and run the **same two-arm stock-vs-Spectrum LoRA
probe** that 06 and `04-cpt-sft`'s fresh arm already ran, so that two
independent questions get answered off one training battery:

1. **RQ1 — does Spectrum replicate on a third dataset?** `04-cpt-sft` found a
   significant Spectrum lift at every stage on the fresh arm. `06-nitin-ds-sft`
   did **not** replicate it — one significant win, one significant loss, four
   nulls — and its report closes with *"worth a third replication before
   treating either direction as settled."* This is that third replication, on a
   third independently-sourced corpus, same base checkpoint, same recipe.
2. **RQ2 — is the new dataset better than 06's?** Both arms eval against
   `04-cpt-sft`'s *unchanged* 855-row code-graded holdout, which makes every
   number directly comparable to 06's table cell-for-cell — and, through it,
   to the original jacgen2 fresh-arm table as well.

> **Framing is NOT yet settled for this corpus.** 06's data turned out to be
> flat, Python-transpiled utility functions — no `node`/`edge`/`walker`
> anywhere — the same *shape* as this project's existing
> `python_to_jac_function` task type, not graph-native OSP material. Whether
> the successor corpus is the same shape is `[TODO: verify by sampling]`. Every
> inherited design decision below (single SFT task type, correctness-axis DPO
> instead of the flagship idiomatic axis) follows from 06's finding and is
> **inherited, not reconfirmed**. Do not let a later write-up describe this data
> as "idiomatic graph-native Jac" without evidence — 06 explicitly refused to.

> **Status (2026-08-18): scaffold only.** Directory tree, scripts, configs and
> docs are in place and path-verified. No corpus is pinned, no dataset exists,
> no adapters, no eval numbers. Every count in `dataset-structure.md` is TBD on
> purpose, and every number in `spec.md` §4.2 is a *prior* (06 / 04-cpt-sft)
> result, not a 07 one.

> **Everything here is Jac.** All scripts and drivers in this phase are `.jac`
> (06 mixed `.jac` pipeline scripts with `.py` drivers). Jac compiles to Python
> bytecode, so `mlx` / `mlx_lm` / `mlx_lm_lora` / `matplotlib` import and behave
> unchanged; `.sh` wrappers now call `jac run <driver>.jac <flags>`. See
> `../CONTEXT_BRIEF.md` §5.

## Layout

| File | Contents |
|---|---|
| [`../CONTEXT_BRIEF.md`](../CONTEXT_BRIEF.md) | **Read first, in full.** The settled-facts file: base checkpoint, eval instruments, inherited DPO axis, script-reuse plan, decontam plan, the all-Jac rule, the baseline numbers this phase is measured against, and the failure modes already paid for. |
| [`spec.md`](spec.md) | Full experiment spec: source data + commit pin (TODO), the 2 arms × 2 holdouts × 3 stages design, reuse-vs-new inventory against 06, decontamination requirement, live-monitoring requirement, and the pre-registered acceptance/decision criteria. |
| [`workflow.md`](workflow.md) | The runbook. Exact script names and paths in dependency order: triage → SFT-pair authoring → DPO-pair authoring → SFT ×2 → DPO ×2 → eval ×12 cells → comparison report. |
| [`dataset-structure.md`](dataset-structure.md) | Composition tables for the released SFT/DPO sets, matching 06's format. **Scaffold only** — real counts land when triage does. |
| `reports/corpus-triage-report.md` | **Pending.** Compile-check yield over the whole corpus, within-corpus dedup, decontamination against every existing holdout, and the achievable clean-pool size that sets the holdout carve. |
| `reports/failure_data/` | Per-row generation dumps from `scripts/gen_eval_detail.jac`, for post-hoc failure analysis. |
| `reports/<comparison reports as they land>` | The 12-cell matrix write-up, in the style of `06-nitin-ds-sft/docs/reports/2026-08-final-comparison.md`. |

## Reading order

1. `../CONTEXT_BRIEF.md` — the settled facts, the inherited-but-unconfirmed
   decisions, and every `[TODO]` that must be filled before Stage 0.
2. `spec.md` — what is being measured, against what, and what result counts as
   a win. Read §7 (decision rule) before running anything; it is pre-registered
   so it can't be chosen after seeing numbers.
3. `reports/corpus-triage-report.md` — once it exists. It sets the real pool
   size, and therefore the real holdout size and the real statistical power.
   Several TBDs in `spec.md` and `dataset-structure.md` resolve there.
4. `workflow.md` — the step-by-step execution order.
5. `dataset-structure.md` — what actually got built, after the fact.

## Related, outside this phase

- `model-experiments/06-nitin-ds-sft/` — **the direct predecessor.** Its
  `docs/reports/2026-08-final-comparison.md` holds the numbers this phase
  compares against, and its §4 incident log (external drive disconnect,
  best-checkpoint tracker reset across a resume, bus error, the holdout with no
  `messages` field) is the list of failures to expect and pre-empt.
- `model-experiments/04-cpt-sft/docs/reports/2026-08-spectrum-vs-stock-comparison.md`
  — the original Spectrum result, the lineage baseline, and its §3 incidents
  (the `mlx_lm.fuse` re-quantization bug, the `mx.load` in-place-truncation bug,
  the DPO wired-memory OOM).
- `model-experiments/04-cpt-sft/sft_fresh_probe/` — the ancestor of the entire
  training/eval script suite. Battle-tested; ported, never rewritten.
- `model-experiments/04-cpt-sft/docs/spectrum-plan.md` + `spectrum-workflow.md`
  — the original Spectrum design and runbook, including the `--verify-layers`
  trainable-param gate and the `adapter_config.json` rewrite every eval depends
  on.
- `model-experiments/04-cpt-sft/docs/dpo-plan.md` — the five-axis DPO design.
  This phase inherits 06's use of **only** its correctness axis (§2.3); see
  `spec.md` §3.
- `model-experiments/01-sft-dpo/sft_dpo/jacgen2/decontam_v2.jac` +
  `.../jacgen/dedup2.jac` — the 14-token-shingle / ≥0.5-overlap machinery
  `scripts/pipeline.jac` carries forward rather than rebuilding.
