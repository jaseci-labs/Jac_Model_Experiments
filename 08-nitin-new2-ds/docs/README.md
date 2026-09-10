# 08-nitin-new2-ds — docs index

Phase goal: fold **all four new source types** Nitin's `jac-data-gen` repo
picked up since 07's pin (OSP idiomize gold slice, js2jac translation corpus,
step4 DPO pairs, step4 work-derived SFT pairs) in alongside a filtered
successor of the original py2jac corpus, fix a real eval-leakage bug neither
06 nor 07 caught, and run the same two-arm stock-vs-Spectrum LoRA probe this
lineage has run three times before.

1. **RQ1 — does Spectrum replicate on a fourth dataset?** 04-cpt-sft found a
   significant lift at every stage; 06 did not replicate it; 07 is the most
   recent prior result. This is the next replication.
2. **RQ2 — is this multi-source corpus better than 07's?** Measured
   cell-for-cell against 07 on the identical shared holdout (a).
3. **RQ3 — new — does real graph-native (OSP) and CRUD-walker (farm) content
   change the picture, beyond a same-shape-more-data effect?** 06 and 07 both
   verified zero OSP archetypes in their corpus; 08 has 2,537 genuinely
   graph-native rows (15.7% of the pool). See `../CONTEXT_BRIEF.md` §1.2.

> **Status (2026-09-10): dataset complete, training not started.** Every
> count in `dataset-structure.md` is real, taken from the built releases —
> not TBD, not carried forward from a prior phase. No adapters, no eval
> numbers exist yet.

> **Everything here is Jac.** All pipeline and driver scripts are `.jac`,
> ported unchanged from 07's all-Jac convention. `.sh` wrappers call
> `jac run <driver>.jac <flags>`.

## Layout

| File | Contents |
|---|---|
| [`../CONTEXT_BRIEF.md`](../CONTEXT_BRIEF.md) | **Read first, in full.** Settled facts: the 7-source merge, the eval-denylist fix, the quality-gradient filter, the DPO-axis change from 07, the graph-native framing warning, the holdout-(b)-reused-not-recarved decision, and the drive-disconnect incident during this phase's own build. |
| [`spec.md`](spec.md) | Full experiment spec: source inventory, the 2 arms × 2 holdouts × 3 stages design, reuse-vs-new inventory against 07, decontamination + leakage-fix requirement, pre-registered acceptance/decision criteria. |
| [`workflow.md`](workflow.md) | The runbook. Dataset build (Stage 0, DONE) is documented as executed; training → eval → report (Stages 3-6) as the remaining runbook. |
| [`dataset-structure.md`](dataset-structure.md) | Composition tables for the released SFT/DPO sets — real counts, not TBD. |
| [`reports/corpus-triage-report.md`](reports/corpus-triage-report.md) | The full funnel: 7 sources in, every filter's drop count and reason, the leakage check, the final release sizes. |
| `reports/failure_data/` | Per-row generation dumps for post-hoc failure analysis (populated once eval runs). |
| `reports/<comparison report, once it lands>` | The 12-cell matrix write-up, in the style of 07's and 06's final comparison reports. |

## Reading order

1. `../CONTEXT_BRIEF.md` — the settled facts, especially §1.2 (the
   graph-native framing reversal) and §4 (why holdout (b) is reused rather
   than re-carved).
2. `spec.md` §7 — the pre-registered decision rule, before running anything.
3. `reports/corpus-triage-report.md` — the real funnel and final counts.
4. `workflow.md` — step-by-step execution order for what's left (training,
   eval, report).
5. `dataset-structure.md` — what actually got built.

## Related, outside this phase

- `model-experiments/07-nitin-ds-new-sft/` — the direct predecessor. Its
  `docs/reports/07-final-comparison.md` holds the numbers this phase compares
  against, and is the direct motivation for §2.2's quality-gradient filter
  (07's larger-but-weaker pool did not beat 06's).
- `model-experiments/06-nitin-ds-sft/docs/reports/2026-08-final-comparison.md`
  — one phase further back; its §4 incident log (external drive disconnect,
  best-checkpoint tracker reset, holdout with no `messages` field) is the
  failure list `../CONTEXT_BRIEF.md` §11 carries forward, now with a third
  drive-disconnect incident added from this phase's own build.
- `~/repos/jac-data-gen/docs/PY2JAC_QUALITY_GRADIENT.md` — the audit that
  motivates §2.2's test-count filter; read it before changing the cutoff.
- `~/repos/jac-data-gen/evals/function/v1/denylist_ids.txt` +
  `scripts/eval/apply_function_eval_denylist.py` — the eval-leakage fix this
  phase applies for the first time in this lineage.
- `model-experiments/04-cpt-sft/docs/reports/2026-08-spectrum-vs-stock-comparison.md`
  — the original Spectrum result and the lineage baseline.
- `model-experiments/01-sft-dpo/sft_dpo/jacgen2/decontam_v2.jac` +
  `.../jacgen/dedup2.jac` — the shingle machinery `scripts/pipeline.jac`
  carries forward.
