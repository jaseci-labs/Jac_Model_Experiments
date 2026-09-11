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

> **Status (2026-09-11): complete, reduced scope.** The design was cut to
> Spectrum SFT only, evaluated on both holdouts. RQ1 and the DPO questions were
> not answered. Result: **70.5% (603/855) on holdout (a), 37.7% (322/855) on
> holdout (b)**, lower than 04, 06 and 07. See
> [`reports/08-final-comparison.md`](reports/08-final-comparison.md).

> **Everything here is Jac.** Pipeline and driver scripts are `.jac`; the
> Spectrum SNR scan and layer selection are the only `.py` files. `.sh`
> wrappers call `jac run <driver>.jac <flags>`.

This tree is also the template for new experiments. The generic, step-by-step
version of the recipe is [`../../docs/PLAYBOOK.md`](../../docs/PLAYBOOK.md).

## Layout

| File | Contents |
|---|---|
| [`../CONTEXT_BRIEF.md`](../CONTEXT_BRIEF.md) | Settled facts: the 7-source merge, the eval-denylist fix, the quality-gradient filter, the DPO-axis change from 07, the graph-native framing warning, the holdout-(b)-reused decision, the failure-mode table. |
| [`spec.md`](spec.md) | Experiment spec as designed: source inventory, the 2 arms × 2 holdouts × 3 stages design, reuse-vs-new against 07, decontamination, pre-registered decision criteria. |
| [`workflow.md`](workflow.md) | The runbook as executed, with the scope cuts marked. |
| [`dataset-structure.md`](dataset-structure.md) | Composition tables for the released SFT/DPO sets. |
| [`reports/corpus-triage-report.md`](reports/corpus-triage-report.md) | The full funnel: 7 sources in, every filter's drop count and reason, the leakage check, the final release sizes. |
| [`reports/08-final-comparison.md`](reports/08-final-comparison.md) | Results, incidents, interpretation. |

## Reading order

1. `reports/08-final-comparison.md` — what happened.
2. `../CONTEXT_BRIEF.md` — the settled facts, especially §1.2 (the
   graph-native framing reversal), §4 (why holdout (b) is reused rather than
   re-carved) and §11 (failure modes).
3. `spec.md` §7 — the pre-registered decision rule.
4. `reports/corpus-triage-report.md` and `dataset-structure.md` — what got built.
5. `workflow.md` — the commands, in order.

## Related, outside this phase

- [`../../docs/history/07-nitin-ds-new-sft/07-final-comparison.md`](../../docs/history/07-nitin-ds-new-sft/07-final-comparison.md)
  — the direct predecessor. Its finding (07's larger-but-weaker pool did not
  beat 06's) motivated 08's quality-gradient filter.
- [`../../docs/history/06-nitin-ds-sft/2026-08-final-comparison.md`](../../docs/history/06-nitin-ds-sft/2026-08-final-comparison.md)
  — its §4 incident log is the origin of several rows in
  `../CONTEXT_BRIEF.md` §11.
- [`../../docs/history/04-cpt-sft/2026-08-spectrum-vs-stock-comparison.md`](../../docs/history/04-cpt-sft/2026-08-spectrum-vs-stock-comparison.md)
  — the original Spectrum result and the lineage baseline.
- `~/repos/jac-data-gen/docs/PY2JAC_QUALITY_GRADIENT.md` — the audit behind
  the test-count filter; read it before changing the cutoff.
- `~/repos/jac-data-gen/evals/function/v1/denylist_ids.txt` — the eval-leakage
  denylist this phase applies for the first time in the lineage.
