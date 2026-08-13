# 06-nitin-ds-sft — docs index

Phase goal: take a **new, externally-sourced Jac corpus** (Nitin's
`chess10kp/jac-data-gen`, commit `7c25aff3110f526eec59e0123ffe6c0c152cce91`),
turn it into an SFT + DPO dataset, and run the **same two-arm
stock-vs-Spectrum LoRA probe** that `04-cpt-sft`'s fresh arm already ran —
so that two independent questions get answered off one training battery:

1. **Does Spectrum (SNR / Marchenko-Pastur layer selection) replicate?**
   `04-cpt-sft` found a real, significant Spectrum lift on the fresh arm at
   every stage (+4.9pp SFT, +4.3pp DPO-best, +10.6pp DPO-final —
   `04-cpt-sft/docs/reports/2026-08-spectrum-vs-stock-comparison.md` §1) and
   *no* significant lift on the CPT-v2 arm. That was one dataset. A second,
   independently-sourced dataset on the same base checkpoint is the cheapest
   available test of whether the fresh-arm result was about Spectrum or about
   that particular dataset.
2. **Is this dataset better than the jacgen2-derived one?** Both arms here
   also eval against `04-cpt-sft`'s *unchanged* 855-row code-graded holdout,
   which makes every number directly comparable to the existing fresh-arm
   table cell-for-cell.

> **Honest framing, settled up front (`CONTEXT_BRIEF.md` §1).** The corpus is
> 7,627 files of **flat, Python-transpiled Jac functions** — a docstring plus
> one `def name(params) -> type { ... }`. Sampling found no `node`/`edge`/
> `walker` anywhere. This is the same *shape* as this project's existing
> `python_to_jac_function` task type, **not** graph-native OSP material. Every
> design decision below (single SFT task type, correctness-axis DPO instead of
> the flagship idiomatic axis) follows from that fact. Do not let a later
> write-up describe this as "idiomatic graph-native Jac" — it would overclaim.

> **Status (2026-08-10): docs only.** Corpus triage is running concurrently
> and has not landed. No dataset, no adapters, no eval numbers exist yet.
> Every number in `dataset-structure.md` is marked TBD on purpose.

## Layout

| File | Contents |
|---|---|
| [`../CONTEXT_BRIEF.md`](../CONTEXT_BRIEF.md) | **Read first, in full.** The settled-facts file: base checkpoint, eval instruments, DPO axis, script-reuse plan, decontam plan, directory layout. Everything downstream treats it as given. |
| [`spec.md`](spec.md) | Full experiment spec: source data + commit pin, what the corpus actually is, the 2 arms × 2 holdouts × 3 stages design, reuse-vs-new inventory against `04-cpt-sft`, decontamination requirement, live-monitoring requirement, and the pre-registered acceptance/decision criteria. |
| [`workflow.md`](workflow.md) | The runbook. Exact script names and paths in dependency order: triage → SFT-pair authoring → DPO-pair authoring → SFT ×2 → DPO ×2 → eval ×12 cells → comparison report. |
| [`dataset-structure.md`](dataset-structure.md) | Composition tables for the released SFT/DPO sets, matching `04-cpt-sft/docs/dataset-structure.md`'s format. **Scaffold only** — real counts land when triage does. |
| [`reports/corpus-triage-report.md`](reports/corpus-triage-report.md) | **Pending.** Produced by the concurrent triage agent: compile-check yield over all 7,627 files, within-corpus dedup, decontamination against every existing holdout, and the achievable clean-pool size that sets the holdout carve. |
| `reports/<comparison reports as they land>` | The 2×2×3 matrix write-up, in the style of `04-cpt-sft/docs/reports/2026-08-spectrum-vs-stock-comparison.md`. |

## Reading order

1. `../CONTEXT_BRIEF.md` — the settled facts. Nothing here re-derives them.
2. `spec.md` — what is being measured, against what, and what result counts
   as a win. Read §7 (decision rule) before running anything; it is
   pre-registered so it can't be chosen after seeing numbers.
3. `reports/corpus-triage-report.md` — once it exists. It sets the real
   pool size, and therefore the real holdout size and the real statistical
   power. Several TBDs in `spec.md` and `dataset-structure.md` resolve here.
4. `workflow.md` — the step-by-step execution order.
5. `dataset-structure.md` — what actually got built, after the fact.

## Related, outside this phase

- `model-experiments/04-cpt-sft/docs/reports/2026-08-spectrum-vs-stock-comparison.md`
  — the result being replicated. Its fresh-arm numbers are this phase's
  comparison baseline for research question 2, and its §3 incidents
  (the `mlx_lm.fuse` re-quantization bug, the `mx.load` in-place-truncation
  bug, the DPO wired-memory OOM) are the failure modes to expect.
- `model-experiments/04-cpt-sft/sft_fresh_probe/` — the entire training/eval
  script suite this phase copy-adapts. Battle-tested; not rewritten.
- `model-experiments/04-cpt-sft/docs/spectrum-plan.md` + `spectrum-workflow.md`
  — the original Spectrum design and runbook, including the
  `--verify-layers` trainable-param gate and the §7 `adapter_config.json`
  rewrite that every eval here depends on.
- `model-experiments/04-cpt-sft/docs/dpo-plan.md` — the five-axis DPO design.
  This phase deliberately uses **only** its correctness axis (§2.3) and
  explicitly declines the flagship idiomatic axis; see `spec.md` §3.
- `model-experiments/01-sft-dpo/sft_dpo/jacgen2/decontam_v2.jac` +
  `.../jacgen/decontam.jac` — the 14-token-shingle / ≥0.5-overlap machinery
  this phase reuses rather than rebuilding.
