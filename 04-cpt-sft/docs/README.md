# 04-cpt-sft — docs index

Phase goal: build two independently-generated SFT (+DPO) datasets — **fresh**
(against the pre-CPT-v2 base model/tooling) and **post_cptv2** (against the
CPT-v2-trained checkpoint) — to measure whether CPT v2 actually moved the
needle on the model's ability to *write* Jac, not just recognize it.

> **Status update (2026-07-20): CPT-v2 training completed and was
> REJECTED** on its own acceptance gates (`model-experiments/03-cpt-only/docs/cpt-2/results.md`
> — 0/3 gates cleared; `docs/cpt-2/analysis.md` — root cause is a structural
> limit of next-token CPT on doc prose, not a bad run; that doc recommends
> skipping further CPT and going straight to SFT/DPO). **This phase runs
> the three-arm test anyway, deliberately, as independent confirmation** —
> the SFT-then-eval instrument this phase builds is genuinely different
> from CPT-v2's own Track A/Track B judge, and a second, differently-shaped
> measurement against the same rejected checkpoint is worth having before
> treating CPT-v2 as closed. `post_cptv2` is **no longer blocked on
> training landing** — the CPT-v2 checkpoint already exists
> (`model-experiments/03-cpt-only/adapters/cpt-v2/`) and can be used for Arms B/C now.

Root cause this phase exists: `model-experiments/03-cpt-only/docs/cpt-2/design.md` found CPT-v1 moved
free-generation vocabulary but left MCQ concept-recognition byte-identical
(18/20 before/after). This phase's SFT runs are the downstream instrument
that shows whether CPT-v2's fix (corpus dilution + instrument mismatch)
actually produces a better *coder*, independent of whatever CPT-v2's own
eval says — and, per the status update above, independent of whatever
CPT-v2's own eval already concluded.

> **Status update (2026-07-22): `fresh` datagen COMPLETE at full production
> scale.** `jacgen2/` (built this session, see
> `docs/superpowers/plans/2026-07-20-jacgen2-datagen.md`) ran every
> generator end to end: `releases/sft_train.jsonl` = **9,608 examples**,
> `releases/dpo_train.jsonl` = **826 preference pairs**, 0% contamination,
> pool re-frozen and verified. Real composition undershoots the spec's
> weight targets for `code_gen`/`debug` (real supply-constrained — only 6 of
> 34 planned `code_gen` task types are implemented, `debug`'s eligible-pairs
> pool is genuinely small) and overshoots for `explanation`/`documentation`/
> `conversion` (their real seed pools were simply larger); `migration` hit a
> true ceiling of 11/11 possible examples. Full detail, every bug found and
> fixed, and every problem hit during the run:
> `docs/reports/2026-07-task18-full-run-report.md`. **`post_cptv2` build
> and the Arms A/B/C training+eval battery are still not started** — next
> steps for this phase.

## Layout

| File | Contents |
|---|---|
| [`spec.md`](spec.md) | Umbrella design: what gets built, why, architecture, schema, validation, rollout. Start here. |
| [`workflow.md`](workflow.md) | The three-arm comparison protocol (A: pre-CPT×fresh, C: pre-CPT×post_cptv2, B: CPT×post_cptv2, + incumbent reference). B−C isolates CPT, C−A measures dataset noise. |
| [`dpo-plan.md`](dpo-plan.md) | DPO preference-pair dataset design: idiomatic-vs-non-idiomatic plus additional preference axes. |
| [`datagen/spec.md`](datagen/spec.md) | Full SFT task taxonomy — every category, every task type, gating. Seed sourcing is now 5 tiers (jac-mcp examples/docs, jaclang's own repo, a Fable-reviewed multi-org public Jac scrape, and this repo's own app code) — §8. The "make the model actually write Jac" detail. |
| [`datagen/workflow.md`](datagen/workflow.md) | Datagen pipeline mechanics: module graph, run order, mermaid diagrams, run-tag isolation, cost/scale accounting. |

## Reading order

1. `spec.md` — orientation, decisions already locked.
2. `datagen/spec.md` — the actual task catalog (this is the bulk of the content).
3. `datagen/workflow.md` — how the catalog gets executed into files on disk.
4. `dpo-plan.md` — the preference-pair layer on top of the SFT data.
5. `workflow.md` — what happens after both datasets exist: the two SFT training runs and the CPT-effect comparison.

## Related, outside this phase

- `model-experiments/01-sft-dpo/` — the prior conversion-only SFT/DPO probe. `model-experiments/01-sft-dpo/sft_dpo/jacgen/` is the reusable code-gate + dedup/decontam library this phase's generators import.
- `model-experiments/03-cpt-only/docs/cpt-2/design.md` + `results.md` + `analysis.md` — CPT v2 design, acceptance verdict (REJECTED), and root-cause analysis (the thing whose effect this phase independently re-measures via a different instrument). CPT-v2 training itself was **not** part of this phase and already completed before this phase's Arm B/C could run.
- `model-experiments/01-sft-dpo/dataset/context/jac-context-v1.md` — origin of the 5-category schema (`code_gen`/`debug`/`explanation`/`conversion`/`trajectory`) this phase completes and extends (this phase adds `documentation` and `migration`, for 7 total). Categories `code_gen`, `debug`, `trajectory` were never generated before this phase; `explanation` was never generated either — this phase's `explanation` is docs-grounded quiz only (narrower than the original open-ended code-explanation idea).
