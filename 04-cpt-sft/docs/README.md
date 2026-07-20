# 04-cpt-sft — docs index

Phase goal: build two independently-generated SFT (+DPO) datasets — **fresh**
(against the current pre-CPT-v2 base model/tooling) and **post_cptv2**
(generated after CPT v2 training lands) — to measure whether CPT v2 actually
moved the needle on the model's ability to *write* Jac, not just recognize it.

Root cause this phase exists: `model-experiments/03-cpt-only/docs/cpt-2/design.md` found CPT-v1 moved
free-generation vocabulary but left MCQ concept-recognition byte-identical
(18/20 before/after). CPT-v2 is un-run as of this writing. This phase's two
SFT runs are the downstream instrument that will show whether CPT-v2's fix
(corpus dilution + instrument mismatch) actually produces a better *coder*,
independent of whatever CPT-v2's own eval says.

## Layout

| File | Contents |
|---|---|
| [`spec.md`](spec.md) | Umbrella design: what gets built, why, architecture, schema, validation, rollout. Start here. |
| [`workflow.md`](workflow.md) | The three-arm comparison protocol (A: pre-CPT×fresh, C: pre-CPT×post_cptv2, B: CPT×post_cptv2, + incumbent reference). B−C isolates CPT, C−A measures dataset noise. |
| [`dpo-plan.md`](dpo-plan.md) | DPO preference-pair dataset design: idiomatic-vs-non-idiomatic plus additional preference axes. |
| [`datagen/spec.md`](datagen/spec.md) | Full SFT task taxonomy — every category, every task type, seed sourcing, generation prompts, gating. The "make the model actually write Jac" detail. |
| [`datagen/workflow.md`](datagen/workflow.md) | Datagen pipeline mechanics: module graph, run order, mermaid diagrams, run-tag isolation, cost/scale accounting. |

## Reading order

1. `spec.md` — orientation, decisions already locked.
2. `datagen/spec.md` — the actual task catalog (this is the bulk of the content).
3. `datagen/workflow.md` — how the catalog gets executed into files on disk.
4. `dpo-plan.md` — the preference-pair layer on top of the SFT data.
5. `workflow.md` — what happens after both datasets exist: the two SFT training runs and the CPT-effect comparison.

## Related, outside this phase

- `model-experiments/01-sft-dpo/` — the prior conversion-only SFT/DPO probe. `model-experiments/01-sft-dpo/sft_dpo/jacgen/` is the reusable code-gate + dedup/decontam library this phase's generators import.
- `model-experiments/03-cpt-only/docs/cpt-2/design.md` — CPT v2 design (the thing whose effect this phase measures). CPT-v2 training itself is **not** part of this phase.
- `model-experiments/01-sft-dpo/dataset/context/jac-context-v1.md` — origin of the 5-category schema (`code_gen`/`debug`/`explanation`/`conversion`/`trajectory`) this phase completes and extends (this phase adds `documentation` and `migration`, for 7 total). Categories `code_gen`, `debug`, `trajectory` were never generated before this phase; `explanation` was never generated either — this phase's `explanation` is docs-grounded quiz only (narrower than the original open-ended code-explanation idea).
