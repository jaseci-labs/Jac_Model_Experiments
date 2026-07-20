# 04-cpt-sft — Design Spec

Status: approved, pre-implementation.
Date: 2026-07-20.

## 1. Purpose

Build a full 5-category SFT dataset (`code_gen`, `debug`, `explanation`,
`conversion`, `trajectory`) plus a companion DPO preference-pair dataset,
**twice, independently**:

- **`fresh`** — generated now, against the current pre-CPT-v2 project state.
- **`post_cptv2`** — generated later, after CPT v2 training (`model-experiments/03-cpt-only/docs/cpt-2/design.md`)
  actually runs.

Both builds use the identical pipeline, task taxonomy, and prompts (see
`datagen/spec.md`, `datagen/workflow.md`) but are **not** required to produce
identical content — each is a fresh LLM-generation pass. The two resulting
datasets each SFT-train a separate copy of the model; the eval delta between
those two SFT'd models is the signal for whether CPT v2 had a real effect.
Full protocol in `workflow.md`.

This spec is the umbrella. It does not restate the task catalog
(`datagen/spec.md`) or the pipeline mechanics (`datagen/workflow.md`) — it
covers what's shared across both: motivation, architecture, schema,
validation policy, directory layout, and rollout plan.

## 2. Why this exists

Traced via git history + doc dig (2026-07-19):

- The original model-experiments/01-sft-dpo plan (`docs/tasks/README.md`, deleted from git at
  commit `57366c0`, restated in `model-experiments/01-sft-dpo/dataset/context/jac-context-v1.md:324-407`)
  targeted 5 categories at 10,000-15,000 examples: `code_gen`, `debug`,
  `explanation`, `conversion`, `trajectory`.
- Only `conversion` was ever built (~1640 examples, Python→Jac + a graph/OSP
  tier). `code_gen`, `debug`, `explanation`, `trajectory` scaffold directories
  exist on disk and are empty — the project deliberately narrowed to a cheap
  "conversion probe" to pick a base model first (`model-experiments/01-sft-dpo/docs/model-experiments/01-sft-dpo-phase.md`),
  and never returned to the full plan.
- Separately, `model-experiments/03-cpt-only/docs/cpt-2/design.md` found CPT-v1 shifted free-generation
  vocabulary but left MCQ concept-recognition completely unmoved. CPT-v2's
  hypothesis is that the right instrument to see CPT's effect is downstream
  generation quality, not multiple-choice recognition.

Put together: nobody has ever measured what happens when a *properly broad*
SFT set (not just conversion) is trained on top of a CPT'd base vs a non-CPT'd
base. This phase builds that SFT set and the comparison protocol.

## 3. Decisions locked (do not re-litigate without a new brainstorming pass)

| Decision | Choice |
|---|---|
| Categories in scope | All 5: `code_gen`, `debug`, `explanation`, `conversion`, `trajectory` |
| `explanation` scope | Docs-grounded quiz Q&A only (Jac lang docs → question/answer). Not open-ended code-explanation — deferred. |
| `trajectory` scope | LLM-simulated multi-turn (single model plays both user and assistant across a coding task). Not live agent-session capture — deferred. |
| Scale target | 10,000-15,000 examples per dataset build, split per `datagen/spec.md` §weights |
| Seed source for code | jac-mcp `list_examples`/`get_example` + Jac lang doc code-fences via `search_docs`/`get_resource`. Not this repo's own app code (avoids overfitting to studio-app style, keeps canonical language coverage). |
| Generator model | Split by generator — Opus for token-heavy/bulk, Fable for precision/error-prone. See §4.1. Both via API, called from Jac using `by llm()` (jac-by-llm pattern) |
| Number of builds | Two, independent (`fresh`, `post_cptv2`) — not one dataset reused twice |
| Relationship to CPT v2 training | Decoupled. This phase does not train or run CPT v2. `fresh` can and should proceed now; `post_cptv2` waits for CPT v2 to actually land. |
| DPO dataset | Yes, separate plan — see `dpo-plan.md` |

## 4. Architecture

New directory, sibling to the existing `model-experiments/01-sft-dpo/sft_dpo/jacgen/` conversion
pipeline (left untouched):

```
model-experiments/01-sft-dpo/sft_dpo/jacgen2/
  seed_pool.jac          # shared seed corpus builder (all code-bearing categories)
  llm.jac                # shared by-llm() wrapper functions, dual-model (Opus + Fable, see §4.1)
  gen_code_gen.jac
  gen_debug.jac
  gen_explanation.jac
  gen_trajectory.jac
  gen_dpo.jac             # see dpo-plan.md
  build_manifest_v2.jac
  dataset_stats_v2.jac
  decontam_v2.jac         # thin wrapper: reuses jacgen/decontam.jac against jacgen2 output paths
```

`jacgen2` **imports** `jacgen/writer.jac` (the `jac run` compiler+behavioral
gate, `make_sft_example`, `append_jsonl`), `jacgen/dedup.jac`, and
`jacgen/decontam.jac` directly rather than forking them — one gate
implementation, no drift between the conversion pipeline and this one.

All new tooling is Jac, matching the existing `jacgen/` convention (see
`model-experiments/01-sft-dpo/sft_dpo/jacgen/README.md`). LLM-backed generation uses Jac's
native `by llm()` function-body delegation, not hand-rolled HTTP calls.

### 4.1 Model assignment per generator

Two models, split by nature of the work rather than one model for
everything — token-heavy/bulk generation goes to Opus, precision-critical
generation prone to subtle errors goes to Fable:

| Generator | Model | Why |
|---|---|---|
| `gen_code_gen.jac` | Opus | highest example volume (~5,000, 40% of dataset), one call per example, mostly mechanical reverse-instruction writing off a known-good seed |
| `gen_trajectory.jac` | Opus | highest per-example token volume by design (up to 6 turns/calls per example, §5 of `datagen/spec.md`) |
| `gen_debug.jac` | Fable | precision-critical — injected bug must actually reproduce, symptom description must match the real failure, dual-gate (buggy fails, fixed passes) means a sloppy generation is wasted spend either way |
| `gen_explanation.jac` | Fable | no compiler gate exists for this category (§7 below) — generation quality is the *only* defense against a hallucinated, ungrounded answer |
| `gen_dpo.jac` | Fable | preference correctness has to be unambiguous per axis (`dpo-plan.md` §2), especially `auth_security` and `correctness` — a subtly-wrong "chosen" side poisons the pair |

`llm.jac` exposes one `by llm()` wrapper per (task_type, model) pair rather
than a single global model config, so this split is enforced at the wrapper
level, not left to each generator script to remember.

## 5. Run-tag isolation

Every invocation of every `jacgen2` module reads `RUN_TAG` from the
environment (`fresh` | `post_cptv2`). All output paths are tag-scoped:

```
model-experiments/04-cpt-sft/dataset/<RUN_TAG>/raw_output/<category>/
model-experiments/04-cpt-sft/dataset/<RUN_TAG>/clean_dataset/<category>/
model-experiments/04-cpt-sft/dataset/<RUN_TAG>/rejected/<category>/
model-experiments/04-cpt-sft/dataset/<RUN_TAG>/review/
model-experiments/04-cpt-sft/dataset/<RUN_TAG>/logs/{generation,parsing,compiler,test,retry,deduplication}/
model-experiments/04-cpt-sft/dataset/<RUN_TAG>/releases/sft_train.jsonl
model-experiments/04-cpt-sft/dataset/<RUN_TAG>/releases/dpo_train.jsonl
```

(`model-experiments/04-cpt-sft/dataset/` is gitignored, matching `model-experiments/01-sft-dpo/dataset/`'s
policy — same reasoning: large generated artifacts, regenerable from the
`jacgen2` scripts, not source.)

`seed_pool.jsonl` is the one deliberate exception: **not** run-tag-scoped.
The non-LLM part of seeding (which jac-mcp examples and doc chunks get
pulled) stays identical across both builds — see `workflow.md` §confound
mitigation for why. Only the LLM-authored content (instructions, bugs, quiz
answers, trajectories) is regenerated per run.

## 6. Metadata schema

Extends the existing schema from `jac-context-v1.md` verbatim, plus one new
field:

```json
{
  "id": "string",
  "category": "code_gen | debug | explanation | conversion | trajectory",
  "task_type": "string",
  "complexity": "simple | medium | hard",
  "compiler_pass": true,
  "test_pass": true,
  "manually_reviewed": false,
  "generator": "opus-api | fable-api",
  "generation_date": "timestamp",
  "source_prompt_version": "prompt-category-vN",
  "context_bundle_version": "jac-context-vN",
  "validator_version": "validator-vN",
  "dataset_version": "jac-synth-vMAJOR.MINOR.PATCH",
  "run_tag": "fresh | post_cptv2",
  "seed_id": "string | null"
}
```

`task_type` is new relative to `jac-context-v1.md` — required per-example,
values enumerated in `datagen/spec.md`. This is what lets `dataset_stats_v2.jac`
report composition at task-type granularity, not just category granularity,
and is required by `workflow.md`'s comparison protocol (comparing task-type
distributions between `fresh` and `post_cptv2` to separate CPT effect from
generation noise).

`seed_id` traces every generated example back to its `seed_pool.jsonl` entry
(null for categories that don't consume seeds, e.g. some `explanation` items
that are synthesized directly from a doc chunk without a code seed).

## 7. Validation & gating policy

Non-negotiable, inherited from `jacgen/`'s existing rule: **never gate on
`jac check`** — it over-rejects untyped-but-runnable Jac. Always gate on
`jac run` (compiler pass + behavioral pass across distinct test cases).

| Category | Gate |
|---|---|
| `code_gen` | Generated Jac must `jac run` clean and produce output matching the instruction's implied spec across ≥2 distinct inputs where applicable. |
| `debug` | The *buggy* variant must actually fail (`jac run` non-zero, or wrong output vs the seed's known-good output). The *fixed* variant (== original seed, or an LLM-repaired equivalent) must pass. Reject pairs where the "bug" doesn't reproduce. |
| `explanation` | No compiler applicable. Gate is a lexical groundedness check: answer must reference terms/entities present in the source doc chunk (reject hallucinated answers that don't cite anything from the chunk). Sample manually reviewed. |
| `conversion` | Unchanged — existing `jacgen/` gate (transpile + `jac run` behavioral check). |
| `trajectory` | Final turn's code must `jac run` clean. Intermediate turns (the deliberately-broken ones) are not gated — they're supposed to show a plausible error. |

All categories run through `jacgen/dedup.jac` (ROUGE-L near-duplicate guard)
and `jacgen/decontam.jac` (14-token shingle overlap ≥0.5) against the
existing eval holdouts (`model-experiments/01-sft-dpo/dataset/eval_holdout/`) before landing in
`clean_dataset/`.

## 8. Rollout plan

1. Write `datagen/spec.md` task catalog (this phase, done alongside this spec).
2. Build `seed_pool.jac`, `llm.jac`, and one generator (`gen_code_gen.jac`) end to end.
3. Pilot: ~20-30 examples for `code_gen` only, manually inspect actual `jac run` output — not a claim, an actual run and read.
4. Once pilot quality confirmed, build remaining 3 generators (`gen_debug`, `gen_explanation`, `gen_trajectory`) + `gen_dpo` (per `dpo-plan.md`).
5. Pilot each new generator the same way before scaling.
6. Full `RUN_TAG=fresh` generation to 10,000-15,000 target, per `datagen/spec.md` weights.
7. `dataset_stats_v2.jac` composition report + `decontam_v2.jac` audit on the `fresh` release.
8. Park. `RUN_TAG=post_cptv2` waits for CPT v2 training to complete (outside this phase's scope).
9. When CPT v2 lands: re-run steps 6-7 with `RUN_TAG=post_cptv2`.
10. Proceed to `workflow.md`'s two-test SFT comparison.

## 9. Out of scope for this phase

- Running CPT v2 training itself (`model-experiments/03-cpt-only/`'s scope).
- Live agentic trajectory capture (deferred per §3).
- Open-ended code-explanation generation (deferred per §3).
- Actually running the two SFT training jobs and the comparison eval — planned in `workflow.md`, executed as a later phase once both datasets exist.
