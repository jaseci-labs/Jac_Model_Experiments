# 04-cpt-sft — Design Spec

Status: approved, pre-implementation.
Date: 2026-07-20.

## 1. Purpose

Build a full 7-category SFT dataset (`code_gen`, `debug`, `explanation`,
`conversion`, `trajectory`, `documentation`, `migration`) plus a companion
DPO preference-pair dataset, **twice, independently**:

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
| Categories in scope | All 7: `code_gen`, `debug`, `explanation`, `conversion`, `trajectory`, plus `documentation` and `migration` (added 2026-07-20, see `datagen/spec.md` §6-§7) |
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
  seed_pool.jac          # shared seed corpus builder; pins expected_output per seed; frozen+hashed after fresh build
  llm.jac                # by-llm() wrappers, dual-model (Opus + Fable, §4.1) — pinned snapshot IDs, retry/backoff, per-call exception capture, spend cap
  gate.jac               # run_jac_project(files) multi-file gate extension + make_sft_example_v2 + category-aware extract_payload
  gen_code_gen.jac
  gen_debug.jac           # also emits raw_output/debug/buggy_variants.jsonl (interface for gen_dpo + gen_trajectory)
  gen_explanation.jac
  gen_trajectory.jac
  gen_documentation.jac
  gen_migration.jac
  gen_dpo.jac             # see dpo-plan.md
  holdout_v2.jac          # carves per-category eval holdouts from the fresh build (fixed thereafter)
  build_manifest_v2.jac   # excludes holdout ids for BOTH run-tags
  dataset_stats_v2.jac
  decontam_v2.jac         # per-category content extractors feeding jacgen/decontam.jac's shingle machinery
```

Reuse from `jacgen/` is deliberately narrow — audit-verified: only
`writer.jac`'s `run_jac` + `append_jsonl` and `decontam.jac`'s generic
shingle machinery survive contact with the new categories. `writer.jac`'s
`make_sft_example` hardcodes the conversion prompt and old schema, and
`extract_jac`/`dedup.jac` assume `messages[1]` is a single assistant turn
with a jac fence — wrong for trajectories and prose categories. Hence
`gate.jac`'s v2 builders rather than a broader import claim.
`decontam_v2.jac` is **not** a path-only wrapper: `decontam.jac`'s
extractor only understands ```` ```python ```` fences and would silently
pass everything Jac or prose (a clean audit that checked nothing); v2
supplies per-category extractors (jac fence / instruction text / doc chunk)
into the same shingle + overlap machinery.

All new tooling is Jac, matching the existing `jacgen/` convention (see
`model-experiments/01-sft-dpo/sft_dpo/jacgen/README.md`). LLM-backed generation uses Jac's
native `by llm()` function-body delegation, not hand-rolled HTTP calls.
There is no in-repo precedent for `by llm()` at ~25-30k batch calls
(existing jacgen used agentic sessions + a deterministic transpiler), so
`llm.jac`'s contract is explicit: retry with exponential backoff per call,
per-call exception capture routed to `rejected/` with reason (one 429 must
not kill a batch), periodic output flush, a per-generator call-budget
kill-switch, per-batch token-usage logging, and pinned snapshot IDs
recorded into every example's `generator_model_id`. The Anthropic Message
Batches API is the preferred transport for the Opus bulk tier (purpose-built
for this shape, ~half the cost).

### 4.1 Model assignment per generator

Two models, split by nature of the work rather than one model for
everything — token-heavy/bulk generation goes to Opus, precision-critical
generation prone to subtle errors goes to Fable:

| Generator | Model | Why |
|---|---|---|
| `gen_code_gen.jac` | Opus | highest example volume (~4,500, 36% of dataset), one call per example-variant, mostly mechanical reverse-instruction writing off a known-good seed |
| `gen_trajectory.jac` | Opus | highest per-example token volume — one call per example produces the whole 3-6-turn conversation (§5 of `datagen/spec.md`), so calls are few but each is long |
| `gen_debug.jac` | Fable | precision-critical — injected bug must actually reproduce, symptom description must match the real failure, dual-gate (buggy fails, fixed passes) means a sloppy generation is wasted spend either way |
| `gen_explanation.jac` | Fable | no compiler gate exists for this category (§7 below) — generation quality is the *only* defense against a hallucinated, ungrounded answer |
| `gen_dpo.jac` | Fable | preference correctness has to be unambiguous per axis (`dpo-plan.md` §2), especially `auth_security` and `correctness` — a subtly-wrong "chosen" side poisons the pair |
| `gen_documentation.jac` | Fable | prose output with no compiler gate — hallucinated parameter names or invented behavior is exactly the ungated failure mode; only the lexical symbol-existence check catches it |
| `gen_migration.jac` | Opus | token-heavy whole-file rewrites, fully compiler-gated (migrated file must pass, original must fail/warn) — errors get caught mechanically, so the bulk model is safe |

Task-type-level overrides within a category: `code_gen`'s
`error_message_authoring` and `debug`'s `code_critique` are ungated prose
outputs — both run on Fable even where the category default is Opus
(`datagen/spec.md` §1.1, §2.1).

`llm.jac` exposes one `by llm()` wrapper per (task_type, model) pair rather
than a single global model config, so this split is enforced at the wrapper
level, not left to each generator script to remember.

## 5. Run-tag isolation

Every invocation of every `jacgen2` module reads `JAC_RUN_TAG` from the
environment (`fresh` | `post_cptv2`) — `JAC_` prefix matching the existing
jacgen env-var convention, and **no default**: modules hard-fail at entry
if it's unset or not one of the two values (a defaulted tag would silently
write one build's data into the other's tree). Like the existing modules,
jacgen2 uses repo-root-relative paths — run everything from the repo root.
All output paths are tag-scoped:

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
answers, trajectories) is regenerated per run. The pool is **frozen after
the `fresh` build** (its sources ship inside the jaclang package and drift
with `jac` upgrades): sha256 recorded at freeze, verified — not
regenerated — at `post_cptv2` build time. The conversion slice gets the
same snapshot-and-hash treatment (`datagen/workflow.md` §2), since its
upstream lives in gitignored, single-copy, truncatable files.

## 6. Metadata schema

Extends the existing schema from `jac-context-v1.md` verbatim, plus one new
field:

```json
{
  "id": "string",
  "category": "code_gen | debug | explanation | conversion | trajectory | documentation | migration",
  "task_type": "string",
  "complexity": "simple | medium | hard",
  "compiler_pass": true,
  "test_pass": "true | false | null (null = gate class has no behavioral check, see §7)",
  "manually_reviewed": false,
  "generator": "opus-api | fable-api",
  "generator_model_id": "exact dated snapshot ID — pinned, asserted equal across run-tags",
  "gate_class": "behavioral | compile_only | compile_project | prose_lexical",
  "variant_idx": "int — fan-out index within (seed_id, task_type), see datagen/spec.md §0",
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

Non-negotiable, inherited from `jacgen/`'s existing rule: **never gate
generated output on `jac check`** — it over-rejects untyped-but-runnable
Jac. One narrow, audit-driven exception exists (migration input warning
detection, below): `jac check`/`lint_jac` may be used to *detect
deprecation warnings on an input*, never to reject generated code.

### Gate classes

Not every task type can be behaviorally gated — the audit confirmed
`writer.jac`'s `run_jac` is single-file (one snippet in a bare tempdir, no
`jac.toml`, no client build), and several failure modes (auth leaks,
client re-render bugs) are unobservable in a single process. Every example
carries its `gate_class` in metadata (§6) so downstream consumers know what
was actually verified:

| Gate class | What it verifies | Applies to |
|---|---|---|
| `behavioral` | `jac run` exit 0 **plus** output matches the seed's pinned `expected_output` (recorded at seed-pool build time — nothing else populates expected outputs for doc-fence seeds); "≥2 distinct inputs" applies only to seeds that are pure functions | single-file `code_gen`, `debug` (most bug types), `trajectory` final turn, `migration`, `conversion` |
| `compile_project` | `run_jac_project(files)` (the `gate.jac` extension): multi-file program compiles+runs together in a scaffolded project dir | `hard`-tier multi-file examples, `cross_boundary_debug`, fullstack/client-touching seeds |
| `compile_only` | compiles but no behavioral assertion (seed has no runnable entry, or failure mode unobservable in-process) | client-component snippets, `auth_leak` + `stale_client_state` debug rows (plus a Fable critique pass confirming the bug is real), `by llm()`-bearing seeds (never executed in the gate — live API calls inside a gate are nondeterministic spend) |
| `prose_lexical` | mechanical lexical check, no compiler | `explanation` (groundedness: chunk-term overlap, threshold calibrated on pilot), `documentation` (backticked-symbol existence), `error_message_authoring`, `code_critique` |

### Per-category summary

| Category | Gate |
|---|---|
| `code_gen` | `behavioral` (single-file) or `compile_project` (`hard` tier); design-and-prose types per their §1.1 rows. |
| `debug` | Dual-gate: buggy variant must fail (`jac run` non-zero or output ≠ pinned `expected_output`), fixed variant must pass. Reject pairs where the bug doesn't reproduce. `auth_leak`/`stale_client_state` rows: `compile_only` + critique (failure unobservable in-process — flagged `test_pass: null`). |
| `explanation` | `prose_lexical` groundedness. Sample manually reviewed. |
| `conversion` | Unchanged — existing `jacgen/` gate (transpile + `jac run` behavioral check), consumed via snapshot. |
| `trajectory` | Final turn `behavioral`/`compile_project`; intermediate turns deliberately ungated (they're supposed to show a plausible error). |
| `documentation` | `prose_lexical` symbol-existence. Sample manually reviewed. |
| `migration` | Migrated file passes `jac run`. Deprecated original must fail `jac run` **or** produce a deprecation warning under `jac check`/`lint_jac` (the narrow exception — `jac run` emits no deprecation warnings, verified against jaclang 0.16.1). Pairs where the original passes silently with no check-warning are rejected. |

### Dedup + decontamination

All categories run through the **scaled** dedup policy (`datagen/spec.md`
§0.4: exact-hash first, bucketed near-dup, instruction+code jointly,
same-seed exemption — not `dedup.jac`'s raw O(n²) pairwise pass, which has
never run above n≈147) and through `decontam_v2.jac`'s per-category
extractors feeding the 14-token-shingle ≥0.5 overlap machinery, against
the existing eval holdouts (`model-experiments/01-sft-dpo/dataset/eval_holdout/`) **plus the
new `holdout_v2.jac` slices**, before landing in `clean_dataset/`.

## 8. Rollout plan

1. Write `datagen/spec.md` task catalog (this phase, done alongside this spec).
2. Pin Opus/Fable snapshot IDs in `llm.jac`. Build `seed_pool.jac` (with per-seed `expected_output` pinning and the deprecated-pattern inventory — the inventory's size decides `migration`'s real target, `datagen/spec.md` §7), `gate.jac`, and one generator (`gen_code_gen.jac`) end to end.
3. Pilot: ~20-30 examples for `code_gen` only, manually inspect actual `jac run` output — not a claim, an actual run and read. Measure seed fan-out yield and per-example token cost; calibrate the `prose_lexical` thresholds here.
4. Once pilot quality confirmed, build remaining generators (`gen_debug`, `gen_explanation`, `gen_trajectory`, `gen_documentation`, `gen_migration`) + `gen_dpo` (per `dpo-plan.md`). `gen_debug` before `gen_trajectory`/`gen_dpo` — both consume its `buggy_variants.jsonl`.
5. Pilot each new generator the same way before scaling.
6. Full `JAC_RUN_TAG=fresh` generation to the 10,000-15,000 target — or the honest lower number the pilot's fan-out measurement supports (`datagen/spec.md` §0).
7. Snapshot the conversion slice + freeze/hash `seed_pool.jsonl`. Carve holdouts (`holdout_v2.jac`). `dataset_stats_v2.jac` composition report + `decontam_v2.jac` audit on the `fresh` release.
8. Park. `JAC_RUN_TAG=post_cptv2` waits for CPT v2 training to complete (outside this phase's scope).
9. When CPT v2 lands: verify snapshot-ID pins and pool/slice hashes still hold, then re-run step 6 + the stats/decontam audit with `JAC_RUN_TAG=post_cptv2` (holdouts are NOT re-carved — the fresh-build slices score both datasets' training runs).
10. Proceed to `workflow.md`'s SFT comparison.

## 9. Out of scope for this phase

- Running CPT v2 training itself (`model-experiments/03-cpt-only/`'s scope).
- Live agentic trajectory capture (deferred per §3).
- Open-ended code-explanation generation (deferred per §3).
- Actually running the two SFT training jobs and the comparison eval — planned in `workflow.md`, executed as a later phase once both datasets exist.
