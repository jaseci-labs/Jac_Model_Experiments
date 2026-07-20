# datagen/spec.md — SFT Task Catalog

Companion to `../spec.md` (architecture/schema/rollout) and `workflow.md`
(pipeline mechanics + mermaid). This file is the actual task taxonomy: every
category, every task type inside it, what it teaches the model, and where its
seed material comes from. This is the "make the model actually able to write
Jac" content — breadth here is the point.

Grounding principle: task types are drawn from the real surface area of the
Jac language + the Jac fullstack framework (OSP core, typed archetypes,
server endpoints, client components, auth, styling, packaging, native
compile) — not just "write a function." A model that only ever sees
standalone functions never learns walkers, graph queries, or the fullstack
wiring that is Jac's actual differentiator over plain Python.

## Category weights (target: 12,500 examples, range 10,000-15,000)

| Category | Weight | Target @ 12,500 |
|---|---|---|
| `code_gen` | 40% | ~5,000 |
| `debug` | 16% | ~2,000 |
| `explanation` | 12% | ~1,500 |
| `conversion` | 12% | ~1,500 |
| `trajectory` | 10% | ~1,250 |
| buffer / hard-to-classify | 10% | ~1,250 |

Buffer absorbs overflow from whichever category's seed pool turns out
richer than expected — `build_manifest_v2.jac` reallocates it at build time
rather than forcing an artificial per-category cap.

---

## 1. `code_gen` — NL instruction → Jac code (40%)

Reverse-instruction generation: take a seed (canonical Jac snippet from
jac-mcp examples or a doc code-fence), have the LLM write the natural-language
task description that snippet solves, and pair them. This avoids the failure
mode of forward-generation (LLM invents both problem and solution and can
drift into non-idiomatic Python-shaped Jac) — the *code* is always a real,
canonical, jac-mcp-sourced or doc-sourced artifact; only the instruction is
synthesized.

| `task_type` | What it teaches | Seed source |
|---|---|---|
| `core_language_basics` | variables, typed literals, control flow, functions, lambdas, string formatting, error handling | `jac-mcp` examples tagged basic; doc-fences from core-syntax pages |
| `node_edge_definition` | defining nodes/edges with typed `has` fields, connecting them, default values, optional references | `jac-node-edge-patterns` doc examples |
| `walker_traversal` | walker entry points, `visit`, `spawn`, `disengage`, moving node-to-node, early stop | `jac-walker-patterns` doc examples |
| `graph_query_patterns` | filtering node/edge subsets by type or field, multi-hop reads, collecting results across a traversal | `jac-node-edge-patterns` + `jac-walker-patterns` combined examples |
| `llm_delegated_function` | `by llm()` function-body delegation, structured output shape, tool-use wiring | `jac-by-llm` doc examples |
| `typed_signatures` | generics, unions, optionals, correct inference vs explicit annotation | `jac-types` doc examples |
| `impl_file_split` | separating archetype declarations from method bodies into `.impl.jac` | `jac-impl-files` doc examples |
| `sv_endpoint_authoring` | `def:pub`/`def:priv` server functions, typed responses, basic CRUD shape | `jac-sv-endpoints` doc examples |
| `sv_persistence_queries` | modeling relationships + multi-step graph queries inside endpoint bodies | `jac-sv-persistence` doc examples |
| `sv_auth_gating` | deciding which endpoints are public vs authenticated, isolating data per user | `jac-sv-auth` doc examples |
| `sv_multi_user_sharing` | cross-user permission grants, per-user roles, scanning every user's root correctly | `jac-sv-multi-user` doc examples |
| `cl_component_authoring` | `.cl.jac` component shape, reactive state, mount effects, event handlers | `jac-cl-components` doc examples |
| `cl_routing` | route definitions, redirects, navigation from handlers | `jac-cl-routing` doc examples |
| `cl_auth_ui` | signup/login/logout UI, guarding pages behind login | `jac-cl-auth` doc examples |
| `cl_styling` | Tailwind conditional classes, `cn()` (clsx+tailwind-merge), semantic color tokens, `.style.css` annex | `jac-cl-styling` doc examples |
| `cl_organization` | file layout, component reuse, hook pattern, domain-meaningful naming for growing apps | `jac-cl-organization` doc examples |
| `fullstack_wiring` | `main.jac` entry, server-endpoint registration, client mount, `sv import` rules | `jac-fullstack-patterns` doc examples |
| `npm_integration` | `jac.toml` package deps, importing npm packages + React hooks (`useRef`/`useCallback`/`useMemo`) in `.cl.jac` | `jac-npm-packages` doc examples |
| `native_compile_subset` | `.na.jac` compute subset, native FFI import of a precompiled C library | `jac-native` doc examples — smaller share, advanced-only |
| `packaging_cli` | `jac.toml` metadata, console-script entry points, `jac bundle` | `jac-packaging` doc examples |
| `shadcn_component_composition` | `jac add --shadcn`, import paths, composition, `jac retheme` | `jac-shadcn-components` doc examples |
| `scaffold_bootstrap` | `jac create --use <template>`, fixing deprecated syntax the scaffold ships with | `jac-scaffold` doc examples |

Per-task-type target: roughly even split of the 40% budget across the 21
types above, weighted 1.5x for `node_edge_definition`, `walker_traversal`,
`graph_query_patterns`, `sv_endpoint_authoring`, `cl_component_authoring` —
these five are Jac's actual differentiators and the ones a Python-trained
base model is most likely to get wrong by defaulting to Python idiom.

Complexity tiers per task_type: `simple` (one archetype/one function, single
concept), `medium` (2-3 archetypes or a short walker chain), `hard`
(multi-file: e.g. a `.sv.jac` endpoint + its `.cl.jac` consumer together, or a
walker with multiple typed edge filters and early-exit logic). Target
distribution simple:medium:hard = 45:35:20 per task_type.

---

## 2. `debug` — broken code + symptom → fixed code (16%)

Cross a **bug taxonomy** against a subset of the `code_gen` domain areas.
Not all 21 domains get debug coverage — niche ones (`native_compile_subset`,
`packaging_cli`, `scaffold_bootstrap`) are excluded; bug injection there
would teach edge cases nobody actually hits during normal Jac writing.

### Bug taxonomy

| Bug type | Description |
|---|---|
| `syntax_typo` | missing semicolon, wrong keyword, mismatched brace — compiler-level failure |
| `type_mismatch` | wrong type annotation, causes a type-check failure or silent wrong behavior |
| `wrong_edge_direction` | walker/query filters the wrong edge direction or wrong edge type |
| `walker_missing_disengage` | walker never terminates / revisits — needs `disengage` or a visited-guard |
| `off_by_one_logic` | boundary error in a loop, slice, or index |
| `missing_impl_binding` | `.impl.jac` method signature doesn't match its declaration, or binding is missing entirely |
| `auth_leak` | endpoint missing `def:priv` or missing a per-user filter — returns another user's data |
| `unhandled_exception` | missing error handling at an actual trust/system boundary (not spurious — only where it can really happen) |
| `npm_import_missing` | wrong import path or missing `jac.toml` dependency entry |
| `byllm_schema_mismatch` | `by llm()` function's declared return type doesn't match what the prompt actually produces |

### Domain × bug coverage

| Domain (from §1) | Applicable bug types |
|---|---|
| `node_edge_definition` | `syntax_typo`, `type_mismatch` |
| `walker_traversal` | `wrong_edge_direction`, `walker_missing_disengage`, `off_by_one_logic` |
| `graph_query_patterns` | `wrong_edge_direction`, `off_by_one_logic` |
| `llm_delegated_function` | `byllm_schema_mismatch`, `type_mismatch` |
| `typed_signatures` | `type_mismatch` |
| `impl_file_split` | `missing_impl_binding` |
| `sv_endpoint_authoring` | `syntax_typo`, `unhandled_exception` |
| `sv_persistence_queries` | `off_by_one_logic`, `wrong_edge_direction` |
| `sv_auth_gating` | `auth_leak` |
| `sv_multi_user_sharing` | `auth_leak` |
| `cl_component_authoring` | `syntax_typo`, `unhandled_exception` |
| `cl_routing` | `syntax_typo` |
| `cl_auth_ui` | `auth_leak` |
| `npm_integration` | `npm_import_missing` |

Generation method: take a clean, jac-run-passing seed from the `code_gen`
seed pool for the matching domain, have the LLM inject exactly one bug from
the applicable list, and require it to also produce a one-line symptom
description (compiler error text, or "returns X instead of Y", or "user A
can see user B's data") — the symptom is part of the training example, since
real debugging starts from a symptom, not a bug label. Fixed target = the
original clean seed, or an LLM-authored equivalent if the injected bug
required restructuring beyond a single-line fix.

`auth_leak` entries are dual-purpose: security-relevant behavior the model
should learn to flag and fix, sourced from the same domains
`jac-sv-auth`/`jac-sv-multi-user` cover.

---

## 3. `explanation` — docs-grounded quiz Q&A (12%)

Scope: **grounded in Jac language/framework documentation only** — not
open-ended "explain this code" (that variant is explicitly deferred, see
`../spec.md` §3). Every question must be answerable from a specific doc
chunk, and the chunk is stored alongside the answer for traceability and
decontamination.

Source material: `jac-mcp` `search_docs`/`get_resource`, specifically
`jac://guide/pitfalls`, `jac://guide/patterns`, and the doc pages backing
every skill area in §1's seed-source column.

| `task_type` | Format | Example shape |
|---|---|---|
| `concept_recall` | "What does `disengage` do inside a walker?" | direct factual recall, single doc chunk |
| `pitfall_identification` | "Why does this pattern fail?" (given a snippet drawn from `jac://guide/pitfalls`) | explain the specific gotcha, cite the fix |
| `pattern_application` | "Given this scenario, which pattern applies?" (multiple candidate approaches, pick + justify) | tests judgment, not just recall |
| `compare_contrast` | "`jac check` vs `jac run` — when does each over/under-reject?" or "node vs walker — which owns state?" or "`def:pub` vs `def:priv`" | forces the model to hold two concepts apart |
| `doc_example_reading` | "Given this canonical doc example, what does it output / what would break if line N were removed?" | grounds reasoning in a real example, not an abstract question |

Chunking: doc pages split at heading boundaries, chunks capped ~500 tokens
so grounding is checkable (see `../spec.md` §7 groundedness gate). One
question minimum per chunk, up to 3 for chunks covering multiple concepts.

---

## 4. `conversion` — Python → Jac (12%)

**Reused as-is** from the existing `model-experiments/01-sft-dpo/sft_dpo/jacgen/` pipeline —
`mine.jac` (HF corpus mining), `scale_conversion.jac` (transpile + jac-run
gate), `idiomatic_batch*.jac` (hand-authored idiomatic core), `graph_seeds.jac`
(node/edge/walker conversion tier). No new generation code needed here; this
phase's contribution is folding its existing output into the unified
`sft_train.jsonl` via `build_manifest_v2.jac` with `task_type` back-filled as
`python_to_jac_function` (the transpile-tier majority) or
`python_to_jac_graph` (the graph tier).

If the existing ~1,640-example pool falls short of the 12% target
(~1,500 at 12,500 total — currently already exceeds this, no action needed at
current dataset size), re-run `scale_conversion.jac` with a higher `limit` to
mine additional examples from the same Vezora corpus.

---

## 5. `trajectory` — LLM-simulated multi-turn sessions (10%)

Single Fable call plays both sides of a coding conversation, 3-6 turns,
seeded by a `code_gen` task (reuses the same seed pool — a trajectory is a
`code_gen` task's instruction, unrolled into a realistic back-and-forth
instead of a single-shot answer).

| `task_type` | Turn shape |
|---|---|
| `build_from_scratch` | user describes a feature in plain language → assistant builds it incrementally, user asks 1-2 follow-up refinements |
| `debug_session` | user pastes a real error (drawn from a `debug` category example) → assistant diagnoses, may propose a wrong fix first, user reports it didn't work, assistant corrects |
| `refactor_session` | user provides existing working code, asks to restructure (e.g. split into `.impl.jac`, extract a walker from inline logic) |
| `add_feature_to_existing` | user provides an existing snippet, asks to extend it (add a field, add an endpoint, add a route) |
| `code_review_session` | assistant reviews user-provided Jac code unprompted-style, flags 1-2 real issues, user asks to fix one, assistant revises |

Gate: the **final** assistant turn's code must `jac run` clean. Intermediate
turns showing a wrong attempt or an error are intentionally not gated — a
trajectory with zero friction teaches the model nothing about recovering from
its own mistakes, which is the entire point of this category existing.

Storage shape: `messages: [{role, content}, ...]` — standard chat-format
list, compatible with the existing `mlx` split builder pattern in
`model-experiments/01-sft-dpo/sft_dpo/jacgen/build_splits.jac`.

---

## 6. Seed pool construction

Single shared builder (`seed_pool.jac`, see `workflow.md` for its place in
the module graph) feeding `code_gen`, `debug`, `explanation`, and
`trajectory`:

1. Enumerate every `jac-mcp` example via `list_examples`, fetch each with
   `get_example`.
2. Enumerate doc pages behind every skill area listed in §1's seed-source
   column via `search_docs`, extract fenced code blocks via `get_resource`.
3. Tag every seed with its origin (`jac-mcp:<example_id>` or
   `doc:<page_id>#<chunk_idx>`), the domain/task_type it plausibly belongs
   to (inferred from which skill's docs it came from), and a stable `seed_id`.
4. Dedup seeds against each other (exact-match + `dedup.jac` near-duplicate)
   before they ever reach a generator — no point spending Fable calls on the
   same snippet twice.

Output: `seed_pool.jsonl`, not run-tag-scoped (see `../spec.md` §5 — kept
identical across `fresh` and `post_cptv2` builds so only LLM-authored content
varies between the two datasets).
