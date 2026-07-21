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

> Revision note (2026-07-20 audit): a jac-mcp coverage audit against the
> full doc surface (`jac://docs/cheatsheet`, `jac://docs/osp`,
> `jac://guide/pitfalls`, `jac://docs/diagnostics`, …) added six task
> types, five bug types, and folded ~20 named language features into
> existing rows' seed scopes. A feasibility audit added §0 (seed supply and
> fan-out — read it first, it constrains everything below).
>
> Revision note (2026-07-20, seed pool expansion): the jac-mcp-only supply
> (~300-600 seeds) forced a ~25-40x fan-out ratio — thin enough that a
> follow-up decision expanded seed sourcing to 5 tiers (§8). This lowers
> the required fan-out ratio; §0.1 below reflects the expanded estimate,
> but the fan-out *mechanism* itself (§0.2) stays in place regardless
> of supply — it's cheap insurance, not purely a shortage-driven hack.

## 0. Seed supply and fan-out (read first)

### 0.1 Measured / estimated supply, 5 tiers (§8 has full detail per tier)

| Tier | Source | Estimated usable seeds |
|---|---|---|
| 1 | jac-mcp `list_examples`/`get_example` | ~9 apps, low count but high quality |
| 2 | jac-mcp doc code-fences | ~300-600 (measured, most already counted here) |
| 3 | jaclang repo's own test fixtures + stdlib source (new) | unmeasured, pending pilot — likely low hundreds given a compiler test suite's typical size |
| 4 | Multi-org public Jac GitHub scrape, Fable-reviewed (new) | unmeasured, pending pilot — CPT's own 17-org code corpus pull landed ~992K tokens before its own (different) filters; expect this tier to dwarf tiers 1-3 once scraped, net of the tier-4 rejection rate |
| 5 | This repo's own app code — `jms/`, `model-experiments/*/sft_dpo/jacgen/*.jac`, studio `.sv.jac`/`.cl.jac` (new, reverses a prior exclusion) | low hundreds, real production-shaped Jac |

Net effect: total supply should land well above the old ~300-600 ceiling,
plausibly into four figures once tiers 3-5 are actually pulled and
counted — but **do not assume** the number until the pilot measures it
per tier. Tier 4 in particular has an unknown rejection rate (§8.4) that
only a real run will reveal.

### 0.2 Fan-out mechanism (unchanged in kind, lower ratio needed now)

Whatever the real supply turns out to be, the fan-out mechanism stays —
even a four-figure seed pool benefits from more than 1 example per seed,
and the machinery costs nothing extra to keep:

1. **Example key is `(seed_id, task_type, variant_idx)`** — everywhere:
   output filenames, resumability skip-checks, dedup bookkeeping. (A key of
   `seed_id` alone caps fan-out at one example per seed and silently
   converges the whole build to pool size.)
2. **k instructions per seed** — for reverse-instruction task types, each
   seed yields several distinct instructions (different personas, different
   framings, different complexity tiers), each a separate `variant_idx`.
3. **Code-mutating variants** — beyond instruction variety, the generator
   produces behavior-changing variants of the seed (parameter changes,
   extension by one feature, composition of two seeds into one program),
   each re-gated by `jac run` before use. This is what makes the *code*
   side differ, not just the prompt side.
4. **Dedup policy** (amends the naive per-code-body rule): exact-hash first
   pass, then bucketed near-dup (shingle/minhash buckets, ROUGE-L only
   within a bucket — the O(n²) pairwise LCS in `dedup.jac` has never run
   above n≈147 and will not survive 12,500), computed over
   **instruction+code jointly**, with a same-`seed_id` exemption for
   unmutated-code variants (otherwise every k-instruction sibling of a seed
   scores 1.0 on the code side and dies).

If, after the pilot, measured usable-seed count × achievable fan-out cannot
reach the target, the honest move is cutting the target and saying so in
the release notes — not padding with near-duplicates. With 5 tiers this is
a much less likely outcome than it was at jac-mcp-only supply, but the
rule doesn't change: measure before committing to scale.

## Category weights (target: 12,500 examples, range 10,000-15,000)

Seven categories (the original five plus `documentation` and `migration`,
added 2026-07-20):

| Category | Weight | Target @ 12,500 |
|---|---|---|
| `code_gen` | 36% | ~4,500 |
| `debug` | 16% | ~2,000 |
| `explanation` | 10% | ~1,250 |
| `conversion` | 10% | ~1,250 |
| `trajectory` | 10% | ~1,250 |
| `documentation` | 6% | ~750 |
| `migration` | 4% | ~500 — contingent, see §7's inventory caveat |
| buffer / hard-to-classify | 8% | ~1,000 |

Buffer absorbs overflow from whichever category's seed pool turns out
richer than expected — `build_manifest_v2.jac` reallocates it at build time
rather than forcing an artificial per-category cap. It also absorbs
`migration`'s shortfall if the deprecated-pattern inventory can't support
500 examples (§7).

Generator model per category (full rationale in `../spec.md` §4.1):
`code_gen`, `trajectory`, and `migration` → Opus (bulk/token-heavy or
compiler-gated-mechanical). `debug`, `explanation`, `documentation`, and
the DPO layer (`dpo-plan.md`) → Fable (precision-critical, error-prone, or
ungated-prose output). `conversion` uses no LLM — reused deterministic
transpile pipeline. Task-type-level overrides exist inside `code_gen` and
`debug` for ungated-prose outputs (see §1.1 and §2.1).

---

## 1. `code_gen` — NL instruction → Jac code (36%)

Reverse-instruction generation: take a seed (canonical Jac snippet from
jac-mcp examples or a doc code-fence), have the LLM write the natural-language
task description that snippet solves, and pair them. This avoids the failure
mode of forward-generation (LLM invents both problem and solution and can
drift into non-idiomatic Python-shaped Jac) — the *code* is always a real,
canonical, jac-mcp-sourced or doc-sourced artifact (or a re-gated mutation
of one, §0.2); only the instruction is synthesized.

| `task_type` | What it teaches | Seed source |
|---|---|---|
| `core_language_basics` | variables, typed literals, control flow, functions, lambdas, string formatting, error handling; **enums** (auto-valued, typed-base, `sem` on members), **`glob` + `global` statement**, pipe operators `\|>`/`<\|`, decorators, walrus | doc-fences from `jac://docs/cheatsheet` + `jac://docs/foundation` core-syntax sections (no "tagged basic" example set exists — the fence corpus is the source) |
| `jac_control_flow` | the Jac-specific control forms a Python-trained model won't guess: `switch` with fall-through, `match`, `for i = 0 to i < 10 by i += 2`, `while...else`, `skip` | `jac://docs/cheatsheet` Control Flow/Match/Switch; patterns §10 |
| `object_modeling` | plain `obj` (non-graph) modeling: obj-vs-class choice, archetype inheritance (`node Employee(Person)`, obj inheritance), `override`, `static def`/`static has`, generic archetypes (`obj Result[T, E]`), decl-only archetypes | `jac://docs/cheatsheet` Objects/Has; pitfalls §4-6, §10-11 |
| `node_edge_definition` | defining nodes/edges with typed `has` fields, connecting them, default values, optional references, **`postinit` / `has x: T by postinit`** and the field-ordering rule (E2004) | `jac-node-edge-patterns` + `jac-has-fields` doc examples |
| `node_ability_authoring` | node-side/edge-side abilities: `can ... with WalkerType entry/exit`, the `visitor` reference, walker-type-union abilities, anonymous abilities — entry points live on both sides, and self/here/visitor confusion is the docs' "#1 walker bug" | `jac://docs/cheatsheet` Node & Edge Abilities; `jac://docs/osp` |
| `walker_traversal` | walker entry points, `visit`, `spawn`, `disengage`, moving node-to-node, early stop; **visit variants** (`visit :0: [-->]` ordering, `visit ... else {}`, `visit here`), **typed context blocks** `->Dog{...}`, spawn-origin semantics (spawn runs the *origin node's* entry — the documented "classic off-by-one") | `jac-walker-patterns` doc examples + `jac://docs/cheatsheet` Visit Statement Variants |
| `graph_query_patterns` | filtering node/edge subsets by type or field, multi-hop reads, collecting results across a traversal; **graph deletion** (`del node`, `a del --> b`), assign comprehensions `[...](=field=val)`, null-safe access `?.`/`?[]` in query chains | `jac-node-edge-patterns` + `jac-walker-patterns` + `jac://docs/cheatsheet` Connection Operators |
| `llm_delegated_function` | `by llm()` function-body delegation, structured output shape, tool-use wiring, **`sem` statements** on fields/params/returns (central to MTP), enum-constrained LLM output | `jac-by-llm` + `jac://docs/byllm` doc examples |
| `byllm_schema_design` | design the structured-output schema itself for a `by llm()` call given a described extraction/classification need — deeper than `llm_delegated_function`, which only *uses* a given schema | `jac-by-llm` + `jac://docs/byllm` doc examples |
| `typed_signatures` | generics, unions, optionals, correct inference vs explicit annotation; type aliases, `as` casts, gradual-`any` flow rules | `jac-types` + `jac://docs/cheatsheet` Type Aliases |
| `impl_file_split` | separating archetype declarations from method bodies into `.impl.jac`; **properties** (`has` accessor blocks — getter/setter/deleter, impl-block accessors) | `jac-impl-files` doc examples |
| `concurrency_async` | `flow`/`wait` concurrent tasks (Jac-unique keywords); `async walker`, `async can`, `async def:pub` endpoints and the unawaited-coroutine failure | `jac://docs/cheatsheet` Flow/Wait + Async Walkers; `jac-sv-endpoints` async pitfall |
| `python_interop` | inline Python `::py::` blocks (including in enums), bidirectional py↔jac interop — calling Python from Jac and vice versa (distinct from `conversion`, which *translates*) | `jac://docs/python-integration`; `jac://docs/cheatsheet` Inline Python |
| `sv_endpoint_authoring` | `def:pub`/`def:priv` server functions, typed responses, basic CRUD shape; **`@restspec`** (custom method/path, webhook/websocket protocols, broadcast); the plain-`def`-auto-registers rule and `_` prefix to keep helpers off the API | `jac-sv-endpoints` + `jac://docs/jac-scale` protocol table |
| `walker_endpoint_authoring` | walkers as REST endpoints: `walker:pub`/`walker:priv`, `has` fields = request body, `report` = response, the walker-vs-`def:pub` decision | pitfalls §37-39; `jac://docs/tutorial-fullstack-backend` comparison table |
| `sv_persistence_queries` | modeling relationships + multi-step graph queries inside endpoint bodies | `jac-sv-persistence` doc examples |
| `sv_auth_gating` | deciding which endpoints are public vs authenticated, isolating data per user | `jac-sv-auth` doc examples |
| `sv_multi_user_sharing` | cross-user permission grants (`grant`/`revoke`/`allroots` + access levels), per-user roles, scanning every user's root correctly | `jac-sv-multi-user` + `jac://docs/osp` builtins table |
| `cl_component_authoring` | `.cl.jac` component shape, reactive state, mount effects, event handlers; advanced JSX (statement slots `{if..skip;}`, dynamic tags `<@expr>`, `{**props}` spread, `unsafe_html`, `JacForm`/`JacSchema`, `glob:pub` context, `new(Class, args)` JS-constructor interop) | `jac-cl-components` + `jac://docs/jac-client` JSX reference |
| `cl_routing` | route definitions, redirects, navigation from handlers | `jac-cl-routing` doc examples |
| `cl_auth_ui` | signup/login/logout UI, guarding pages behind login | `jac-cl-auth` doc examples |
| `cl_styling` | Tailwind conditional classes, `cn()` (clsx+tailwind-merge), semantic color tokens, `.style.css` annex | `jac-cl-styling` doc examples |
| `cl_organization` | file layout, component reuse, hook pattern, domain-meaningful naming for growing apps | `jac-cl-organization` doc examples |
| `fullstack_wiring` | `main.jac` entry, server-endpoint registration, client mount, `sv import` rules; codespace section headers (`to cl:`/`to sv:`, `na {}` blocks, single-statement prefix forms) | `jac-fullstack-patterns` + `jac://docs/jac-client` section headers |
| `npm_integration` | `jac.toml` package deps, importing npm packages + React hooks (`useRef`/`useCallback`/`useMemo`) in `.cl.jac` | `jac-npm-packages` doc examples |
| `native_compile_subset` | `.na.jac` compute subset, native FFI import of a precompiled C library | `jac-native` doc examples — smaller share, advanced-only |
| `packaging_cli` | `jac.toml` metadata, console-script entry points, `jac bundle` | `jac-packaging` doc examples |
| `shadcn_component_composition` | `jac add --shadcn`, import paths, composition, `jac retheme` | `jac-shadcn-components` doc examples |
| `scaffold_bootstrap` | `jac create --use <template>`, fixing deprecated syntax the scaffold ships with | `jac-scaffold` doc examples |
| `test_authoring` | given a working walker/function, write Jac-native tests for it: `test "name" {}` blocks (a `.test.jac` annex where idiomatic), walker-spawn + `.reports` asserts, runnable via `jac test` — not bare Python-style asserts | `jac://docs/testing` + cheatsheet Test Blocks; gate = authored tests pass against the seed and fail against a **mechanically mutated variant** (operator flip / constant perturbation / edge-direction flip, produced inside `gen_code_gen` — no dependency on `gen_debug`) |
| `perf_optimization` | given a correct-but-wasteful walker/query (redundant traversal, missing early-exit, repeated filtering), rewrite it tighter with behavior held constant | walker/query seeds, deliberately de-optimized by the generator; gate = both versions produce identical output, optimized version passes `jac run` |

### 1.1 Design-and-prose task types (forward generation, not reverse-instruction)

Three `code_gen` task types break the reverse-instruction pattern because
their *output* is not a seed-derived snippet:

| `task_type` | What it teaches | Method + gate |
|---|---|---|
| `schema_design` | plain-language app idea → choose and define the node/edge/walker schema from scratch. Distinct from `node_edge_definition`, which teaches syntax *given* a known shape — this teaches choosing the shape | forward generation seeded by the app-idea prompt bank (§8.5); output gated by `jac run` (schema must compile + a smoke walker must traverse it) |
| `syntax_migration` | deprecated Jac syntax → current syntax, single-snippet scope (broader whole-file scope lives in the `migration` category, §7) | seeds = the deprecated-pattern inventory (§8.4); gate per §7's warning-detection rule |
| `error_message_authoring` | given a failure scenario (bad code + context), write the diagnostic message a good compiler/runtime would emit — teaches the model to *explain* failures precisely, the inverse of `pitfall_identification` (§3) which only recalls documented pitfalls | seeds = `jac://docs/diagnostics` (the structured E/W error-code scheme — E1001/E1030/W1104 etc.) paired with code that triggers each code; prose output, **no compiler gate** — runs on Fable (`../spec.md` §4.1 override); lexical check: message must reference the actual failing symbol/line |

Per-task-type target: roughly even split of the 36% budget across the 34
types above (31 in the main table + 3 design-and-prose), weighted 1.5x for
`node_edge_definition`, `node_ability_authoring`, `walker_traversal`,
`graph_query_patterns`, `sv_endpoint_authoring`, `cl_component_authoring` —
these are Jac's actual differentiators and the ones a Python-trained base
model is most likely to get wrong by defaulting to Python idiom.

Complexity tiers per task_type: `simple` (one archetype/one function, single
concept), `medium` (2-3 archetypes or a short walker chain), `hard`
(multi-file: e.g. a `.sv.jac` endpoint + its `.cl.jac` consumer together, or a
walker with multiple typed edge filters and early-exit logic). Target
distribution simple:medium:hard = 45:35:20 per task_type. Gate class varies
by tier — `hard` multi-file examples need the project-level gate
(`run_jac_project`, `../spec.md` §7), not the single-file one.

---

## 2. `debug` — broken code + symptom → fixed code (16%)

Cross a **bug taxonomy** against a subset of the `code_gen` domain areas.

### Bug taxonomy

| Bug type | Description |
|---|---|
| `syntax_typo` | missing semicolon, wrong keyword, mismatched brace — compiler-level failure |
| `type_mismatch` | wrong type annotation, causes a type-check failure or silent wrong behavior |
| `wrong_edge_direction` | walker/query filters the wrong edge direction or wrong edge type |
| `walker_missing_disengage` | walker never terminates / revisits — needs `disengage` or a visited-guard |
| `here_visitor_self_confusion` | node ability uses `self` where `visitor` is meant (or walker uses `here` wrongly) — the docs' #1 walker bug, distinct from edge-direction errors |
| `spawn_origin_confusion` | walker spawned on the wrong node so the origin node's entry ability fires unexpectedly (or doesn't) — the documented spawn "classic off-by-one" |
| `off_by_one_logic` | boundary error in a loop, slice, or index |
| `missing_impl_binding` | `.impl.jac` method signature doesn't match its declaration, or binding is missing entirely |
| `auth_leak` | endpoint missing `def:priv` (or a plain `def` left auto-registered on the API), or missing a per-user filter — returns another user's data |
| `unhandled_exception` | missing error handling at an actual trust/system boundary (not spurious — only where it can really happen) |
| `missing_async_await` | endpoint body uses `await` without `async def:pub`, or an async walker/can is called unawaited — coroutine never runs |
| `wrong_import_form` | `import` where `include` is needed, missing `import type from` on a circular import, bad alias/relative form |
| `stale_client_state` | in-place `append`/`pop`/dict-mutation on reactive client state → no re-render (pitfalls §28); or `result.reports[0]` shape misuse; or missing `sv import` |
| `npm_import_missing` | wrong import path or missing `jac.toml` dependency entry |
| `byllm_schema_mismatch` | `by llm()` function's declared return type doesn't match what the prompt actually produces |

### Domain × bug coverage

| Domain (from §1) | Applicable bug types |
|---|---|
| `node_edge_definition` | `syntax_typo`, `type_mismatch` |
| `node_ability_authoring` | `here_visitor_self_confusion`, `syntax_typo` |
| `walker_traversal` | `wrong_edge_direction`, `walker_missing_disengage`, `spawn_origin_confusion`, `off_by_one_logic` |
| `graph_query_patterns` | `wrong_edge_direction`, `off_by_one_logic` |
| `llm_delegated_function` | `byllm_schema_mismatch`, `type_mismatch` |
| `typed_signatures` | `type_mismatch` |
| `impl_file_split` | `missing_impl_binding` |
| `object_modeling` | `type_mismatch`, `syntax_typo` |
| `jac_control_flow` | `off_by_one_logic`, `syntax_typo` |
| `concurrency_async` | `missing_async_await` |
| `sv_endpoint_authoring` | `syntax_typo`, `unhandled_exception`, `missing_async_await` |
| `walker_endpoint_authoring` | `auth_leak`, `syntax_typo` |
| `sv_persistence_queries` | `off_by_one_logic`, `wrong_edge_direction` |
| `sv_auth_gating` | `auth_leak` |
| `sv_multi_user_sharing` | `auth_leak` |
| `cl_component_authoring` | `syntax_typo`, `unhandled_exception`, `stale_client_state` |
| `cl_routing` | `syntax_typo` |
| `cl_auth_ui` | `auth_leak` |
| `fullstack_wiring` | `wrong_import_form`, `stale_client_state` |
| `npm_integration` | `npm_import_missing` |
| `core_language_basics` | `syntax_typo`, `off_by_one_logic`, `wrong_import_form` |

**Explicitly excluded domains** (bug injection would teach edge cases nobody
hits during normal Jac writing, or the domain's output isn't code):
`native_compile_subset`, `packaging_cli`, `scaffold_bootstrap`,
`cl_styling`, `cl_organization`, `shadcn_component_composition`,
`python_interop`, `test_authoring`, `perf_optimization`,
`byllm_schema_design`, `schema_design`, `syntax_migration`,
`error_message_authoring`. Every §1 domain is now either in the coverage
table or on this exclusion list — nothing unaccounted.

Generation method: take a clean, jac-run-passing seed from the `code_gen`
seed pool for the matching domain, have the LLM inject exactly one bug from
the applicable list, and require it to also produce a one-line symptom
description (compiler error text, or "returns X instead of Y", or "user A
can see user B's data") — the symptom is part of the training example, since
real debugging starts from a symptom, not a bug label. Fixed target = the
original clean seed, or an LLM-authored equivalent if the injected bug
required restructuring beyond a single-line fix.

**Gate class caveat**: the "buggy variant must fail" dual-gate only works
where the failure is observable under the gate. `auth_leak` symptoms
("user A sees user B's data") and `stale_client_state` (browser
re-render) are **not observable in a single-process `jac run`** — those
rows use the compile-plus-critique gate class instead (`../spec.md` §7 gate
matrix): buggy and fixed variants must both compile, and a Fable critique
pass must confirm the injected bug is real and the fix addresses it. This
is weaker than behavioral gating and is flagged as such in metadata
(`test_pass: null` rather than `true`).

**Persisted intermediate**: `gen_debug` writes every accepted injected-bug
variant (buggy code + `bug_type` + `seed_id`, pre-pairing) to
`raw_output/debug/buggy_variants.jsonl`. This file is a declared interface:
`gen_dpo` (correctness/auth/typing axes, `dpo-plan.md` §2.3-2.5) and
`gen_trajectory` (`debug_session` task type, §5) consume it — which makes
`gen_debug` an ordering prerequisite for both (see `workflow.md` §2's
partial order; the "any order" claim was wrong and is retired).

`auth_leak` entries are dual-purpose: security-relevant behavior the model
should learn to flag and fix, sourced from the same domains
`jac-sv-auth`/`jac-sv-multi-user` cover.

### 2.1 Extended debug task types (added 2026-07-20)

Two task types that don't fit the single-file inject-one-bug template:

| `task_type` | Shape | Gate |
|---|---|---|
| `cross_boundary_debug` | a `.sv.jac` endpoint and its `.cl.jac` consumer disagree on a type/shape/route name — find and fix across **both** files. The single-file taxonomy above never produces this, but it's the most common real-world fullstack failure mode | project-level gate (`run_jac_project`, `../spec.md` §7): both files must compile together after the fix; the broken pair must fail integration before it |
| `code_critique` | given working-or-broken Jac code, list what's wrong (or confirm it's clean) **without fixing it** — teaches review judgment separate from repair. Distinct from every other debug task, which always pairs symptom→fix | prose output, **no compiler gate** — every flagged issue must reference a real line/symbol in the input; critiques of *known-buggy* seeds (drawn from `buggy_variants.jsonl`) must flag the injected bug to pass. Runs on Fable like the rest of `debug` |

`cross_boundary_debug` seeds come from paired `.sv.jac`+`.cl.jac` doc
examples (`jac-fullstack-patterns`, `jac-sv-endpoints` + `jac-cl-components`
combined), with the LLM breaking exactly one side of the contract.

---

## 3. `explanation` — docs-grounded quiz Q&A (10%)

Scope: **grounded in Jac language/framework documentation only** — not
open-ended "explain this code" (that variant is explicitly deferred, see
`../spec.md` §3). Every question must be answerable from a specific doc
chunk, and the chunk is stored alongside the answer for traceability and
decontamination.

> Note for the eval protocol (`../workflow.md`): because CPT-v2 trains
> directly on these same docs (3x upsampled per its design.md §2), any
> `explanation` holdout row measures CPT's *doc absorption*, not downstream
> capability — treat it as a reported-but-non-voting metric in the CPT
> comparison. That doesn't reduce its value as SFT *training* data.

Source material: `jac-mcp` `search_docs`/`get_resource` — the guide pages
(`jac://guide/pitfalls`, `jac://guide/patterns`, every `jac://guide/*`
skill page) **plus the reference docs** the original catalog missed:
`jac://docs/cheatsheet`, `jac://docs/foundation`, `jac://docs/osp`,
`jac://docs/byllm`, `jac://docs/jac-client`, `jac://docs/jac-scale`,
`jac://docs/testing`, `jac://docs/cli`, `jac://docs/diagnostics`,
`jac://docs/python-integration`, and the tutorials.

| `task_type` | Format | Example shape |
|---|---|---|
| `concept_recall` | "What does `disengage` do inside a walker?" | direct factual recall, single doc chunk |
| `pitfall_identification` | "Why does this pattern fail?" (given a snippet drawn from `jac://guide/pitfalls`) | explain the specific gotcha, cite the fix |
| `pattern_application` | "Given this scenario, which pattern applies?" (multiple candidate approaches, pick + justify) | tests judgment, not just recall |
| `compare_contrast` | "`jac check` vs `jac run` — when does each over/under-reject?" or "node vs walker — which owns state?" or "walker endpoint vs `def:pub`" | forces the model to hold two concepts apart; may span two chunks — store both |
| `doc_example_reading` | "Given this canonical doc example, what does it output / what would break if line N were removed?" | grounds reasoning in a real example, not an abstract question |

Chunking: doc pages split at heading boundaries, chunks capped ~500 tokens
so grounding is checkable (see `../spec.md` §7 groundedness gate;
`compare_contrast` items may carry two chunks). One question minimum per
chunk, up to 3 for chunks covering multiple concepts. The groundedness
check's known false modes (correct paraphrase using synonyms fails; wrong
answer quoting chunk terms passes) are logged as an overlap score in
metadata, with the accept threshold calibrated on the pilot rather than
hard-coded.

---

## 4. `conversion` — Python → Jac (10%)

**Mostly reused as-is** from the existing `model-experiments/01-sft-dpo/sft_dpo/jacgen/` pipeline —
`mine.jac` (HF corpus mining), `scale_conversion.jac` (transpile + jac-run
gate), `idiomatic_batch*.jac` (hand-authored idiomatic core), `graph_seeds.jac`
(node/edge/walker conversion tier). This phase's contribution is folding its
existing output into the unified `sft_train.jsonl` via `build_manifest_v2.jac`
with `task_type` back-filled as `python_to_jac_function` (the transpile-tier
majority) or `python_to_jac_graph` (the graph tier) — **plus one new
generator**, `gen_graph_conversion.jac` (added 2026-07-20), that grows the
graph tier.

The slice is **snapshotted, not referenced live** — see `workflow.md` §2's
snapshot-and-hash rule (the upstream `01-sft-dpo/dataset/` files are
gitignored, single-copy, and truncatable by their own rebuild chain; a live
pointer would let the fresh/post builds silently read different data).

If the existing function-tier pool falls short of the 10% target (~1,250
at 12,500 total — currently already exceeds this at ~1,500+147, no action
needed at current dataset size), re-run `scale_conversion.jac` with a
higher `limit` to mine additional examples from the same Vezora corpus
**before** the snapshot is taken.

### 4.1 `python_to_jac_graph` scale-up (added 2026-07-20)

The graph tier is the thin, high-value slice: 31 hand-authored examples
(`graph_data/train.json`) against ~1,500 function-tier examples, despite
being where the real idiom-teaching headroom is — idiomatic-vs-transpile
similarity ~0.26 for graph examples vs 0.97 for functions
(`model-experiments/01-sft-dpo/sft_dpo/jacgen/README.md`). Growing it:

- **Target**: ~150-200 total (up from 31) — a ~5-6x increase.
- **Method**: `gen_graph_conversion.jac`, **Opus**-generated, forward
  authorship rather than mining (no public corpus of graph-shaped Python
  exists to mine the way Vezora covers plain functions). Opus authors a
  graph/tree/relational-shaped Python program (a scenario prompt bank
  seeded by the existing 31 examples' problem shapes — recursive
  structures, adjacency lists, tree walks, dependency graphs — extended to
  new shapes Opus proposes itself, reviewed against the bank for
  duplication before generation), then produces the idiomatic
  walker/node/edge Jac equivalent.
- **Gate**: `behavioral` — both Python and Jac outputs must match across
  the same distinct test-input set (`writer.jac`'s existing behavioral-gate
  pattern, unchanged). No new gate class needed; this is squarely inside
  what `run_jac` + pinned expected-output already verifies.
- **Run-tag treatment — deliberately shared, not per-tag**: unlike every
  other Opus/Fable generator in this catalog, `gen_graph_conversion.jac`
  runs **once, during the `fresh` build**, and its output is folded into
  the same snapshotted, hash-verified conversion slice both run-tags
  consume (`../spec.md` §5). This preserves `conversion`'s existing
  "no-variance-between-builds" property, which `../workflow.md` §5 relies
  on for `idiom_eval`'s arm-neutrality claim — if the graph tier were
  independently regenerated per run-tag like `code_gen`/`debug`, that claim
  would break and `idiom_eval` would need re-justifying as a voting metric.
- **Dedup**: new graph examples run through the same bucketed near-dup
  check (§0.2) against the existing 31 before acceptance — growing the
  tier should widen scenario coverage, not near-duplicate the existing set.

---

## 5. `trajectory` — LLM-simulated multi-turn sessions (10%)

**One Opus call per example** produces the entire 3-6-turn conversation
(the model writes both sides in a single generation — it is not unrolled
turn-by-turn across separate calls; cost accounting in `workflow.md` §5
reflects 1 call/example). Seeded by a `code_gen` task from the shared pool.

| `task_type` | Turn shape |
|---|---|
| `build_from_scratch` | user describes a feature in plain language → assistant builds it incrementally, user asks 1-2 follow-up refinements |
| `debug_session` | user pastes a real error (drawn from `raw_output/debug/buggy_variants.jsonl` — ordering dependency on `gen_debug`, see §2) → assistant diagnoses, may propose a wrong fix first, user reports it didn't work, assistant corrects |
| `refactor_session` | user provides existing working code, asks to restructure (e.g. split into `.impl.jac`, extract a walker from inline logic) |
| `add_feature_to_existing` | user provides an existing snippet, asks to extend it (add a field, add an endpoint, add a route) |
| `code_review_session` | assistant reviews user-provided Jac code unprompted-style, flags 1-2 real issues, user asks to fix one, assistant revises |

Gate: the **final** assistant turn's code must `jac run` clean (or
project-gate for multi-file). Intermediate turns showing a wrong attempt or
an error are intentionally not gated — a trajectory with zero friction
teaches the model nothing about recovering from its own mistakes, which is
the entire point of this category existing.

Storage shape: `messages: [{role, content}, ...]` — standard chat-format
list, compatible with the existing `mlx` split builder pattern in
`model-experiments/01-sft-dpo/sft_dpo/jacgen/build_splits.jac`.

---

## 6. `documentation` — code → docs (6%, added 2026-07-20)

Inverse of `explanation`: given working Jac code, produce the documentation
for it. Teaches the model to *articulate* what Jac code does — a capability
that also feeds back into the project (generated doc entries can seed future
`explanation` rounds after human review).

| `task_type` | Shape |
|---|---|
| `docstring_authoring` | given a function/walker/node, write its docstring (purpose, params, return, side effects on the graph) |
| `api_reference_entry` | given a `.sv.jac` endpoint (or small endpoint group), write the API-reference entry: route, auth requirement, request/response shape, error cases |
| `module_overview` | given a small multi-archetype file, write the module-level overview: what it models, how the pieces relate, entry points |

Generation: seeds from the same shared pool (code-bearing seeds only).
Prose output, no compiler gate — runs on **Fable** (`../spec.md` §4.1):
factual-precision-critical, and hallucinated parameter names or invented
behavior in docs is exactly the ungated failure mode Fable is assigned to
guard. Lexical gate: the generation prompt **requires backticks around
every code symbol**, and the check is "every backticked token occurs in the
seed code," with a small allowlist (`root`, `walker`, `jac.toml`, builtin
type names) to kill false positives — precise and mechanically cheap.

---

## 7. `migration` — deprecated → current Jac, file scope (4%, added 2026-07-20)

Whole-file sibling of `code_gen`'s single-snippet `syntax_migration`
(§1.1): given a small complete file written against deprecated Jac
syntax/patterns, produce the fully migrated current version.

**Inventory caveat (from the feasibility audit — governs this category's
size):** no changelog resource exists in jac-mcp. The deprecated-pattern
inventory must be hand-assembled from scattered doc mentions, and the
currently confirmable pool is small: 3 compiler-warnable forms (W0061
paren filter `(?:...)`, W0062 `root()`, W0063 JSX spread), hard-fail v1
forms (`import:py ...`, `include:jac`), the `jac js` CLI alias, plus
whatever the `jac-scaffold` guide lists. **Build the inventory first**
(seed-pool step §8.4); if it can't support 500 distinct examples at "2-5
distinct deprecated patterns per file," shrink the category and return the
difference to buffer — do not pad with repetition.

| `task_type` | Shape |
|---|---|
| `file_syntax_migration` | one `.jac` file, 2-5 distinct deprecated patterns (drawn from the inventory) → current syntax throughout |
| `scaffold_modernization` | output of `jac create --use <template>` (which ships deprecated syntax per `jac-scaffold`) → modernized equivalent, the post-scaffold checklist applied |

**Gate (corrected 2026-07-20 — verified compiler behavior):** `jac run`
emits **no deprecation warnings, ever** — warnings like W0062 surface only
under `jac check`. So the gate is three-part (the third part added
2026-07-21 — see below):

1. Migrated file must pass `jac run` (the normal generation gate —
   unchanged, and `jac check` is still never used to reject generated
   code).
2. The deprecated original must either **fail `jac run`** (hard-fail v1
   forms) or **produce a deprecation warning under `jac check`/`lint_jac`**
   — an explicit, narrow exception to the never-`jac check` rule: check is
   used only to *detect warnings on the input*, never to reject output.
   Pairs where the original passes `jac run` silently *and* produces no
   check-warning are rejected (nothing was actually migrated).

**Amendment (2026-07-21) — a second, necessary exception to "never gate on
`jac check`":** part 1 alone (migrated code passes `jac run`) cannot
detect a PARTIAL migration — a file combining several `check_warning`-type
deprecated patterns (e.g. W0061 + W0062 together) where only some were
fixed will still pass `jac run` cleanly, since `jac run` never surfaces
deprecation warnings regardless of whether they're present. So there is a
**gate part 3**: the migrated (output) code is re-checked under full
`jac check`, and rejected if any of the original combo's specific
deprecation-warning codes still appear. This is the second, narrowly-scoped
place this project's "never gate output on `jac check`" rule is
deliberately relaxed (`../spec.md` §7 documents both exceptions) — it
never gates on general `jac check` pass/fail, only on the presence of the
*specific* codes belonging to that example's own composed patterns.
Implemented in `gen_migration.jac`'s `run_collect` as "gate part 2" (the
input-detection check above is folded into prepare-time verification in
the real implementation, not a separate collect-time step — see the file's
own docstring for the exact mechanics).

Runs on **Opus** (`../spec.md` §4.1): token-heavy whole-file rewrites,
mechanically checkable, low judgment risk.

**Real inventory + pilot result (2026-07-20/21):** the confirmable
deprecated-pattern inventory landed at **4 entries**, each empirically
verified against the real compiler: W0061 (paren filter), W0062
(`root()`), W0063 (JSX spread), and the hard-fail v1 `import:py` prefix
form. `include:jac` was investigated and deliberately excluded (a
single-file `run_jac`-style gate structurally cannot demonstrate an
`include` migration, which requires a real companion file). `jac js` (CLI
alias) and the `jac-scaffold` guide were not investigated — noted as an
open gap, not a blocker. With only 4 patterns, `itertools.combinations`
over sizes 2-4 yields 11 valid distinct combos (6+4+1); a 10-item pilot
(all 6 two-pattern + 4 of the three-pattern combos) ran end-to-end for
real — Opus-generated, gated, **10/10 accepted, 0 rejected**. The ~500
example target this category was originally scoped for is not reachable
from a 4-pattern inventory; per this section's own contingency, the
buffer category absorbs the shortfall rather than padding with
repetition.

---

## 8. Seed pool construction

Single shared builder (`seed_pool.jac`, see `workflow.md` for its place in
the module graph) feeding `code_gen`, `debug`, `explanation`, `trajectory`,
`documentation`, and `migration`. Five source tiers (expanded 2026-07-20
from a jac-mcp-only pool — `../spec.md` §3 "Seed source for code"):

### 8.1 Tier 1 — jac-mcp examples

Enumerate every `jac-mcp` example via `list_examples` (9 app-scale
examples, untagged), fetch each with `get_example`, **tolerating
per-example fetch failures** (the knowledge map itself only guarantees a
subset). Decompose fetched apps into archetype/walker/endpoint-level
seeds; drop shadcn `ui/*.cl.jac` boilerplate.

### 8.2 Tier 2 — jac-mcp docs

Enumerate doc pages via `search_docs`/`get_resource` — every
`jac://guide/*` skill page (including `jac-has-fields` and
`jac-core-cheatsheet`, previously missed) **and every `jac://docs/*`
reference** (`cheatsheet`, `foundation`, `osp`, `byllm`, `jac-client`,
`jac-scale`, `testing`, `cli`, `diagnostics`, `python-integration`,
tutorials) — extract fenced code blocks.

### 8.3 Tier 3 — jaclang's own repo (added 2026-07-20)

Clone the `jaclang` compiler repo itself (separately from jac-mcp's
bundled docs) and mine its test fixtures and stdlib source directly — the
same canonical-language category as tiers 1-2, just untapped because
jac-mcp only bundles docs, not the language repo's own tests. Filter to
files that parse standalone (`jac check -p`) and, where they carry an
assertable behavior, run (`jac run`) to pin `expected_output` same as
tier 2. No Fable review needed here — compiler test fixtures are
by-definition canonical, that's their job.

### 8.4 Tier 4 — multi-org public Jac GitHub scrape, Fable-reviewed (added 2026-07-20)

Reuses the cloning/manifest-SHA infrastructure `03-cpt-only/cpt_build/build_cpt.py`
already built for CPT's own corpus pull (`seed_scrape.jac`, new module,
`../spec.md` §4) — clone candidate public repos containing `.jac` files to
scratchpad, record SHAs in a manifest, same convention. Two gates, in order:

1. **Parse gate**: `jac check -p` (parse-only, not the full type-checker —
   matches the precedent CPT's own corpus build already used for its code
   corpus, and avoids the project's standing "never gate on `jac check`"
   rule, which is specifically about *rejecting generated output*, not
   about *filtering scraped input* before it's ever a candidate).
2. **Ponytail-rubric Fable review gate** (`gen_seed_review.jac`, new
   module) — this is the step the user specifically asked for: "make sure
   no bad or inefficient code is being used." Fable reviews each
   parse-gate-surviving candidate against the `/ponytail:ponytail` rubric
   and rejects anything that:
   - reinvents something the Jac stdlib or a native language feature
     already covers,
   - carries unneeded abstraction for a single use site (interface with
     one implementation, config for a value that never changes),
   - is obviously inefficient (O(n²) where O(n) is trivial, redundant
     re-computation, dead code paths),
   - is Python-shaped Jac rather than idiomatic OSP where the problem is
     graph-shaped (the same idiom bar `dpo-plan.md` §2.1 already holds
     conversion examples to).

   Accepted candidates get a `seed_source: "github_scrape"` tag and a
   `review_note` field recording Fable's verdict, for audit. This gate is
   the reason tier 4 is Fable-assigned, not Opus (`../spec.md` §4.1) —
   judging code quality is exactly the precision/consequence-sensitive
   work Fable is assigned across this catalog; a bad call here poisons
   every downstream example the seed fans out into.

   Rejection rate is unmeasured until the pilot runs — budget for it
   (`datagen/workflow.md` §5 cost accounting).

Exclude repos already consumed by CPT's own corpus in ways that would
make this tier redundant with tier 3 or introduce a different kind of
contamination — check CPT's corpus manifest (`03-cpt-only/dataset/cpt-v2/manifest.json`)
for the org list already covered there and avoid the exact same corpus
tier's *code* slice, which CPT-v2 deliberately dropped (`docs/cpt-2/design.md` §1
"drop code entirely") — that's a CPT-corpus-composition decision, not a
statement about this tier's validity, but re-scraping the identical set
CPT already excluded is a reasonable place to widen the org list rather
than duplicate CPT's discovery work outright. Same decontamination
requirement as every other tier: `decontam_v2.jac`'s shingle overlap check
against eval holdouts, applied at seed-acceptance time, not just at
example-acceptance time.

### 8.5 Tier 5 — this repo's own app code (added 2026-07-20, reverses a prior exclusion)

Mine `jms/`, `model-experiments/*/sft_dpo/jacgen/*.jac`, and studio
`.sv.jac`/`.cl.jac` files directly. The original exclusion
(`../spec.md` §3, "avoids overfitting to studio-app style") is reversed
now that tiers 3-4 add enough independent-style volume to dilute any
single-project style dominance — this tier alone would have posed the
overfitting risk the original decision was guarding against; alongside
four other tiers it's additive real-world signal instead. No Fable
review needed — this is the project's own reviewed, working code.

### 8.6 Shared post-processing (all tiers)

1. **Run every code seed once and pin its stdout as `expected_output`** in
   `seed_pool.jsonl`. This is what the behavioral gate compares against
   (nothing else populates expected output for doc-fence or scraped
   seeds), and it freezes behavior identically for both run-tags. Seeds
   that don't run standalone are kept but flagged `gate_class: compile_only`
   (client components, endpoint snippets, `by llm()` demos — the gate
   matrix in `../spec.md` §7 governs what they can seed).
2. Assemble the deprecated-pattern inventory (§7's caveat) as a tagged
   sub-pool for `syntax_migration` and `migration`.
3. Author the app-idea prompt bank for `schema_design` (§1.1) — short
   plain-language app descriptions, hand-curated once, stored alongside the
   pool (these are prompts, not code seeds, but versioned identically so
   both run-tags draw from the same bank).
4. Dedup seeds against each other (exact-match + bucketed near-dup, §0.2)
   before they ever reach a generator — no point spending LLM calls (Opus
   or Fable, see `../spec.md` §4.1) on the same snippet twice. Cross-tier
   dedup matters here specifically — tier 5 (this repo) and tier 1
   (jac-mcp examples) could plausibly overlap if any jac-mcp example was
   itself drawn from this project's history; the same bucketed check
   catches it regardless of tier.
5. Every seed carries a `seed_tier` field (1-5) in metadata, alongside the
   existing `seed_id`/origin tag — `dataset_stats_v2.jac` reports
   composition by tier as well as by category/task_type, so the eventual
   dataset's tier mix is visible, not just its category mix.

Output: `seed_pool.jsonl`, not run-tag-scoped (see `../spec.md` §5 — kept
identical across `fresh` and `post_cptv2` builds so only LLM-authored content
varies between the two datasets). **Frozen after the `fresh` build**: its
sources ship inside the jaclang package (tiers 1-2) or are cloned from
external repos that can be updated or go private (tiers 3-4) — either
way, they can drift — so the file is hash-recorded at freeze time and the
`post_cptv2` build verifies the hash instead of regenerating
(`workflow.md` §2).
