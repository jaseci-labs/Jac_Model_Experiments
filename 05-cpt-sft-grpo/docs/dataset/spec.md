# dataset/spec.md — Multi-Source RL Corpus + Type-B AST-Equivalence Grading

Companion to `../spec.md` (umbrella design, training matrix, eval) and `workflow.md`
(the mining runbook). This file is the corpus and grader design: which repos, how a task
is mined, what exactly the AST-equivalence check compares, how partial credit is
computed, how the splits are built, how contamination is checked, and what the
provenance/licensing position is.

Nothing here is built yet. Every count marked **TBD** is measured during mining and
written into the corpus manifest (`workflow.md` Phase 6) — not guessed now.

---

## 0. What changes, in one table

| | `02-rl-grpo` (built, run) | `05-cpt-sft-grpo` (this design) |
|---|---|---|
| Source repos | `this_is_jac` only | 17 pinned repos (`this_is_jac` + 16), from CPT-v1's corpus |
| Task shape | HOLE-marker hole-fill (Type A) | unchanged — HOLE-marker hole-fill |
| Task authoring | hand-authored drivers in `rl/drivers/` | unchanged convention, extended per-repo |
| Pass gate | byte-exact stdout | **α-normalized AST equality**, OR byte-exact stdout where deterministically runnable |
| Reward shape | monotone tiers, `difflib` body similarity in every tier | monotone tiers, **AST similarity** in every tier (difflib fallback when the completion does not parse) |
| Runnability requirement | hard — a nondeterministic or non-runnable file cannot become a task | soft — non-runnable files are minable via the AST route, tagged `grade_mode: ast_only` |
| Holdout | file-disjoint, family-interleaved, `difflib`≥0.85 decontam | same logic, generalized to **repo × family** |

---

## 1. Source corpus — the 17 repos

Reuses CPT-v1's already-cloned, already-parse-gated, already-decontaminated code corpus.
`03-cpt-only/dataset/cpt/manifest.json` pins one commit SHA per repo; **18 repos are
pinned there, 17 of which contributed `.jac` code through the parse gate** — the 18th
(`jaseci-llmdocs`) has no `code_gate` entry and contributed documentation text, not
code, so it yields no minable drivers.

Per-repo `.jac` files seen and passing CPT-v1's parse gate, verbatim from that
manifest's `code_gate` block:

| repo | pinned SHA (short) | `.jac` files | passed parse gate | notes for mining |
|---|---|---|---|---|
| `this_is_jac` | `30dcca9b` | 72 | 72 | the existing RL pool's source; 84 of the 116 built drivers trace here |
| `jac` (jaclang itself) | `199900a8` | 86 | 85 | compiler examples + stdlib; richest single new source |
| `Agentic-AI` | `a6d82ff4` | 130 | 107 | largest file count; also the largest gate-failure count (23) |
| `jac-shadcn` | `1d691009` | 87 | 87 | almost entirely `.cl.jac` client components — **not** deterministically runnable |
| `jac-load-test` | `8832ffcd` | 74 | 73 | |
| `jac-mcp-playground` | `9394d762` | 52 | 52 | |
| `jasketch` | `50a4f944` | 47 | 47 | |
| `jac-client-playground` | `d919205b` | 47 | 47 | client-heavy, same runnability caveat as `jac-shadcn` |
| `Algo` | `c31ad99d` | 45 | 44 | algorithmic, mostly pure-function — the best deterministic-stdout supply |
| `llvm-slice` | `279401bd` | 28 | 27 | |
| `littleX` | `059b8375` | 17 | 16 | OSP social-graph idiom; overlaps `this_is_jac/littlex/` — see §7 |
| `jaseci-blogs` | `b0e79297` | 17 | 17 | prose repo with embedded code |
| `jaseci-studio` | `1f8bebe4` | 12 | 12 | fullstack `.sv.jac`/`.cl.jac` |
| `agentic-ai-tutorial` | `3ec45e5b` | 12 | 12 | |
| `the-jac-workshop` | `297cd43f` | 8 | 7 | |
| `inr-codelabs` | `48278f04` | 4 | **0** | contributed no gate-passing file in the CPT build; expect zero yield |
| `tree-sitter-jac` | `825e595d` | 3 | 2 | grammar fixtures, likely not runnable |
| `jaseci-llmdocs` | `e2f53b0b` | — | — | docs-only; no `code_gate` entry, no drivers |

Totals across the 17 code repos: **741 `.jac` files seen, 707 passed the parse gate.**
Excluding `this_is_jac`: **669 files, 635 passing** — the new supply this phase adds.
The CPT build's own `sources.code` block records 691 rows / 992,184 tokens landing in
its code corpus after its decontam pass dropped 7 files; the residual difference between
707 gate-passing files and 691 recorded rows is not explained inside the manifest, so
treat 635 as an **upper bound on new minable files**, not a task count.

**Files ≠ tasks.** A file yields between 0 and ~30 hole-fill tasks depending on how many
clean, self-contained unit bodies it has (`this_is_jac/littlex/social_graph.jac` alone
yields 32 in the current pool; `this_is_jac/analytics.jac` yields 2). Expected corpus
size is **TBD, measured during mining**.

---

## 2. The HOLE-marker driver convention

Read `02-rl-grpo/rl/build_tasks.jac` — this is the convention being extended, described
here exactly as that file implements it.

### 2.1 What a driver is

A driver is a complete, deterministic `jac run`-able `.jac` file living in
`02-rl-grpo/rl/drivers/*.jac`, with exactly one unit's body wrapped in sentinel
comments:

```
# >>>HOLE id="lib_analyse_text" instruction="Return the word count and mean word length"
<body statements>
# <<<HOLE
```

Drivers are **authored**, not auto-extracted: a real source file is turned into a
self-contained runnable driver (imports inlined or stubbed, a deterministic entry point
added, an `# Origin: this_is_jac/<path>.jac` comment recorded), then one unit body is
wrapped in the sentinels. 116 such drivers exist today.

### 2.2 What `build_tasks.jac` does with one

Per driver, in order:

1. **Parse the markers.** `parse_hole` scans for the first `>>>HOLE` line and the first
   `<<<HOLE` line, requires `end > start`, and extracts `id` and `instruction` with
   `>>>HOLE\s+id="([^"]+)"\s+instruction="(.*)"`. The `>>>HOLE` line's leading
   whitespace becomes the `indent` used for splicing. A malformed marker → `SKIP`.
2. **Capture ground truth.** Run the driver **as authored** (real body present) in an
   isolated temp cwd (`cwd=d`, so jac's `.jac/` persistence is fresh per run and no
   persistent `root` graph accumulates across drivers). Non-zero exit or empty stdout →
   `FAIL`.
3. **Determinism guard.** Run it a second time; if stdout differs, `FAIL` — a task whose
   output varies (`jid()`, time, unseeded RNG) is unusable as an exact-match target.
4. **Record origin.** `Origin:\s*(this_is_jac/\S+?\.jac)` → the `source` field, which is
   what makes the file-disjoint holdout possible. No `Origin:` line → `source:
   "unknown"` (32 of the current 116 tasks are in that state — authored/synthetic
   drivers with no upstream file).
5. **Emit three artifacts:**
   - `dataset/rl/templates/<id>.jac` — the file with the whole marker region replaced by
     a single `indent + "__HOLE__"` line. This is what the reward splices into.
   - `dataset/rl/refbodies/<id>.txt` — the real body text between the sentinels. **Never
     shown to the model**; reward-side only, for the similarity term.
   - one line in `dataset/rl/tasks.jsonl`:
     `{"prompt": …, "answer": "{\"id\":…, \"source\":…, \"expected_output\":…, \"timeout\":20}"}`
     — the `{prompt, answer}` schema `mlx_lm_lora`'s `GRPODataset` reads.
6. **Build the prompt view.** The marker region is replaced by
   `# TODO: write the body for this unit`, and the prompt is
   `instruction + "\n\nComplete the Jac file below: replace the `# TODO` line with the
   correct Jac code for that unit. Return only the replacement code, in a ```jac code
   block."` + the fenced file.

`tasks.jsonl` is truncated and fully rebuilt on every run — the corpus is deterministic
given the drivers.

### 2.3 The `sg_bucket` precedent (why it matters for 17 repos)

`build_tasks.jac` carries an opt-in (`JAC_RL_SG_BUCKETS=1`) that splits
`this_is_jac/littlex/social_graph.jac` — one file supplying 32 tasks, ~28% of the current
116-task corpus — into four synthetic sub-sources (`channel` / `tweet` / `profile` /
`feed`) so a coherent slice of it can sit in a file-disjoint holdout. Without it, that
file is larger than the per-file holdout cap and can only ever be trained on, making its
idiom's generalization unmeasurable.

That is the general problem the multi-repo corpus inherits at a larger granularity: a
single dominant *repo* would recreate exactly the imbalance `build_rl_splits.jac` was
written to fix. §5 handles it.

### 2.4 What extends for 05

| aspect | change |
|---|---|
| Driver location | one subdirectory per repo — `rl/drivers/<repo>/*.jac` — so `glob` yields a repo-attributable path even when a driver has no `Origin:` line |
| `Origin:` regex | generalize from `this_is_jac/\S+?\.jac` to `<repo>/<path>.jac` across all 17 repo names; keep the failure mode loud (`source: "unknown"` is reported, never silently accepted, once repo subdirectories exist) |
| New `answer` fields | `repo` (one of the 17), `grade_mode` (`stdout_and_ast` \| `ast_only`), `license` (§7), `upstream_sha` (the repo's pinned SHA, so a task is traceable to an exact commit) |
| Determinism guard | no longer a hard reject. A driver that runs but is nondeterministic, or that does not run at all, is admitted with `grade_mode: ast_only` and **no** `expected_output`. A driver that runs deterministically keeps `grade_mode: stdout_and_ast` and both pass routes |
| Gold body | unchanged — `refbodies/<id>.txt`, reward-side only |

The `ast_only` path is what unlocks `jac-shadcn` (87 gate-passing files, almost all
client components), `jac-client-playground`, and `jaseci-studio` — repos that are
categorically unminable under a stdout-only gate.

---

## 3. Task selection rules (per repo)

1. **One hole per driver.** `parse_hole` takes the first marker pair; multiple holes per
   file are not supported and must not be authored.
2. **The hole is a unit body**, not an expression or a partial statement — the splice
   and the `unwrap_unit` scar both assume unit granularity.
3. **The gold body must be non-trivial and non-guessable.** A body that is a single
   `return None;` teaches nothing and inflates the pass rate. Minimum: TBD, calibrated
   during mining (candidate rule: ≥2 statements or ≥1 graph/OSP operation).
4. **The surrounding file must be self-contained.** No network, no filesystem outside the
   temp cwd, no live LLM call (`by llm()`-bearing units are excluded — a live API call
   inside a reward loop is nondeterministic spend, the same rule
   `04-cpt-sft/docs/spec.md` §7 applies to its `compile_only` gate class).
5. **Per-repo yield is reported, never quotaed.** If `inr-codelabs` yields 0 (its CPT
   parse-gate record says 0/4), that is the finding; do not manufacture drivers to
   balance a table.
6. **Family labels** (`id` prefix before the first `_`, per `build_rl_splits.jac`'s
   `family_of`) must be assigned deliberately per repo — they are the interleave axis,
   so `id`s like `shadcn_button_render` / `algo_bfs_visit` need prefixes that mean
   something.

---

## 4. Type-B AST-equivalence grading

The core new design. Replaces byte-exact stdout as the pass bar in **both** the GRPO
reward and the eval, per `../spec.md` §3.

### 4.1 Why exact-stdout had to go

| failure mode of exact-stdout | consequence in `02-rl-grpo` |
|---|---|
| A structurally correct body that prints in a different order, or whose driver output legitimately varies, scores **0** | Real capability invisible; the ladder can only move if the model reproduces the gold *behavior* byte-for-byte |
| A body that hardcodes the expected print scores **1.0** | The gate rewards cheating that AST-equivalence rejects outright |
| A file with no deterministic entry point cannot be a task at all | The corpus was structurally limited to runnable, deterministic code — which is why it never left `this_is_jac` |
| The gate says nothing about *how close* a miss was | Handled only by the diagnostic `osim`; the pass bar itself was all-or-nothing |

`02-rl-grpo/docs/rl/01-design.md` §6 left "Type B AST-equivalence grader — design when
the whole-file track starts" as an open item. This is that design, applied to hole-fill
rather than whole-file.

### 4.2 What the check compares

**Comparison unit:** the *hole unit's body*, not the whole file. Procedure:

1. Extract the completion's code (strip a ```` ```jac ```` fence if present) — the
   existing `reward_logic.jac:extract_jac` path.
2. Pull **that specific unit's** body by brace-matching on the unit name recovered from
   the template (`hole_unit_name` → `unit_body`), falling back to `unwrap_unit` for a
   bare body. Both carried scars (`../strat.md` #1, #2) apply unchanged.
3. Splice into the template at `__HOLE__`.
4. Parse the **spliced file** (not the bare body — a body is not a parseable unit on its
   own) and locate the sub-tree of the hole unit.
5. Do the same for the gold: splice `refbodies/<id>.txt` into the same template, parse,
   locate the same sub-tree. This is done once per task and cached; it also gives the
   grader's self-test its ground truth (gold must score exactly 1.0 against itself).
6. Normalize both sub-trees (§4.3), then compare (§4.4).

**Parser:** whatever produces a stable, structural representation of a Jac file. The
options that exist in this repo today are jaclang's own AST access (the `jac-mcp`
`get_ast` tool exposes one; `jac check -p` is already used as a parse-only gate in
`reward_logic.jac` and in the CPT corpus build). **Which exact entry point is used is an
implementation decision, not a design one** — the requirement is that it (a) is
deterministic, (b) survives a syntactically valid but semantically odd body, and (c) is
cheap enough to run once per rollout (GRPO samples `group_size` rollouts per prompt).
Benchmark it in the Phase-3 grader self-test (`../workflow.md`); if per-call parse cost
dominates, cache by `(task_id, normalized body text)` exactly as
`reward_logic.jac:jac_behavioral` already caches scores by `(answer, completion)`.

### 4.3 Normalization — what is allowed to differ

The normalizer defines the equivalence relation. Everything in the "erased" column may
differ between two bodies that still count as equivalent.

| Property | Treatment | Rationale |
|---|---|---|
| Source positions (line/col/span) | **erased** | Formatting, not semantics |
| Whitespace, indentation, line breaks | **erased** (implicit — the AST has none) | The entire point of moving off text comparison |
| Comments | **erased** | A correct body with no comments is still correct |
| Docstrings | **erased** | Same; also the source of `reward_logic.jac`'s historical ~4× undercount when a naive unwrap grabbed one |
| Local variable / parameter-local binding names | **α-renamed** to `v0, v1, …` in order of first binding, consistently within the body | `total`/`sum`/`acc` are the same program |
| Loop induction variables, comprehension binders, `with … as` targets | **α-renamed**, same scheme | Same |
| Numeric literal formatting (`1` vs `1.0` vs `0x1`) | **canonicalized** by value **only when the literal's static type is unambiguous**; otherwise preserved | `1` and `1.0` are not the same value in a typed language; do not over-normalize |
| String quote style (`'x'` vs `"x"`) | **canonicalized** | Lexical only |
| f-string / interpolation structure | **preserved** | Changing it changes output |
| Statement order | **preserved** | Order is semantics |
| Identifiers that escape the body — parameter names as *called* by the caller, `has` field names, node/edge/walker/obj type names, called function names, imported symbols, `report`/`visit` targets | **preserved verbatim** | Renaming any of these changes the program's interface or its graph semantics. This is the single most important normalization boundary: α-renaming must be scoped strictly to bindings created *inside* the hole body |
| Redundant parentheses | **erased** (they do not survive parsing) | |
| Dead code, unused locals | **preserved** | An extra unused statement is a real difference; it costs partial credit, it is not free |

**Explicitly not done:** no commutativity rewriting (`a+b` ≢ `b+a`), no algebraic
simplification, no reordering of independent statements, no control-flow canonicalization
(`if not x {A} else {B}` ≢ `if x {B} else {A}`). Every one of those is a semantics-preserving
transform in *some* contexts and not others, and a grader that guesses wrong in the
model's favor is worse than one that is merely strict. Strictness is the safe direction:
a strict grader under-credits, which biases the experiment toward the null, which is the
conservative bias for both H1 and H2.

### 4.4 Equality and partial credit

**Pass (`ast_equal`)** = the two normalized sub-trees are identical — same node types,
same structure, same preserved identifiers, same literal values, same order. Implemented
as equality of a canonical serialization (a deterministic normalized AST dump), so the
check is a string compare on a canonical form, not a hand-written tree walk.

**Partial credit (`ast_sim` ∈ [0,1])** is required in every tier — the σ>0 scar
(`../strat.md` #4). Two-level scheme, cheapest first:

1. **Primary — normalized tree-edit similarity.** `1 − (edit_distance / max(|A|,|B|))`
   over the normalized trees, where `|·|` is node count. Gives a graded, monotone signal
   that degrades smoothly as a body drifts from gold.
2. **Fallback — node-type bag similarity.** If tree-edit distance is too slow at rollout
   rate (measured, not assumed), substitute the F1 of the multiset of
   `(node_type, depth)` pairs. Coarser, but O(n) and still structural.
3. **Last-resort fallback — `difflib` text ratio on the normalized body text.** Used
   **only when the completion does not parse at all**, so `ast_sim` is never undefined
   and a group of syntactically broken rollouts still has within-group variance. This is
   exactly the role `difflib` plays in `reward_logic.jac` today; it is demoted, not
   removed.

**Reward tiers** — the monotone structure of `reward_logic.jac` is preserved verbatim in
shape; only the pass condition and the similarity term change:

| tier | condition | score |
|---|---|---|
| T4 | `ast_equal` **or** (task is `stdout_and_ast` **and** runs with byte-exact stdout) | **1.0** |
| T3 | runs, stdout differs (or task is `ast_only` and the spliced file runs) | `0.40 + 0.25·line_frac + 0.15·ast_sim` (≤ 0.80) |
| T2 | parses (`jac check -p` exit 0), does not run | `0.20 + 0.15·ast_sim` (≤ 0.35) |
| T1 | neither | `0.15·ast_sim` (≤ 0.15) |

`line_frac` is `reward_logic.jac:output_score` unchanged — the fraction of expected
stdout lines reproduced exactly at the same position; for `ast_only` tasks there is no
expected stdout, so `line_frac = 0` and T3's ceiling drops to 0.55. Each tier's ceiling
stays strictly below the next tier's floor, so a pass always dominates a non-pass — the
property the old additive dense formula lacked.

**Idiom stays out of the reward.** `reward_logic.jac` removed it (trivially gamed by
stuffing `-->` tokens, anti-correlated with terse gold bodies). It is reported as a
descriptive statistic only.

### 4.5 How this differs from the two things it replaces

**vs. the old exact-stdout gate:**

| | exact-stdout | Type-B AST-equivalence |
|---|---|---|
| What passes | output bytes match | structure matches after α-normalization (or output bytes match, where available) |
| Hardcoded-print cheat | **passes** | fails |
| Correct body, different variable names | passes (output is unaffected) | passes |
| Correct body, nondeterministic driver | impossible — task rejected at mining | passes |
| Non-runnable file (client component, library) | cannot be a task | is a task, `ast_only` |
| Correct-but-different algorithm reaching the same output | passes | **fails** — this is the one place AST-equivalence is *stricter*, and it is deliberate: partial credit via `ast_sim`, not a pass |
| Partial credit | none in the pass bar | graded, and it is the same signal the reward uses |

Both routes are kept for `stdout_and_ast` tasks precisely because they fail in opposite
directions. Report the split (`ast_only_pass` / `stdout_only_pass` / `both`) in every
eval row — the size of the disagreement is itself a finding about the grader.

**vs. the old dense v2 reward** (`02-rl-grpo/docs/rl/01-design.md` §4:
`0.25·compiles + 0.25·runs + 0.25·output + 0.10·idiom + 0.15·body_sim`):

- The dense formula was **additive and non-monotone** — a wrong-output completion with
  high idiom and similarity could outscore a correct terse one. `reward_logic.jac`
  already replaced it with the monotone tiers above; 05 inherits the replacement, not the
  original.
- The `idiom` term is gone (§4.4).
- `body_sim` (raw `difflib` on text) becomes `ast_sim` (structural), which is the actual
  substantive change: a completion that is structurally right but textually reformatted
  used to score like a near-miss and now scores as a pass.
- What is **kept from the dense era** is the principle it existed for: a similarity term
  present in every tier, including for completions that do not compile. That is the only
  reason a group of all-failing rollouts has non-zero variance.

---

## 5. Splits — holdout, trainpool, valid

Generalizes `02-rl-grpo/rl/build_rl_splits.jac`. The three bugs that file fixed, plus
the one the multi-repo corpus adds:

| # | bug | fix (existing) | 05 generalization |
|---|---|---|---|
| 1 | Holdout leaked structure — a family-stride split put the same source file in both train and holdout | **File-disjoint**: whole source files are held out; a held-out file contributes nothing to train | unchanged in kind, now across 17 repos' files |
| 2 | Trainpool was family-ordered, so `pick_rung`'s front-slice gave rungs 1–20 **zero** `social_graph` tasks while the holdout was 47% `social_graph` — the "more tasks" curve was really a "which family" curve | **Round-robin family interleave**: every prefix-N is family-balanced *and* a strict superset of the previous rung | interleave on **(repo, family)** round-robin, so a prefix-N is balanced across repos *and* families. Otherwise the same failure returns one level up |
| 3 | `valid.jsonl == holdout.jsonl` — the trainer selected checkpoints on the test set | **valid carved from the trainpool tail**, never from holdout | unchanged |
| 4 | *(new)* one repo could dominate the holdout | — | per-repo holdout cap in addition to the per-file cap |

**Existing mechanics, kept:**

- Greedy **family-coverage** holdout selection: repeatedly pick the eligible source file
  that adds the most *new* families to the holdout (tiebreak: smaller file), so the
  holdout spans as many idiom classes as the corpus allows instead of collecting the
  trivial ones.
- **Per-file cap** `max(1, int(n_hold · 0.5) + 1)` — no single file may exceed roughly
  half the holdout. A file bigger than the cap is ineligible and trains instead (which is
  exactly why `social_graph.jac` can never be held out whole, and why `sg_bucket` exists).
- **Body-level decontam**: drop any holdout task whose gold body is ≥`JAC_RL_DECONTAM`
  (default 0.85) `difflib`-similar to **any** train body — catches cross-file
  near-duplicates a file split alone misses.
- **Loud warnings** rather than silent degradation: `build_rl_splits.jac` prints a WARN
  when the holdout lands under target or spans fewer than 2 families. Keep both, and add
  a third for repo count.
- Env knobs: `JAC_RL_HOLDOUT_N` (15), `JAC_RL_VALID_N` (4), `JAC_RL_DECONTAM` (0.85).

**05 targets:** holdout size and composition are **TBD, set after mining measures per-repo
yield.** Two hard constraints, not negotiable: the holdout must span **≥2 families** (the
existing warn condition) and **≥4 repos**, and no single repo may exceed ~⅓ of it. A
holdout of 15–18 items (the current corpus's size) is too small to resolve the effects
H1/H2 predict — `02-rl-grpo`'s own holdouts were n=11–32 with correspondingly wide Wilson
intervals, and `04-cpt-sft` needed n=855 to reach z≈1.28 on a 2.8pp gap. Set the target
from the measured supply and report the resulting minimum detectable effect honestly
alongside it.

**On-disk state to be aware of:** `02-rl-grpo/dataset/rl/` currently holds
`holdout.jsonl` (18), `trainpool.jsonl` (62), `valid.jsonl` (4) — 84 rows total, built
when the corpus was 84 tasks, while `tasks.jsonl` has since grown to 116. The splits are
stale relative to the tasks file. 05 rebuilds everything from scratch into its own
`05-cpt-sft-grpo/dataset/`; it does not inherit those files.

---

## 6. Corpus manifest

Mining writes one manifest, `05-cpt-sft-grpo/dataset/rl/manifest.json`, modeled on
`03-cpt-only/dataset/cpt/manifest.json`'s shape (which records `repos` → SHA,
per-repo `code_gate` counts, `decontam` counts, `holdout_files_excluded`, and packed
totals). Required keys:

| key | contents |
|---|---|
| `repos` | repo → pinned SHA, for all repos actually mined |
| `licenses` | repo → SPDX identifier or `"unverified"` (§7) |
| `yield` | repo → `{files_seen, files_parsed, drivers_authored, tasks_built, tasks_failed}` |
| `grade_modes` | `{stdout_and_ast: n, ast_only: n}` |
| `families` | family → task count |
| `splits` | `{holdout: n, trainpool: n, valid: n}` + the holdout's file list, repo histogram, family histogram |
| `decontam` | `{shingle_dropped: n, body_sim_dropped: n, threshold: 0.85, shingle_n: 14, shingle_overlap: 0.5}` |
| `grader` | parser entry point + version, normalizer version, `ast_sim` method actually used (tree-edit vs bag), and the self-test result |
| `built_at`, `jaclang_version` | reproducibility — grading is version-sensitive; a jaclang upgrade can change the AST |

The `jaclang_version` pin is not bookkeeping: the AST is the grader, so a jaclang upgrade
mid-matrix would silently change what "pass" means across the 24 runs. Record it, and
re-run the grader self-test if it ever changes.

---

## 7. Decontamination

Two independent layers, both reused rather than invented.

**Layer 1 — 14-token shingle containment, against the eval holdouts.** The same machinery
CPT's dataset build and `04-cpt-sft`'s `decontam_v2.jac` use: 14-token shingles, drop a
candidate at ≥0.5 overlap against the reference set. CPT-v1's own run recorded 7 code
files dropped this way (`manifest.json:decontam.code = 7`, with the dropped paths listed
— all `raylib_shooter` / `littleX` frontend files duplicated between `jac` and
`this_is_jac`), plus 6 files excluded outright as holdout sources
(`holdout_files_excluded`: `analytics.jac`, `littlex/frontend.cl.jac`,
`raylib_shim.cl.jac`, `raylib_shooter/bench.jac`, `raylib_shooter/web/main.jac`,
`source_lexer.jac`). That precedent matters directly: **the 17-repo set contains real
cross-repo file duplication** (`jac/examples/littleX/` vs `this_is_jac/littlex/` vs the
standalone `littleX` repo; `raylib_shooter` in both `jac` and `this_is_jac`). A
file-disjoint holdout does not catch a file that exists twice under two paths — the
shingle pass is what does.

Reference set for 05's pass, in order:

1. 05's own holdout gold bodies (the primary check — nothing in the trainpool may contain
   a holdout body).
2. `04-cpt-sft`'s holdout slices, since the warm-start lines were SFT-trained on
   `04-cpt-sft/dataset/fresh/releases/sft_train.jsonl`. A 05 *holdout* task whose body
   appears in that SFT training set is contaminated for the warm lines specifically, and
   would inflate exactly the arms H1 compares.
3. `01-sft-dpo/dataset/eval_holdout/` — the legacy reference list the existing decontam
   machinery already carries.

**Layer 2 — body-level near-duplicate drop.** `build_rl_splits.jac`'s existing
`difflib`≥0.85 pass over gold bodies, holdout vs every train body (§5). Cheaper and
narrower than the shingle pass, and it catches restructured near-duplicates within the
same repo that shingles can miss.

Both layers report counts into the manifest (§6). A contamination rate above ~0 on layer
1 is a finding to investigate, not a number to quietly accept — CPT-v1's run landed at 7
files out of 741 and named every one.

---

## 8. Licensing and provenance

**Position: no new provenance surface.** All 17 repos are the same set already cloned,
gated, and used for CPT-v1's corpus (`03-cpt-only/dataset/cpt/manifest.json`) and
subsequently for `04-cpt-sft`'s tier-4 seed scrape. 05 introduces **no new
organizations, no new repos, and no new scraping** — it re-reads a corpus this project
has already vetted and shipped training runs on. That is the explicit reason the spec
chose this source set over a wider scrape.

What the reuse does and does not carry:

- **Carried:** pinned commit SHAs (one per repo, recorded in the CPT manifest), the
  parse-gate record per repo, the decontam record, and the existing clone/manifest-SHA
  infrastructure (`03-cpt-only`'s `build_cpt.py` cloning path, which
  `04-cpt-sft/docs/spec.md` §4 notes `seed_scrape.jac` already reuses).
- **Not carried, and a real gap:** the CPT manifest records SHAs, **not license
  identifiers**. No per-repo license string exists anywhere in the artifacts read for
  this design. The repos are public and predominantly `jaseci-labs`-org, but "public"
  is not a license. Therefore: 05's mining manifest adds a `licenses` key (§6), populated
  by reading each repo's `LICENSE` at its pinned SHA during Phase 1
  (`workflow.md`), with `"unverified"` for any repo that has none. This is a one-time,
  mechanical step and it is the honest place to close the gap — not a claim to make from
  a spec.
- **Attribution in artifacts:** every mined task carries `repo` + `upstream_sha` in its
  `answer` blob (§2.4), so any driver in the corpus is traceable to an exact upstream
  commit without consulting a separate index.
- **Derivation note:** drivers are *authored adaptations* of upstream files (imports
  inlined, entry point added, one body wrapped in sentinels), not verbatim copies — but
  they are unambiguously derivative work, so the license position applies to them
  identically. Record it; do not treat authoring as laundering.

---

## 9. Open items (decide at implementation, not now)

1. **Parser entry point** for the AST — jaclang API vs `jac-mcp get_ast` vs a
   `jac check`-adjacent path. Requirement is determinism + per-rollout cost, §4.2.
2. **`ast_sim` method** — tree-edit distance vs node-type bag, chosen by measured cost at
   rollout rate, §4.4.
3. **Non-trivial-body threshold** for admitting a mined hole, §3.3.
4. **Holdout target size** and the resulting minimum detectable effect, §5.
5. **Numeric-literal canonicalization scope** — how much static type information is
   available at the AST level to decide when `1` and `1.0` are the same value, §4.3.
6. **Per-repo driver authoring effort** — 635 gate-passing new files is far more than can
   be hand-authored into drivers; the mining runbook's Phase 3 (`workflow.md`) has to
   pick a per-repo sampling strategy and record it. This is the single biggest unknown
   in the corpus plan.
