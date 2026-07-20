# 04-cpt-sft Datagen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. On approval, copy this plan to `docs/superpowers/plans/2026-07-20-jacgen2-datagen.md` and commit it before Task 1.

**Goal:** Implement the full `jacgen2` data-generation pipeline specified in `model-experiments/04-cpt-sft/docs/` and produce the `fresh` SFT+DPO dataset (~12,500 SFT + ~2,500 DPO examples) end to end.

**Architecture:** New all-Jac module directory `model-experiments/01-sft-dpo/sft_dpo/jacgen2/`, importing only `run_jac`/`append_jsonl` (writer.jac) and the shingle machinery (decontam.jac) from the old pipeline. 5-tier seed sourcing → per-category generators (Opus/Fable via `by llm()`) → gates → dedup/decontam → manifest/holdouts/stats. Vertical-slice-first: one generator proven end-to-end on a pilot before the rest are built.

**Tech Stack:** jaclang 0.16.1 (project `.venv`), byllm (to install — NOT currently in venv), Anthropic API (Opus + Fable, pinned snapshots), existing `jacgen/` gate library.

## Model orchestration (user directive)

- **Sonnet** = oversight + implementation. Every implementer/reviewer subagent in this plan's execution is dispatched with `model: "sonnet"`. The controller session coordinates; Sonnet does the coding.
- **Opus** (in-pipeline, per spec §4.1) = `gen_code_gen`, `gen_trajectory`, `gen_migration`, `gen_graph_conversion`.
- **Fable** (in-pipeline, per spec §4.1) = `gen_debug`, `gen_explanation`, `gen_documentation`, `gen_dpo`, `gen_seed_review`, plus the `error_message_authoring`/`code_critique` task-type overrides.

## Global constraints (from specs — binding on every task)

- Specs are the source of truth: `model-experiments/04-cpt-sft/docs/{spec.md,datagen/spec.md,datagen/workflow.md,dpo-plan.md}`. Task briefs cite sections; implementers read the cited section before coding.
- All new tooling is Jac (`.jac`, run via `jac run` from repo root). `/ponytail:ponytail` governs implementation: no speculative abstraction, shortest working diff, one runnable self-check per non-trivial module (assert-based `with entry` check or a tiny `check_*.jac`, no frameworks).
- **Never gate generated output on `jac check`** — behavioral gate is `jac run` + pinned `expected_output`. Single narrow exception: `jac check`/lint may *detect deprecation warnings on migration inputs* (spec.md §7).
- `jac check -p` pass detection = the string `" PASSED"` in stdout+stderr, NOT exit code (verified unreliable — build_cpt.py:291 precedent).
- `JAC_RUN_TAG` env var, values `fresh|post_cptv2`, **no default** — every jacgen2 entrypoint hard-fails if unset/invalid. All jacgen2 env vars use the `JAC_` prefix.
- Example key = `(seed_id, task_type, variant_idx)` everywhere (filenames, resumability skip-checks, dedup bookkeeping). Metadata schema = spec.md §6 verbatim (`generator_model_id`, `gate_class`, `run_tag`, `seed_tier`, tri-state `test_pass`).
- Output tree: `model-experiments/04-cpt-sft/dataset/$JAC_RUN_TAG/{raw_output,clean_dataset,rejected,review,logs,releases}/...` (gitignored — verify `.gitignore` covers `model-experiments/04-cpt-sft/dataset/` in Task 1).
- LLM calls: pinned dated snapshot IDs, retry w/ exponential backoff, per-call exception capture → `rejected/` with reason (one 429 never kills a batch), per-generator call-budget kill-switch (`JAC_CALL_BUDGET`, no default → unlimited is NOT allowed; hard-fail if unset), append-only flush, call+token logging to `logs/generation/`.
- Spend gates: a pilot (~20-30 examples) per generator with **user review checkpoint** before any full-scale run; full run only after cost extrapolation from pilot logs.
- Heavy-job lock (`jms/jobs.sv.jac` `heavy_acquire`) NOT needed for generation (API-bound, no local model loads). It IS needed later for SFT training runs — out of this plan's scope.
- API key: `ANTHROPIC_API_KEY` in gitignored `model-experiments/01-sft-dpo/sft_dpo/jacgen2/.env` (directory-scoped, matching the CPT oracle's `.env` convention). Never committed, never exported globally.
- Commit after every task (Co-Authored-By footer per session convention). Never touch `model-experiments/01-sft-dpo/sft_dpo/jacgen/` (old pipeline stays frozen) or anything under `models/`/`adapters/` (never-delete rule).

## Verified ground truth (recon 2026-07-20 — implementers trust this, don't re-derive)

- `writer.jac` exact interfaces: `run_jac(jac_code: str, timeout: int = 30) -> tuple[int, str, str]` (tempdir + `cwd=d` root-store isolation); `append_jsonl(path: str, rec: dict)`. `make_sft_example` is conversion-hardcoded — do NOT reuse; `extract_jac`/`revalidate_example`/`dedup_examples` assume `messages[1]` single-assistant-turn — do NOT reuse for trajectory/prose.
- `decontam.jac` reusables: `shingles(code: str, n: int = 14) -> list[str]`, `build_training_shingles(paths: list[str]) -> dict`, `is_contaminated(py: str, train: dict, threshold: float = 0.5) -> bool`; its `extract_python` only understands ```` ```python ```` fences.
- `dedup.jac` reusables: `code_tokens(code)`, `rouge_l(a, b)`; `dedup_examples` is O(n²) raw-content — do NOT use above pilot scale.
- **byllm and anthropic SDK are NOT installed** in `.venv` (only jaclang 0.16.1, jac-client, jac-desktop). In-repo byllm precedent (worktree oracle): `import from byllm.lib { Model }`, `glob llm = Model(model_name="...");`, `def f(...) -> T by llm(...);` — only OpenAI model strings so far; Anthropic model strings unproven → Task 1 proves them.
- `build_cpt.py` does NOT clone repos (manual precondition); reusable pieces: `CODE_REPOS` list (17 jaseci-labs repos, build_cpt.py:46-52), `repo_sha` via `git -C <p> rev-parse HEAD`, fence-aware `md_chunks`/`split_paragraphs` (Python — port ~30 lines to Jac), exclusion precedent (jac_ml_studio, archived-jaseci, forks; `jac/tests/**`, `jaclang/**` internals).
- Tier 3 source is **local**: `.venv/lib/python3.14/site-packages/jaclang/` ships 435 `.jac` files (includes intentionally-broken fixtures — filter by `jac check -p`/`jac run` pass).
- Tier 5 counts: 55 `.jac` in `jms/`, 157 in `model-experiments/` (excl. `dataset/`).
- Tier 1-2 source: the jac-mcp content bundle (50 md pages, ~800-950 jac fences + 9 examples) — Task 5 locates it on disk (search `~/.local/share/uv/tools/*/…/jac_mcp/content/` and venv site-packages) rather than calling MCP tools from batch scripts.
- Existing SFT record shape (live sample verified): `{"messages":[{role:user},{role:assistant}], "meta":{...}}` with fenced code in content and `expected_output` = exact stdout.

---

## Phase 0 — Foundations

### Task 1: Environment bootstrap + model pinning smoke test

**Files:** Create `model-experiments/01-sft-dpo/sft_dpo/jacgen2/{.env(untracked),models.json,check_llm.jac}`. Modify `.gitignore` (ensure `model-experiments/04-cpt-sft/dataset/` + `jacgen2/.env` covered).
**Produces:** installed `byllm`; `models.json` = `{"opus": "<exact dated snapshot>", "fable": "<exact dated snapshot>", "pinned_at": "<date>"}`; proof both models answer via `by llm()`.

- [ ] `. .venv/bin/activate && uv pip install byllm` (fallback `pip install byllm`). Record installed version.
- [ ] Resolve exact dated snapshot IDs for Opus 4.8 and Fable 5 (query `https://api.anthropic.com/v1/models` with the key, or take the alias-resolved ID from a probe response). Write `models.json`.
- [ ] Write `check_llm.jac`: two `Model(model_name=...)` globs (try bare `claude-…` ID first; if byllm/litellm requires it, `anthropic/claude-…` prefix — record which worked in models.json as `"prefix"`), one `def ping(word: str) -> str by llm();` per model, `with entry { assert "pong" in ping("say pong"); }` for both. `.env` loader = 6-line KEY=VAL parse + `os.environ.setdefault` (no python-dotenv dep).
- [ ] Run `jac run …/check_llm.jac` → both asserts pass. This is the go/no-go on the whole `by llm()` approach: if Anthropic models can't be driven through byllm, STOP and escalate (fallback = `uv pip install anthropic` + thin Jac wrapper; decision needs user).
- [ ] Commit (models.json + check_llm.jac + .gitignore; never .env).

### Task 2: `common.jac` — run-tag guard, paths, keys

**Files:** Create `jacgen2/common.jac`, `jacgen2/check_common.jac`.
**Produces:** `require_run_tag() -> str` (hard-fail unless `fresh|post_cptv2`); `data_root(tag: str) -> str` = `model-experiments/04-cpt-sft/dataset/<tag>`; `out_path(tag, area, category) -> str`; `example_key(seed_id: str, task_type: str, variant_idx: int) -> str` (`"<seed_id>__<task_type>__<variant_idx>"`); `load_env()`; `existing_keys(jsonl_path) -> dict` (set of `meta.key` values for resumability skip).

- [ ] Write module (globals typed; semicolons — jaclang 0.16.1 checker quirks per studio-overhaul memory).
- [ ] `check_common.jac`: asserts — missing `JAC_RUN_TAG` raises; bad value raises; `example_key` stable; `existing_keys` on a temp 2-line jsonl returns both keys.
- [ ] `JAC_RUN_TAG=fresh jac run …/check_common.jac` → all asserts pass. Commit.

### Task 3: `llm.jac` — dual-model wrappers with retry/budget/logging

**Files:** Create `jacgen2/llm.jac`, `jacgen2/check_llm_wrap.jac`.
**Consumes:** `models.json`, `common.load_env`.
**Produces:** `glob llm_opus`, `glob llm_fable` (Models from models.json); typed by-llm defs used by generators, each with a plain-Jac retry wrapper. Core surface:
  - `gen_text_opus(prompt: str) -> str by llm_opus();` / `gen_text_fable(prompt: str) -> str by llm_fable();` (generic single-string calls — generators build their own prompts; no per-task-type schema explosion, ponytail)
  - `call_with_retry(fn_name: str, prompt: str, model: str) -> dict` → `{"ok": bool, "text": str, "err": str}`; 3 attempts, sleep 2/8/32s; any exception on final attempt returns `ok=False` (caller routes to `rejected/` — never raises out of a batch)
  - budget: `glob CALLS_MADE: int = 0;` incremented per attempt; hard-stop raise once `> int(os.environ["JAC_CALL_BUDGET"])`; `require_budget()` fails fast if env unset
  - logging: every call appends `{ts, model, model_id, fn, ok, attempt, prompt_chars, resp_chars}` to `logs/generation/calls_<tag>.jsonl` via `append_jsonl`

- [ ] Write module (import `Model` per Task 1's proven convention; read snapshot IDs from models.json — never hardcode).
- [ ] `check_llm_wrap.jac`: budget=2 then 3 calls → third raises; a call with a bogus model name returns `ok=False` (not an exception); log file gains rows. Run with `JAC_RUN_TAG=fresh JAC_CALL_BUDGET=2`. Commit.

### Task 4: `gate.jac` — v2 gates + record builder

**Files:** Create `jacgen2/gate.jac`, `jacgen2/check_gate.jac`.
**Consumes:** `writer.jac` `run_jac`/`append_jsonl` (import from `jacgen`), `common.jac`.
**Produces:**
  - `run_jac_project(files: dict[str, str], main_file: str, timeout: int = 60) -> tuple[int, str, str]` — tempdir, write all files, `jac run <main_file>` with `cwd=tempdir` (same isolation trick as writer.jac:51-55)
  - `behavioral_gate(jac_code: str, expected_output: str) -> tuple[bool, str]` — `run_jac`, rc==0 AND stdout.strip()==expected.strip()
  - `make_example_v2(category: str, task_type: str, complexity: str, messages: list[dict], gate_class: str, test_pass: any, seed_id: str, seed_tier: int, variant_idx: int, generator: str, generator_model_id: str, extra_meta: dict = {}) -> dict` — full spec.md §6 schema (`run_tag` from `require_run_tag()`, `generation_date` iso now, `source_prompt_version`/`validator_version` from globs `PROMPT_VER="prompt-v2.0"`, `VALIDATOR_VER="validator-v2.0"`, `dataset_version="jac-synth-v2.0.0"`, `id` = example_key, `compiler_pass` per gate result)
  - `extract_payload(rec: dict, category: str) -> str` — jac fence from last assistant turn for code categories; full assistant text for prose categories; concatenated assistant turns for trajectory
- [ ] Write module + `check_gate.jac`: `behavioral_gate("with entry { print(1+1); }", "2")` passes and `…"3"` fails; `run_jac_project` with a 2-file include pair compiles; `make_example_v2` record round-trips through `json.dumps` and carries every §6 field. Run, commit.

## Phase 1 — Seed sourcing (fresh build only; pool frozen after)

### Task 5: `seed_pool.jac` part 1 — tiers 1, 2, 3, 5 (all local, no LLM)

**Files:** Create `jacgen2/seed_pool.jac`, `jacgen2/mdfence.jac` (fence-aware md chunk/fence extraction — port `md_chunks`/`split_paragraphs`/fence-regex logic from `build_cpt.py:67-146` to Jac, ~40 lines), `jacgen2/check_seed_pool.jac`.
**Produces:** `model-experiments/04-cpt-sft/dataset/shared/seed_pool.jsonl` — one record per seed: `{seed_id (sha1-12 of source+content), seed_tier, origin (e.g. "doc:cheatsheet#12", "jaclang:tests/…", "self:jms/models.sv.jac#fn"), jac_code, gate_class ("behavioral"|"compile_only"), expected_output (str|null), domains (list[str] — task_type hints), doc_chunk (str|null, tier-2 prose context for explanation)}` — plus `dataset/shared/doc_chunks.jsonl` (all ~500-token doc chunks for `gen_explanation`, with chunk_id + heading + text).

- [ ] Locate jac-mcp content bundle on disk (`find ~/.local/share/uv/tools -type d -name "content" -path "*jac_mcp*"` then venv site-packages as fallback); glob `*.md` + example dirs. Hard-fail with a clear message if not found (escalate — do not silently skip tiers 1-2).
- [ ] Tier 1: decompose example apps into per-file seeds; drop `ui/*.cl.jac` shadcn boilerplate; tolerate missing/unreadable examples.
- [ ] Tier 2: `mdfence.jac` fence extraction over all md pages → jac-fenced blocks ≥3 lines; store surrounding chunk as `doc_chunk`. Also emit `doc_chunks.jsonl` (heading-split, ~500-token cap, per datagen/spec.md §3).
- [ ] Tier 3: walk `.venv/lib/python3.14/site-packages/jaclang/**/*.jac`; keep only files passing `jac check -p` (string `" PASSED"` check) — this drops the intentionally-broken fixtures; cap file size 200 lines.
- [ ] Tier 5: walk `jms/**/*.jac` + `model-experiments/**/*.jac` (excluding `*/dataset/*`, `jacgen2/` itself); split multi-archetype files into archetype/walker/def-level seeds where a simple top-level split is clean, else whole-file seed.
- [ ] For every seed: attempt `run_jac(code)`; rc==0 → `gate_class="behavioral"`, pin `expected_output=stdout`; else `compile_only` if `jac check -p` passes; else drop (log count).
- [ ] Assign `domains` by origin heuristics (path/doc-page name → task_type hints per datagen/spec.md §1 seed-source column; multi-tag fine; `["core_language_basics"]` default).
- [ ] Seed-level dedup: exact sha1 first; then bucket by `(len(code)//200, first token)` and `rouge_l ≥ 0.85` within bucket (import from `jacgen.dedup`).
- [ ] Print per-tier accepted/dropped counts (this IS the supply measurement datagen/spec.md §0.1 demands). `check_seed_pool.jac` asserts: pool non-empty, every record has all fields, no duplicate seed_ids, ≥1 seed from each of tiers 1/2/3/5. Run, commit.

### Task 6: `seed_scrape.jac` + `gen_seed_review.jac` — tier 4

**Files:** Create `jacgen2/seed_scrape.jac`, `jacgen2/gen_seed_review.jac`, `jacgen2/repos.json` (the repo list — start from build_cpt.py's 17 `jaseci-labs` CODE_REPOS verbatim; same exclusions: jac_ml_studio, archived-jaseci, forks).
**Consumes:** `llm.jac` (`gen_text_fable` + `call_with_retry`), `common.jac`.
**Produces:** scratchpad clones + `dataset/shared/scrape_manifest.json` (repo→SHA via `git -C <p> rev-parse HEAD`); `dataset/shared/tier4_candidates.jsonl`; `dataset/shared/tier4_reviewed.jsonl` (accepted, with `review_note`); accepted records appended into `seed_pool.jsonl` with `seed_tier=4`.

- [ ] `seed_scrape.jac`: clone each repo (shallow, `git clone --depth 1`) into the session scratchpad dir (env `JAC_SCRAPE_DIR`, hard-fail if unset); skip repos already cloned; record SHAs; collect `**/*.jac` minus `jac/tests/**`+`jac/jaclang/**` (CPT precedent); parse-gate `jac check -p` (" PASSED" string); emit candidates with provenance `repo@sha:path`.
- [ ] `gen_seed_review.jac`: per candidate one Fable call. Prompt (verbatim core): *"You are a strict code reviewer for Jac training data. Reject code that: (1) reinvents Jac stdlib or a native language feature, (2) has speculative abstraction — interfaces with one implementation, config for constants, (3) is clearly inefficient (O(n²) where O(n) is trivial, redundant recomputation, dead paths), (4) solves a graph-shaped problem in Python-shaped Jac (dict/list graph simulation instead of nodes/edges/walkers). Otherwise accept. Reply exactly `ACCEPT: <one-line reason>` or `REJECT: <one-line reason>`.\n\n```jac\n<code>\n```"* Parse verdict; anything unparseable → reject with reason `review_unparseable`.
- [ ] Decontam accepted candidates vs eval holdouts (`build_training_shingles` over `model-experiments/01-sft-dpo/dataset/eval_holdout/*.jsonl`, `is_contaminated` with a jac-fence extractor, threshold 0.5) — at seed time, per datagen/spec.md §8.4.
- [ ] Append accepted to `seed_pool.jsonl` (same record shape, `expected_output` pinned via `run_jac` where runnable); re-run the Task 5 dedup pass across the merged pool. Print tier-4 candidate/accept/reject counts + review-call count (the open cost line item from datagen/workflow.md §5).
- [ ] Runnable check: run with `JAC_CALL_BUDGET=25` against ONE small repo first (e.g. `this_is_jac`) and eyeball 5 verdicts before unleashing the full list. Commit (repos.json + modules; manifest/jsonl are gitignored data).

### Task 7: Pool finishing — deprecated inventory, idea bank, freeze prep

**Files:** Create `jacgen2/data/deprecated_inventory.jsonl` (tracked — hand-authored, from datagen/spec.md §7: W0061 paren-filter `(?:…)`, W0062 `root()`, W0063 JSX spread, hard-fail `import:py …`/`include:jac`, `jac js` alias, + whatever `jac-scaffold` guide lists — each row `{pattern_id, deprecated_form, current_form, detect ("check_warning"|"hard_fail"), example_before, example_after}` with REAL before/after snippets, verified: `example_after` passes `run_jac`, `example_before` either fails `run_jac` or triggers the check-warning); `jacgen2/data/app_ideas.jsonl` (tracked — 40 hand-written one-paragraph app ideas for `schema_design`; graph-shaped domains: library/loans, org-chart, delivery routing, social feed, course-prereqs, inventory, chat threads, tournament bracket, …); `jacgen2/freeze_pool.jac` (sha256 of seed_pool.jsonl + doc_chunks.jsonl + both data files → `dataset/shared/pool_hashes.json`; verify mode via `JAC_VERIFY_POOL=1` hard-fails on mismatch).
- [ ] Author both data files (Sonnet implementer writes them; deprecated inventory is small — if it can't support ~500 migration examples at 2-5 patterns/file, print the honest ceiling; buffer absorbs shortfall per datagen/spec.md §7).
- [ ] Verify every inventory row's `example_before`/`example_after` behavior claim by actually running them; print the confirmed-pattern count.
- [ ] `freeze_pool.jac` write + verify modes work. Commit.
- [ ] **USER CHECKPOINT: report per-tier supply numbers + confirmed migration-inventory size + tier-4 rejection rate → user approves scale targets (or cuts them per §0) before any generation spend.**

## Phase 2 — Vertical slice

### Task 8: `gen_code_gen.jac` + pilot

**Files:** Create `jacgen2/gen_code_gen.jac`.
**Consumes:** `seed_pool.jsonl`, `llm.jac` (Opus for main; Fable for `error_message_authoring`), `gate.jac`, `common.existing_keys`.
**Produces:** `clean_dataset/code_gen/examples.jsonl`, `rejected/code_gen/rejected.jsonl` — records per spec.md §6; the fan-out loop other generators copy.

- [ ] Implement the datagen/workflow.md §3 loop: for each seed × applicable task_type (from `domains`) × `variant_idx < k` (`JAC_FANOUT_K`, pilot=2): skip if key in `existing_keys`; Opus reverse-instruction prompt (verbatim core): *"Here is a canonical Jac program. Write the natural-language task a developer would have been given, for which this program is the correct solution. Vary register per variant_idx (0=plain user request, 1=spec-style requirement, 2=terse expert ask). Do not describe the code line-by-line; describe the TASK. Reply with only the instruction text.\n\n```jac\n<code>\n```"*; variant_idx≥3 → code-mutating variant first (Opus: extend-by-one-feature / parameter change), re-gate via `behavioral_gate` re-pinning expected_output; gate per seed's `gate_class`; dedup vs kept (bucketed rouge_l on instruction+code, same-seed exemption per §0.2); decontam; `append_jsonl`.
- [ ] `error_message_authoring` branch: seeds = deprecated/diagnostic-triggering snippets; Fable call; `prose_lexical` gate = message mentions the failing symbol (substring check).
- [ ] Design-and-prose types: `schema_design` from `app_ideas.jsonl` (Opus forward-gen; gate = schema compiles + a smoke walker traverses — wrap output in `run_jac_project` if multi-file); `syntax_migration` from `deprecated_inventory.jsonl` (gate per §7 two-part rule).
- [ ] Pilot: `JAC_RUN_TAG=fresh JAC_CALL_BUDGET=80 JAC_FANOUT_K=2 JAC_PILOT_SEEDS=12 jac run …/gen_code_gen.jac` → expect ~20-30 clean examples. Print acceptance rate + calls used; report cost extrapolation.
- [ ] Commit module.
- [ ] **USER CHECKPOINT: user reads actual pilot examples** (`head` the jsonl; spot-run 3 examples' code through `jac run` live) — test-before-claiming. No further generators until approved.

## Phase 3 — Remaining generators (each: implement → self-check → pilot ~20 → log cost; Sonnet implements, review per subagent-driven flow; commit each)

### Task 9: `gen_debug.jac`
Per datagen/spec.md §2: seed × applicable bug type (domain×bug table); Fable injects exactly one bug + one-line symptom; dual gate (buggy fails `behavioral_gate`/differs from expected_output; fixed passes); `auth_leak`/`stale_client_state` rows → `compile_only` + second Fable critique call confirming the bug is real (`test_pass=null`); emits `raw_output/debug/buggy_variants.jsonl` (`{seed_id, bug_type, buggy_code, symptom, clean_code}`) — **declared interface, ordering prerequisite for Tasks 12/15**. `code_critique` task_type: Fable critique of known-buggy variants, gate = flags the injected bug (bug_type keyword match) + references a real symbol. `cross_boundary_debug`: paired sv/cl seeds via `run_jac_project`.

### Task 10: `gen_explanation.jac`
`doc_chunks.jsonl` → Fable quiz Q&A per 5 formats (datagen/spec.md §3); groundedness = chunk-term overlap score logged to meta (`overlap` float; accept threshold `JAC_GROUND_MIN`, pilot-calibrated, start 0.25); `compare_contrast` may carry two chunks.

### Task 11: `gen_documentation.jac`
Code-bearing seeds → Fable docs per 3 formats; generation prompt REQUIRES backticks around every code symbol; gate = every backticked token occurs in seed code, allowlist `{root, walker, node, edge, jac.toml, str, int, list, dict, bool, float}`.

### Task 12: `gen_trajectory.jac`
One Opus call → whole 3-6-turn `messages` list (JSON-array output, parsed + validated: alternating roles, final assistant turn has jac fence); gate final turn only (`behavioral_gate` or `run_jac_project`); `debug_session` seeds from `buggy_variants.jsonl` (hard-fail if missing → run Task 9 first).

### Task 13: `gen_migration.jac`
Compose 2-5 inventory patterns into one deprecated file (Opus), produce migrated file; gate = migrated passes `run_jac`; original fails `run_jac` OR `jac check` output contains a W006x code (the narrow exception); reject silent-pass pairs. Target honestly capped by Task 7's confirmed inventory count.

### Task 14: `gen_graph_conversion.jac` (one-time, fresh only)
Scenario bank seeded from `model-experiments/01-sft-dpo/sft_dpo/jacgen/graph_data/train.json` problem shapes + Opus-proposed new shapes (dedup vs bank); Opus authors graph-shaped Python + idiomatic walker/node/edge Jac; gate = **both** run and outputs match on same inputs (python via `subprocess python3`, jac via `run_jac`); output to `dataset/shared/graph_conversion_growth.jsonl` (feeds Task 16 snapshot; NOT per-tag).

### Task 15: `gen_dpo.jac`
5 axes per dpo-plan.md §2: idiomatic (Opus? NO — Fable per §4.1; chosen=idiomatic seed, rejected=Fable-Pythonified variant, both compile); graph_native (Fable, rejected=dict/list shoehorn); correctness/auth_security/typing assembled free from `buggy_variants.jsonl` (no LLM calls); output `{prompt, chosen, rejected, axis, …}` per dpo-plan.md §6, compatible with `build_dpo_splits.jac`.

## Phase 4 — Assembly + full run

### Task 16: `decontam_v2.jac` + `dedup_v2.jac`
`decontam_v2`: per-category extractors (jac fence / instruction text / doc chunk / trajectory-concat via `gate.extract_payload`) feeding `shingles`/`is_contaminated` vs old holdouts + (once carved) new ones; audit report json. `dedup_v2`: cross-category pass at scale — exact-hash, then bucket `(category, len//200, first-token)`, `rouge_l` in-bucket, same-`seed_id` exemption; report dropped counts per category (no silent truncation).

### Task 17: `build_manifest_v2.jac` + `holdout_v2.jac` + `dataset_stats_v2.jac`
`build_manifest_v2`: weights table from datagen/spec.md (36/16/10/10/10/6/4/8 buffer-realloc), conversion slice from snapshot (Task 18), holdout-id exclusion for BOTH tags, → `releases/sft_train.jsonl` + `releases/dpo_train.jsonl`. `holdout_v2`: carve ≥100/category from fresh build (seed-provenance recorded; disjoint from training by id; append to decontam reference). `dataset_stats_v2`: composition by category/task_type/seed_tier/generator + call/token rollup from `logs/generation/` + fresh-vs-post diff mode (`JAC_DIFF_TAGS=1`).

### Task 18: Conversion snapshot + full fresh build
- [ ] Snapshot: copy `model-experiments/01-sft-dpo/dataset/conversion/{sft.jsonl,sft_auto.jsonl}` records + `graph_conversion_growth.jsonl` → `dataset/shared/conversion_slice.jsonl`; task_type backfill (`python_to_jac_function`/`python_to_jac_graph`); row-count + sha256 into `pool_hashes.json`.
- [ ] **USER CHECKPOINT: cost extrapolation from all pilots → user approves full-run budget.**
- [ ] Full run, order per datagen/workflow.md §2 (`gen_debug` first), each generator with its production `JAC_CALL_BUDGET`; resumable by construction (re-invoke on failure).
- [ ] `freeze_pool.jac` freeze; `holdout_v2`; `build_manifest_v2`; `dataset_stats_v2`; `decontam_v2` audit.
- [ ] Final report to user: counts vs targets, per-category acceptance rates, spend, honest gaps. Update memory + `docs/README.md` status line. Commit.

**Parked (explicitly NOT this plan):** `post_cptv2` build (same commands, `JAC_RUN_TAG=post_cptv2`, pool-hash verify instead of rebuild), Arms A/B/C training + eval battery (workflow.md §3-§6), incumbent eval column.

## Verification (end-to-end)

1. Every module's `check_*.jac` passes under `JAC_RUN_TAG=fresh`.
2. Pilot artifacts human-reviewed at Tasks 7/8/18 checkpoints (real examples read, 3 spot-run live).
3. `releases/sft_train.jsonl`: every record validates against spec.md §6 (a 20-line validator loop inside `dataset_stats_v2`), no holdout ids present, `dataset_stats_v2` composition within ±20% of weight targets (buffer explains drift).
4. `decontam_v2` audit reports 0 contaminated records in releases.
5. Random-sample re-validation: 50 random `behavioral`-class examples re-run through `behavioral_gate` → 100% pass (mirrors `verify_dataset.jac` discipline).
