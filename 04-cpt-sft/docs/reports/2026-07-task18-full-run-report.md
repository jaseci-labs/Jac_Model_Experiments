# 04-cpt-sft Datagen Pipeline — Full Build Report

**Date:** 2026-07-20 through 2026-07-22
**Scope:** Implementation of the entire `jacgen2` SFT+DPO data-generation pipeline (18-task plan) plus a full production-scale run of every generator, end to end.
**Result:** `sft_train.jsonl` = **9,608 examples**, `dpo_train.jsonl` = **826 preference pairs**, both holdout-clean and 0% contaminated.

---

## 1. What this was

The goal was to build a from-scratch data-generation pipeline for the `04-cpt-sft` phase of the Jac model finetuning project — a 7-category SFT dataset (`code_gen`, `debug`, `explanation`, `conversion`, `trajectory`, `documentation`, `migration`) plus a 5-axis DPO preference dataset — and then actually run it at real production scale, not just pilot it.

The design (from `docs/spec.md`, `docs/datagen/spec.md`, `docs/datagen/workflow.md`, `docs/dpo-plan.md`) was written and approved before this session; this session's job was implementation (`docs/superpowers/plans/2026-07-20-jacgen2-datagen.md`, an 18-task plan).

## 2. The architecture pivot (the first real problem)

The original plan called for generation via `byllm`/direct Anthropic API calls (pinned Opus/Fable snapshot IDs, `Model(model_name=...)` + `by llm()`). Partway into Task 1, it became clear there is no `ANTHROPIC_API_KEY` in this environment — the project never had one, and installing/obtaining one wasn't an option.

**Fix:** pivoted to a **batch-handoff architecture** instead: a Jac script (`batch.jac`) writes a `pending_batch.jsonl` file of prompts ("prepare" phase); a Claude Code Agent (dispatched with `model: "opus"` or `model: "fable"`, matching the project's per-category model assignment) is manually dispatched to read the pending file and fill in a matching `responses_batch.jsonl`; a second Jac phase ("collect") joins pending+responses, gates, and persists. This is why every generator run in this project involved dispatching real Claude Code subagents as the "LLM calls," rather than a fully automated script.

`llm.jac` (the original direct-API module) was built in Task 3 before this pivot happened and is left in the repo unused — harmless, not deleted, since removing already-reviewed code wasn't in scope.

## 3. Pipeline build (Tasks 1–17)

All of `model-experiments/01-sft-dpo/sft_dpo/jacgen2/` was built fresh this session, one task at a time, each with an implementer pass + independent review pass + fix pass where needed. In build order:

| # | File(s) | What it does |
|---|---|---|
| 1 | `models.json`, `check_llm.jac` | Model IDs (superseded by the pivot, kept for record) |
| 2 | `common.jac` | Run-tag guard, path helpers, `example_key`, `existing_keys` (idempotency) |
| 3 | `llm.jac` | Direct-API wrapper (unused after pivot, left in place) |
| 4 | `gate.jac` | `run_jac_local`, `run_jac_project`, `behavioral_gate`, `make_example_v2` (the spec's full record schema) |
| 5 | `seed_pool.jac`, `mdfence.jac`, `dedup2.jac` | 4-of-5 seed tiers (jac-mcp examples/docs, jaclang's own repo, this repo's own code) → `seed_pool.jsonl` (1373 seeds) + `doc_chunks.jsonl` (3311 chunks) |
| 6 | `seed_scrape.jac`, `gen_seed_review.jac` | Tier 5 (public multi-repo Jac scrape, Fable-reviewed for idiom quality) |
| 7 | `data/deprecated_inventory.jsonl`, `data/app_ideas.jsonl`, `freeze_pool.jac` | Migration-pattern inventory (4 confirmed patterns), 40 app ideas, pool freeze/verify |
| 8 | `gen_code_gen.jac` | Reverse-instruction generation (code → task description) |
| 9 | `gen_debug.jac` | Bug injection across 6 bug-type/domain pairs |
| 10 | `gen_explanation.jac` | Doc-grounded quiz Q&A, 5 formats |
| 11 | `gen_documentation.jac` | Docstring/API-reference/module-overview generation |
| 12 | `gen_trajectory.jac` | Multi-turn conversations, 5 scenario types |
| 13 | `gen_migration.jac` | Deprecated-pattern-combo migration |
| 14 | `gen_graph_conversion.jac` | Forward-authored Python↔Jac graph/tree problem pairs |
| 15 | `gen_dpo.jac` | 5-axis DPO pairs (3 free, 2 LLM-generated) |
| 16 | `decontam_v2.jac`, `dedup_v2.jac` | Cross-category contamination/near-dup audits |
| 17 | `holdout_v2.jac`, `build_manifest_v2.jac`, `dataset_stats_v2.jac` | Holdout carving, release assembly, composition stats |

Each of these went through a pilot run first (20–30 examples) with real generated data, spot-checked by a human-equivalent reviewer pass before being trusted. The pilot totals (before the full-scale run) were small on purpose — code_gen 30, debug 4, documentation 25, explanation 24, migration 10, trajectory 16, graph_conversion 10, dpo 17.

## 4. Problems found and fixed during the build (Tasks 1–17)

These were real bugs caught by the review process, not hypothetical risks:

- **`existing_keys()` schema bug** — was reading a nested `rec["meta"]["id"]` path that doesn't exist in real records (which are flat `rec["id"]`). Would have `KeyError`'d on every real record. Fixed in Task 8.
- **Cross-directory Jac imports fail at runtime** even though `jac check` passes them — worked around by keeping local copies of shared logic inside `jacgen2/` instead of importing across directories.
- **Wrong venv resolution** — a bare `"jac"` on `PATH` resolves to a different, stale venv. Fixed by hardcoding the absolute path everywhere: `/Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac`.
- **`with entry { }` executes on *import*, not just direct invocation** — a real Jac-language behavior that would have caused pipeline scripts to run as side effects of being imported by other scripts. Fixed by using `with entry:__main__ { }` everywhere.
- **`gen_debug.jac`'s auth_leak gate** had no guard against being run on a seed with no real auth surface, and was silently missing a documented critique pass — fixed with a guard + an honest docstring disclosing the gap rather than building the full (out-of-scope) critique machinery.
- **`gen_migration.jac`'s second gate check** (re-verifying no deprecation warnings remain after migration) was mislabeled in code as "nice-to-have" while actually functioning as a real rejection gate — this contradicted the spec's stated single-exception wording. Fixed by relabeling it honestly and formalizing it as a second, documented exception in both spec files.
- **`gen_graph_conversion.jac`** was missing a near-duplicate check against the existing 31-example graph tier (an explicit spec requirement) — added, using the existing `dedup2.jac` rouge_l machinery.
- **`gen_dpo.jac`'s Part A** used a static bug-type lookup instead of a real compile check for one field (`compiler_pass_rejected`) — coincidentally correct for the data that existed at the time, but would have silently mislabeled a different bug type later. Fixed to use a real gate call.
- **`gen_dpo.jac`'s Part B** wasn't carrying enough metadata to do a real behavioral re-check on the "chosen" side — fixed, then all already-persisted pilot records were re-verified against the fixed logic at no new LLM cost (all survived).
- **The big one — `seed_pool.jsonl`'s gate_class bug**: 813 of 1100 seeds tagged `gate_class="behavioral"` had an *empty* `expected_output`, which silently made every behavioral gate check on them a no-op (a truthiness check on an empty string just always "passes"). This was root-caused during Task 9's investigation into `gen_debug`'s surprisingly low pilot accept rate (4/25), tracked as a priority fix, and actually fixed before the full-scale run: a one-off patch flipped those 813 seeds' `gate_class` to `compile_only`. This alone took `gen_debug`'s real accept rate from ~16% to ~52%.
- **A holdout-manifest design bug (Task 17, the most serious one)**: the holdout exclusion list was originally tag-scoped (per `fresh`/`post_cptv2` build) instead of shared/frozen like `seed_pool.jsonl`. This would have silently let held-out evaluation examples leak into `post_cptv2` training data with only a soft warning — a real, invisible eval-integrity bug. Fixed by moving the holdout manifest to `dataset/shared/holdouts/` and hard-failing (not warning) if it's ever missing or built under the wrong tag.
- **95% of the assembled dataset (the legacy `conversion` records) was missing 11 of 17 required schema fields** — backfilled with honestly-labeled legacy placeholder values, and independently verified that the backfilled `gate_class="behavioral"` claim is genuinely true for all 1920 legacy records (checked against their real historical pass/fail data, 100% match).
- **One incident during review**: a reviewer subagent, working on an unrelated task, ran `rm -rf .playwright-mcp` outside its assigned scope. This was flagged to the user immediately. The directory was untracked (Playwright browser-automation cache) with no git history, so nothing was recoverable, but nothing of value was actually lost — it's regenerable cache, not project data.

## 5. The full production-scale run (this session's second half)

Once the pipeline was proven end to end at pilot scale, the plan's own text called this out as a "USER CHECKPOINT: cost extrapolation from all pilots → user approves full-run budget" moment. An initial cost estimate (~4,000 more Agent dispatches) was given to the user and was **wrong** — it was based on a stale log file (`calls_fresh.jsonl`) belonging to the now-unused direct-API `llm.jac` path, not the real batch-dispatch mechanism actually in use, where one Agent dispatch fills a whole batch of prompts at once. The real cost was far lower. This was caught and corrected transparently before committing to the scale-up.

With that corrected picture, the user explicitly chose "full production run" — not a scaled-down interim size.

### What actually happened, category by category

| category | pilot | **final** | accept rate | notes |
|---|---:|---:|---:|---|
| `code_gen` | 30 | **2,306** | 99.65% | Ran the full uncapped eligible set (2,314 seed×task_type pairs) for the 6 already-implemented task types (of spec's 34 total — the other 28 remain documented future work, same as at pilot time). 8 rejections were all real, known non-determinism (wall-clock timers, tempfile paths), not quality failures. |
| `debug` | 4 | **210** | 52% (was 16% before the seed_pool fix) | Full eligible-pairs volume (363 items) after fixing the seed_pool `gate_class` bug above. |
| `documentation` | 25 | **1,373** | 100% | Every one of the 1,373 eligible seeds, independently re-verified for backtick/symbol integrity — 0 violations. |
| `explanation` | 24 | **2,556** | 97% | Full 2,678-chunk eligible pool (of 3,311 total doc chunks, 633 too short to use). All rejections were real groundedness-gate failures. |
| `migration` | 10 | **11** | 100% | Hit its **true ceiling** — 11 is literally every possible combination (`C(4,2)+C(4,3)+C(4,4)`) of the 4 confirmed deprecated patterns that exist. No more supply exists to generate from without inventing new deprecated syntax patterns, which was explicitly out of scope. |
| `graph_conversion` | 10 | **23** | 100% | Finished the original 15-scenario bank, then invented 8 new scenarios (height-balance check, path counting, tree isomorphism, k-th-ancestor walk, bipartite coloring, critical-path DAG scheduling, tree flatten, max path sum) — every one self-verified by actually running both the Python and Jac solutions and diffing stdout. |
| `trajectory` | 16 | **1,227** | ~79% | The hardest category to finish — see the stalling problem below. Final breakdown: `code_review_session` 383, `build_from_scratch` 285, `add_feature_to_existing` 202, `debug_session` 194, `refactor_session` 163. |
| `dpo` (free axes) | 7 | **214** | n/a (no LLM call) | correctness 190 new, auth_security 16 new, typing 1 new — assembled directly from `gen_debug`'s much bigger output, zero additional cost. |
| `dpo` (LLM axes) | 10 | **612** | ~89% | idiomatic 289, graph_native 323, over 686 attempted pairs. |
| `conversion` (snapshot) | 1,930 | **1,943** | n/a | Legacy sft.jsonl+sft_auto.jsonl (1,920 records) plus all 23 graph_conversion examples. |

### Final assembled release

- `releases/sft_train.jsonl`: **9,608 examples** (code_gen 2,301; debug 209; documentation 1,369; explanation 2,552; migration 9; trajectory 1,225; conversion 1,943 — each count is the accepted total minus its holdout exclusions)
- `releases/dpo_train.jsonl`: **826 pairs** (correctness 194, auth_security 19, typing 1, idiomatic 289, graph_native 323)
- Holdout set: **18 ids** across 6 categories (code_gen 5, debug 1, documentation 4, explanation 4, migration 2, trajectory 2) — left at its original pilot-era size deliberately, see below.
- `decontam_v2.jac`: **0 contaminated** out of 8,532 checked records.
- `pool_hashes.json`: 5 shared files re-frozen and verified clean (`seed_pool.jsonl`, `doc_chunks.jsonl`, `deprecated_inventory.jsonl`, `app_ideas.jsonl`, `conversion_slice.jsonl`).

### Honest gap: dataset composition vs. spec's target weights

The spec's target weights (code_gen 36%, debug 16%, explanation 10%, conversion 10%, trajectory 10%, documentation 6%, migration 4%, buffer 8%, all against a ~12,500-example target) don't match the real, achieved composition:

| category | target % | actual % | why |
|---|---:|---:|---|
| code_gen | 36% | 23.95% | Real supply is capped by only 6-of-34 task types being implemented |
| debug | 16% | 2.18% | Real eligible-pairs supply (363) is genuinely small even after the seed_pool fix |
| explanation | 10% | 26.56% | Over-represented — doc_chunks pool (2,678 eligible) is larger relative to other categories' real supply |
| conversion | 10% | 20.22% | Over-represented — dominated by the large pre-existing legacy sft.jsonl/sft_auto.jsonl tiers |
| trajectory | 10% | 12.75% | Close to target |
| documentation | 6% | 14.25% | Over-represented relative to its small target share |
| migration | 4% | 0.09% | **Structurally capped** — true ceiling of 11 possible examples total |
| buffer | 8% | 0% | Not implemented — no buffer-reallocation logic exists |

This isn't a bug — every one of these numbers is real supply-constrained or scope-constrained, and each constraint was independently investigated and documented at the time it was found (not just asserted). Hitting the spec's exact weight targets would require implementing code_gen's other 28 task types and finding/inventing far more debug-eligible seed material — both explicitly out of this session's scope.

## 6. Problems during the full-scale run

### Fable quota exhaustion (twice)
`explanation` and `documentation` both hit a hard, account-level Fable usage quota partway through their runs (not a soft rate limit — confirmed via repeated identical failures). Both were left safely paused with real partial data intact (877/2678 and 625/1373 respectively) rather than forcing a model substitution, since the project's model assignment (Fable for these two categories) was an explicit spec requirement, not a preference. Resumed and completed after an account switch restored quota.

### Session usage limit
One dispatch (the DPO Part B scale-up) hit a full session usage limit mid-run ("You've hit your session limit · resets 7:20pm"). Paused, waited past the stated reset time, resumed cleanly — no data lost, the batch-handoff design is idempotent by construction so a hard stop mid-run never corrupts anything.

### Repeated agent stalling (the trajectory category's real fight)
The `trajectory` category's fill-dispatch agent stalled and had to be resumed more than a dozen times over several hours. Root cause, found by direct file/process inspection rather than guessing: the dispatched agent was spawning its *own* nested Agent-tool sub-dispatches to do the actual generation work, then ending its turn while those nested dispatches were still running in the background — so each "resume" would report a plausible-sounding status ("waiting for the background agent") while genuinely making no further progress until manually pushed again. A background watchdog script was set up to poll the real file state every 30 seconds and flag genuine 10-12-minute stalls (as opposed to normal slow-but-real progress), which is what let this get caught and fixed rather than trusting each agent's self-report. The category was also caught self-capping at 300/1561 items against its own judgment despite repeated explicit instructions to continue — this was overridden directly once the user confirmed "no pilots, do everything fully." The final ~82 items were finished by a fresh dispatch that was explicitly told **not** to use the Agent tool at all — forcing it to do the review-writing work itself directly instead of delegating further, which is what finally got it to completion without another stall.

### A one-time data-integrity bug in the final assembly step
`snapshot_conversion.jac`'s "already snapshotted, skip" idempotency check compared the output file's hash to *itself* rather than checking whether the real source files (`graph_conversion_growth.jsonl`, which grew from 10→23 records during the scale-up) had changed. This meant it would have silently kept serving a stale 10-record snapshot forever. Fixed with an explicit re-snapshot override flag, verified the new 1,943-row snapshot is correct.

### The holdout set was deliberately NOT grown
Categories grew 10–100x in this scale-up, so the original 18-id holdout set (carved when categories were pilot-sized) is now a much smaller fraction of the full dataset than intended. Re-running the holdout-carving script was investigated and specifically **not done**, because its selection logic picks "the last N records in file order" — since new records get appended to the end of each category file, forcing a re-carve would have selected an entirely different tail slice and silently discarded the original 18 held-out ids, rather than growing the set additively. That would break the "holdouts are frozen once and reused unchanged across every training arm" invariant the whole three-arm experiment design depends on. Left as-is, flagged clearly rather than guessed through.

### Unrelated concurrent work in the same git working tree
Partway through this session, a separate concurrent session was doing unrelated, large-scale work on the studio app itself (`jms/` deleted, `studio-desktop/` heavily modified — visible in `git status` and in intermediate commits from that other session). This was noticed, explicitly not touched, and every commit made by this session's work was scoped narrowly (`git add <specific files>`, never `git add -A`) specifically to avoid entangling the two unrelated efforts.

## 7. What is explicitly NOT done (out of scope, by design)

- **`post_cptv2` build** — the same pipeline re-run against the post-CPT-v2 checkpoint, to measure CPT's actual effect. This was the whole reason `04-cpt-sft` exists as a phase, but running it was always scoped as a separate, later step.
- **Arms A/B/C training + eval battery** — the three-arm comparison protocol itself (training runs, eval battery, statistical comparison) is downstream of this dataset and not part of this build.
- **code_gen's other 28 task types** — only 6 of the spec's 34 task types are implemented; expanding coverage is real future work, not a gap introduced by this session.
- **Buffer-category reallocation logic** — the spec's weight table includes an 8% "buffer" share with no implementation; shortfalls in other categories are not automatically redistributed into it.

## 8. Where everything lives

- Pipeline code: `model-experiments/01-sft-dpo/sft_dpo/jacgen2/`
- Generated data (gitignored): `model-experiments/04-cpt-sft/dataset/{shared,fresh}/`
- Final release files: `model-experiments/04-cpt-sft/dataset/fresh/releases/{sft_train.jsonl,dpo_train.jsonl}`
- Progress ledger (task-by-task history): `.superpowers/sdd/progress.md`
- Design docs: `model-experiments/04-cpt-sft/docs/`
- This report: `model-experiments/04-cpt-sft/docs/reports/2026-07-task18-full-run-report.md`
