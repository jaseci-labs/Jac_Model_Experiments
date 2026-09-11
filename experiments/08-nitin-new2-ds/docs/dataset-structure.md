# Dataset structure — 08-nitin-new2-ds

Source: `jac-data-gen@c95b7563c16851010396170c269f1fdc5327ef15`, 7 source
files merged (`CONTEXT_BRIEF.md` §1.1). Real counts throughout — the
dataset build is complete. Both arms train on identical copies of the
releases below. Both arms eval on two holdouts: `04-cpt-sft`'s unchanged
1,428-row set (855 code-graded), and 07's `nitin_holdout_eval.jsonl` reused
byte-identically (`spec.md` §4.2).

## Triage yield

| stage | n | Δ | note |
|---|---:|---:|---|
| S0 rows across 7 sources at pin | 17,173 | — | js2jac 8,674 + composer 2,898 + farm 1,592 + step4_dpo 1,375 + osp 943 + step4_work 683 + farm_handler 2, already post per-source filters below |
| — composer: dropped `no_work_join_no_test_signal` | | −2,850 | no joinable `step4*/work/*.json` test-count signal |
| — composer: dropped `below_test_count_cutoff` (<18) | | −3,108 | `docs/PY2JAC_QUALITY_GRADIENT.md` weak-tail fix |
| — js2jac: dropped `license_not_permissive` | | −1,016 | GPL-3.0/AGPL-3.0/OTHER/blank spdx |
| — sft-pool: dropped `eval_denylist` | | −550 | jac-data-gen's own sealed eval set, applied for the first time |
| — dpo-pool: dropped `eval_denylist` | | −103 | same |
| — dpo-pool: dropped `chosen_equals_rejected` | | −4 | degenerate pairs |
| S3-S4 within-corpus dedup (exact + near-dup) | | −206 (sft) / included above | `duplicate_content` |
| global cross-source dedup | | −1,006 | mostly `farm_handler` as a near-total content subset of `farm` |
| holdout leak-check vs both holdouts | | −530 (sft) / −117 (dpo) | `holdout_leak` — checked and caught, not assumed zero |
| other (duplicate id/content, null field) | | −339 | `duplicate_id_or_content` 133, `null_source_field` 5, `duplicate_content` 206 (sft, listed once) |
| **clean pool** (`candidate_pool.jsonl`) | **16,167** | | zero duplicate ids, funnel arithmetic verified to close exactly |
| — holdout (b) | 855 | | **reused verbatim from 07**, not carved from this pool — see `spec.md` §4.2 |
| — training pool after split | 14,792 SFT + 1,375 DPO | | |

Full per-reason breakdown: `experiments/08-nitin-new2-ds/docs/corpus-triage-report.md`.

## Candidate pool composition, by source_type

| source_type | rows | task_kind | tier(s) present |
|---|---:|---|---|
| js2jac | 8,674 | translation | cleaned 6,341→(post-filter 5,585 est.), floor, floor_fallback, idiomized |
| composer | 2,898 | translation | idiomatic (all — floor rows excluded from this source_type) |
| farm | 1,592 | completion | round-trip verified |
| step4_dpo | 1,375 | pair_dpo | idiomatic_vs_floor |
| osp | 943 | completion | test-verified (compiler_pass + test_pass both true) |
| step4_work | 683 | translation | idiomatic (Python-sourced) |
| farm_handler | 2 | completion | round-trip verified |
| **total** | **16,167** | | |

By `task_kind`: translation 12,255 (75.8%) / completion 2,537 (15.7%) /
pair_dpo 1,375 (8.5%).

## SFT — task-type / register composition (n = 14,792, `dataset/sft_train.jsonl`)

Unlike 06/07 (one task shape, one honest row), 08's corpus genuinely spans
six shapes:

| source_type | register | task_type | n | example prompt |
|---|---|---|---:|---|
| js2jac | translate_request | js_to_jac | 8,674 | *"Translate this JavaScript/TypeScript to idiomatic Jac:\n\nexport default function AuthLayout({ children }: { children: React.ReactNode }) { return (\<div className="relative flex min-h-screen items-center justify-center bg-muted/30 px-4 py-12"\>...) }"* |
| composer | translate_request | python_to_jac_function | 2,898 | *"Translate this Python function to idiomatic Jac:\n\ndef compare_elements(prev_hash_dict, current_hash_dict):\n    \"\"\"Compare elements that have changed...\"\"\"..."* |
| farm | spec_style | farm_crud_module | 1,592 | *"Write a single Jac module for the data layer of `2024-Arizona-Opportunity-Hack/...` (ported from `backend/app/app/schemas/token.py`).\n\nNode archetypes to define:\n- `MagicTokenPayload`: sub (str \| None, default None)...\n\nThen add public CRUD walkers..."* |
| osp | user_request | osp_module | 943 | *"In my bug tracker project, tickets hang off a project node. When filing new tickets I'd like to capture the node returned by the connection itself..."* |
| step4_work | translate_request | python_to_jac_function | 683 | (Python source + prompt from `step4*/work/*.json`, idiomatic target joined from `composer_dataset.jsonl`) |
| farm_handler | spec_style | farm_crud_module | 2 | same shape as `farm`, near-total content overlap (981/983 deduped away) |

**Wire format** (`1-scaffold/release.jac`): SFT assistant turns are
` ```jac ... ``` `-fenced, matching 07's convention (07 is 100% fenced; eval
scripts strip fences before scoring). All rows carry `id`, `source_type`,
`task_type`, `tier`, `dataset_version="nitin-new2-ds-v1.0.0"`,
`run_tag="nitin-new2"`, `origin="jac-data-gen:c95b7563…:<path>#id=<id>"`,
`quality_signals` (source-specific — test counts for composer/step4_work,
`compiler_pass`/`test_pass` for osp, `gate` for farm), `register`,
`messages`.

Split sizes:

| file | n |
|---|---:|
| `dataset/sft_train.jsonl` | 14,792 |
| `dataset/sft/train.jsonl` | 12,573 (85.0%, seed 42); **12,570** after `nan_guard.jac` dropped 3 over-length js2jac rows |
| `dataset/sft/valid.jsonl` | 2,219 (15.0% — training-time validation split, NOT a scored eval set); **2,217** after `nan_guard.jac` |
| `dataset/nitin_holdout.jsonl` | 855 (reused verbatim from 07) |
| `dataset/nitin_holdout_eval.jsonl` | 855 (same, with 07's authored instructions) |

For reference: 04-cpt-sft fresh arm 8,100 SFT rows; 06: 5,474; 07: 6,781;
**08: 14,792 — 2.18× 07's, the largest pool in the lineage.** Named confound
on RQ2 per `spec.md` §3 — not a footnote.

## DPO — 1 axis: idiomaticity, not correctness (n = 1,375 pairs, `dataset/dpo_train.jsonl`)

| axis | n | example prompt |
|---|---:|---|
| idiomaticity | 1,375 | *"Translate this Python function to idiomatic Jac:\n\ndef sortMapping(streamPair):\n    \"\"\"Sort mapping by complexity...\"\"\"..."* — chosen = composer-model's idiomatic output (typed, `list[dict[...]]`, lambda with typed param), rejected = sonnet-model's floor output (`Any`/`object` types, untyped lambda) |

**This is a different axis from 07's**, which used synthetic bug-injection on
a correctness dimension. 08's `chosen`/`rejected` are both real model
outputs from the upstream pipeline's own two-model translation pass — the
`source` tier disagreement (`idiomatic` vs `floor`) *is* the preference
label. See `spec.md` §3.1 for the full reasoning and its implications for
report interpretation.

Provenance: cross-referenced `step4{b,c,d,e}/dataset_{sonnet,composer}.jsonl`
pairs by shared id within each step4 variant directory, filtered to rows
where the two models' `source` field disagreed.

Dual-gate accounting (both sides already `jac check`-passed and hidden-test-
verified at upstream generation time, per §2.1 of `spec.md`):

| check | result |
|---|---:|
| `chosen != rejected` (degenerate pairs dropped) | 4 dropped (`chosen_equals_rejected`) |
| `eval_denylist` collision | 103 dropped |
| `holdout_leak` (vs either holdout) | 117 dropped |
| pairs accepted | **1,375** |

Split sizes:

| file | n |
|---|---:|
| `dataset/dpo_train.jsonl` | 1,375 |
| `dataset/dpo/train.jsonl` | 1,169 (85.0%, seed 42) |
| `dataset/dpo/valid.jsonl` | 206 (15.0% — `mlx_lm_lora`'s training-time validation split) |

07 for reference: no DPO pairs existed anywhere in the upstream repo at its
pin (07 had to synthesize its own axis); this is the first phase in the
lineage with naturally-occurring preference data.

## Eval holdouts

| holdout | rows | code-graded | reused or new |
|---|---:|---:|---|
| `dataset/holdout_a_shared855.jsonl` (copy of `04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl`) | 1,428 | 855 | reused **unchanged** across the whole lineage |
| `dataset/nitin_holdout_eval.jsonl` | 855 | 855 | **reused byte-identically from 07** — not a fresh carve from 08's pool |

Composition of the shared 855 (`spec.md` §4.2): unchanged from 06/07 — 322
conversion/behavioral, 227 code_gen/compile_only, 113 trajectory/compile_only,
86 code_gen/behavioral, 70 trajectory/behavioral, 33 debug/behavioral, 3
debug/compile_only, 1 migration/behavioral.

**Neither holdout is in-distribution for 08's new source types.** Holdout (a)
predates this entire lineage (jacgen2-derived); holdout (b) is 100%
py2jac-shaped content from 07's own corpus. None of osp/js2jac/farm/
step4_dpo's content appears in either holdout. This is why RQ3
(`spec.md` §1) can only be answered indirectly this phase — flagged
explicitly, not silently glossed over.

Sources: `dataset/sft_train.jsonl`, `dataset/dpo_train.jsonl`,
`dataset/nitin_holdout.jsonl`, `dataset/nitin_holdout_eval.jsonl`,
`experiments/08-nitin-new2-ds/docs/corpus-triage-report.md`.
