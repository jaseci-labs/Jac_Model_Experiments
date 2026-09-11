# Corpus triage report — 08-nitin-new2-ds

Produced by `experiments/08-nitin-new2-ds/scaffold/pipeline.jac` + `scaffold/copy_holdout.jac` +
`scaffold/release.jac`, 2026-09-09/10. Raw stats:
`experiments/08-nitin-new2-ds/docs/funnel/stats.json`, `experiments/08-nitin-new2-ds/docs/funnel/release.json`. (persisted in the experiment dir.)
Independently re-verified end-to-end after a mid-run external-drive
disconnect (§6) rather than trusted as-is.

## Source

| Field | Value |
|---|---|
| Repo | `https://github.com/chess10kp/jac-data-gen` — same repo 06/07 used, three commits further |
| Commit pin | **`c95b7563c16851010396170c269f1fdc5327ef15`** ("data(osp): test-verified gold slice (1034 PASS) + mm4 issue pool", 2026-09-01) |
| 07's pin | `11fa3f45a0a349337ae4c355708a7e4974b54a36` (2026-08-17) |
| Diff vs 07's pin | 16,069 files changed, +324,870/−9,368 lines — four wholly new source types, not an incremental update to 07's single file |
| Clone | `~/repos/jac-data-gen` (internal disk) |

## Sources merged, with pre-filter counts

| source_type | file(s) | rows at pin | rows kept |
|---|---|---:|---:|
| `js2jac` | `scripts/js2jac_dataset/js2jac_dataset.jsonl` | 9,828 | 8,674 |
| `composer` | `data/composer_dataset.jsonl` | 15,144 | 2,898 |
| `farm` | `data/farm_dataset.jsonl` | 1,731 | 1,592 |
| `farm_handler` | `data/farm_handler_dataset.jsonl` | 983 | 2 |
| `step4_dpo` | `data/step4{b,c,d,e}/dataset_{sonnet,composer}.jsonl`, cross-referenced | 1,599 candidate pairs | 1,375 |
| `osp` | `data/osp_dataset_pass.jsonl` | 1,034 | 943 |
| `step4_work` | `data/step4{,b,c,d,e,f}/work/*.json`, joined against `composer_dataset` idiomatic rows | 7,770 | 683 |

## Filter funnel

| stage | n | Δ | note |
|---|---:|---:|---|
| S0 total rows across 7 sources, pre-filter | 38,089 | — | sum of "rows at pin" above |
| S1 composer: `no_work_join_no_test_signal` | | −2,850 | no joinable `step4*/work/*.json` test count |
| S1 composer: `below_test_count_cutoff` (<18) | | −3,108 | quality-gradient fix, `spec.md` §2.2 |
| S1 js2jac: `license_not_permissive` | | −1,016 | GPL-3.0/AGPL-3.0/OTHER/blank spdx |
| S1 js2jac: null `js`/`path` | | −5 | malformed upstream rows |
| S2 pre-dedup merged total | 17,173 | | (includes the 1,599→1,375 step4_dpo pairing loss, already reflected) |
| S3 within-corpus dedup (`duplicate_content`, sft) | | −206 | |
| S3 within-corpus dedup (`duplicate_id_or_content`, sft) | | −133 | |
| S4 global cross-source dedup | | −1,006 | mostly `farm_handler` as a content subset of `farm` (981/983 dropped) |
| S5 eval-denylist (sft pool) | | −550 | jac-data-gen's own sealed eval set — see §5 |
| S5 eval-denylist (dpo pool) | | −103 | |
| S5 null_source_field | | −5 | |
| S6 dpo: `chosen_equals_rejected` | | −4 | degenerate pairs |
| S7 holdout leak-check (sft) | | −530 | vs both holdouts — real drops, not a zero-check formality |
| S7 holdout leak-check (dpo) | | −117 | |
| **clean pool** (`candidate_pool.jsonl`) | **16,167** | | zero duplicate ids across all 16,167; funnel arithmetic verified to close stage-by-stage |

Total yield: 16,167 / 17,173 pre-dedup merge = **94.1%** survived post-merge
filtering into the final pool (the earlier per-source filters, §1 above,
already removed the bulk of what didn't qualify before the merge).

### Rejected — SFT (9,404 rows, `dataset/rejected/sft/rejected.jsonl`)

| reason | n |
|---|---:|
| `below_test_count_cutoff` | 3,108 |
| `no_work_join_no_test_signal` | 2,850 |
| `license_not_permissive` | 1,016 |
| `global_duplicate_content` | 1,006 |
| `eval_denylist` | 550 |
| `holdout_leak` | 530 |
| `duplicate_content` | 206 |
| `duplicate_id_or_content` | 133 |
| `null_source_field` | 5 |
| **total** | **9,404** |

By source: `composer` 6,473, `js2jac` 1,154, `farm_handler` 981, `step4_work`
566, `farm` 139, `osp` 91.

### Rejected — DPO (224 pairs, `dataset/rejected/dpo/rejected.jsonl`)

| reason | n |
|---|---:|
| `holdout_leak` | 117 |
| `eval_denylist` | 103 |
| `chosen_equals_rejected` | 4 |
| **total** | **224** |

All 224 are `step4_dpo` (the only DPO source).

## Eval-leakage denylist — applied for the first time in this lineage

`jac-data-gen`'s `evals/function/v1/denylist_ids.txt` (1,404 cluster-expanded
ids) exists to keep training data disjoint from that repo's own sealed eval
set. It was verified applicable only to the numeric-HF-row-id namespace used
by `composer`/`step4_work`/`step4_dpo`; `osp`/`js2jac`/`farm` use unrelated
string-id schemes and were confirmed clear rather than assumed clear.

**Neither 06 nor 07 applied this filter.** This does not invalidate their
reported holdout-(a)/(b) numbers (scored against this project's own carved
holdouts, never against `jac-data-gen`'s eval set) — but their released
training data does carry measured overlap with it (5.1% of composer, 7.8% of
py2jac_dataset_idiomatic, per the upstream investigation). 08 is the first
phase to apply the fix; 653 total rows (550 SFT + 103 DPO) were dropped for
it here.

## Quality-gradient filter — mechanism and yield

`~/repos/jac-data-gen/docs/PY2JAC_QUALITY_GRADIENT.md` documents that a
mutation-strength gate (kill ≥80% of injected mutants) was validated on a
20-record run then silently dropped in mass production, leaving
`composer_dataset.jsonl`'s later rows a measurably weaker tail (median 14
hidden tests vs 21 for the strong first half). 08 computes a real per-row
test count by joining against `step4*/work/*.json`'s `test_blocks` field and
keeps `test_count >= 18`. Of composer's `source=="idiomatic"` rows: 2,850 had
no joinable test-count signal at all (excluded, not assumed good), 3,108
scored below 18, and 2,898 survived. **This is the direct testable
explanation for 07's own finding** — its `07-final-comparison.md` reports
07's larger-but-unfiltered pool did not beat 06 (0/6 significant, all point
estimates favor 06).

## Holdout (b) — reused, not re-carved

`dataset/nitin_holdout.jsonl` and `nitin_holdout_eval.jsonl` were copied
byte-identically from `07-nitin-ds-new-sft/dataset/` (855 rows each, verified
by content hash, not just row count). This was a deliberate choice to keep
holdout (b) comparable to 07's own holdout-B numbers cell-for-cell — see
`spec.md` §4.2 for the tradeoff (holdout (b) is not in-distribution for 08's
four new source types).

**Leak-check re-run against 08's own pool, not assumed inherited from 07's
own check:** 0 collisions on all four surfaces (id, `jac` text hash,
`jac_rejected` text hash, holdout-prompt text hash), checked against
`candidate_pool.jsonl` as written to disk, then re-checked post-split against
`sft/{train,valid}.jsonl` and `dpo/{train,valid}.jsonl` — still 0.

## Generation parameters, for comparison across the lineage

| parameter | 06 | 07 | 08 | basis |
|---|---|---|---|---|
| training pool (SFT) | 5,474 | 6,781 | **14,792** | this report |
| training pool (DPO) | 0 (no DPO pairs available) | ~3,614 eligible (synthesized) | **1,375** (naturally-occurring) | this report / 07's triage report |
| pool growth vs prior phase | — | 1.24× | **2.18×** | |
| source count | 1 | 1 | **7** | §2 |
| graph-native rows | 0 | 0 | **2,537 (15.7%)** | `CONTEXT_BRIEF.md` §1.2 |

## Named confound for RQ2

08's training pool is 2.18× 07's, drawn from 7 sources instead of 1, and
15.7% of it is genuinely graph-native content 06/07 never trained on. Dataset
size, source diversity, and content type all differ from 07 at once — the
RQ2 comparison is not a clean single-variable isolation the way 07-vs-06 was
even meant to be. This must be stated in the final report's headline, not
buried — 07 hit an analogous (smaller) version of this confound against 06
and did state it.

## §6 — Incident: external-drive disconnect mid-build, and its recovery verification

During this phase's own dataset build (2026-09-09), the project's external
drive (`/Volumes/ExtremePro`) disconnected mid-run. `.git` and the in-progress
`dataset/` directory both began returning `Input/output error`; a `diskutil
verifyVolume` attempt itself failed with "could not freeze volume:
Input/output error," indicating a live hardware/connection fault, not just
stale directory metadata. Two experiment directories untouched by this
session (`02-rl-grpo`, `03-cpt-only`) also briefly returned `Input/output
error` on `stat`, confirming the fault was drive-wide, not scoped to files
being actively written.

**Recovery:** after a physical reseat, the drive remounted with SMART status
"Verified." Rather than trust the partial `candidate_pool.jsonl` left on disk
from the interrupted run, it was independently re-verified: JSON-parsed on
every one of its 16,167 lines with zero errors, checked for zero duplicate
ids, checked that the funnel's stage-by-stage subtraction closed exactly
against the source counts at the pin, 10-record spot-checked across all 7
`source_type` values for schema completeness and non-truncation, and cross-
checked against `stats.json` (written only after all three output files in
`pipeline.jac`'s final block — its presence alone proved the write loop had
completed). Verdict: trustworthy as-is, no rebuild needed. This is now the
**third** external-drive-disconnect incident in this lineage (06's report §4
incidents 1 and 3 were the first two) — treat "verify after any interruption,
don't assume completion" as a standing operating rule for this project's
external-drive artifacts, not a one-off.
