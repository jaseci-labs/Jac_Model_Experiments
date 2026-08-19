# Corpus triage report — 07-nitin-ds-new-sft

Produced by `scripts/compile_check.jac` + `scripts/pipeline.jac`, 2026-08-18.
Raw stats: `/tmp/nitin_new_triage/stats.json`.

## Source

| Field | Value |
|---|---|
| Repo | `https://github.com/chess10kp/jac-data-gen` — **same repo 06 used**, newer commit |
| Commit pin | **`11fa3f45a0a349337ae4c355708a7e4974b54a36`** ("Add idiomatic py2jac corpus (9,367 records)", 2026-08-17) |
| 06's pin | `7c25aff3110f526eec59e0123ffe6c0c152cce91` (2026-08-10) |
| File | `data/py2jac_dataset_idiomatic.jsonl` — one JSONL, **not** 06's loose `data/jac_outputs/*.jac` |
| Records | 9,367, schema `{id, entrypoint, source, jac}`, `source == "idiomatic"` throughout |
| Clone | `~/repos/jac-data-gen` (internal disk — the project's external drive measured ~4.3 MB/s) |

Each record is given the pseudo-filename `<id>.jac` so every triage stage runs
06's logic unchanged over the same key space (ids are unique ints, 9,367/9,367).

## Corpus shape — full scan, not a sample

Scanned all 9,367 records. **Structurally identical to 06's corpus:**

| Check | Result |
|---|---|
| Top-level `def` per record | **1, for every record** |
| Starts with a docstring | 8,970 / 9,367 (95.8%) |
| **OSP archetype declarations** (`node X {`, `edge X {`, `walker X {`) | **0** |
| `walker` keyword / `with entry` blocks | 0 / 0 |
| node/edge/root/here word hits | 92 records — all incidental (docstring prose, identifiers in tree/graph *algorithms*) |
| `obj` / `class` declarations | 17 |

**"Idiomatic" means richer Jac idiom inside the same flat-function shape, not
graph-native code:** 9,359 records carry an explicit `->` return type, 2,566 use
union types, 479 use `isinstance` narrowing, 143 use `match`, 120 use `lambda`.
That is the real 06-vs-07 contrast — **same task shape, better-written Jac** —
and it is what RQ2 measures. Do not describe this data as graph-native or OSP.

## Funnel

| stage | n | Δ | note |
|---|---:|---:|---|
| S0 records at commit pin | 9367 | — | |
| S1 compile FAIL dropped (`jac run` exit ≠ 0) | | −812 | 8555 survive (**91.3% compile pass**) |
| S2 quality drop: no_docstring | | −382 | nothing to reverse-author from |
| S2 quality drop: trivial_body | | −491 | < 12 body tokens or < 3 code lines |
| S2 quality drop: no_def | | −5 | |
| S2 quality drop: multi_def | | −0 | none — every record is single-`def` |
| S2 survivors | 7677 | | |
| S3 exact-duplicate (normalized token hash) | | −2 | 7675 survive |
| S4 near-duplicate (14-token shingle, ≥0.50) | | −27 | 7648 survive |
| S5 decontamination vs 4 reference holdouts | | −1 | 7647 survive |
| S5b func-name collision, ident-Jaccard ≥0.50 | | −11 | 7636 survive |
| **clean pool** | **7636** | | |
| — carved to `nitin_holdout.jsonl` | 855 | | frozen pre-generation, seed 42, **0 leak-check failures** |
| — remaining training pool (`candidate_pool.jsonl`) | **6781** | | |

Total yield: 6,781 / 9,367 = **72.4%** into training, plus an 855-row holdout.

### Compile failures (812)

All 812 exited rc=1. Overwhelmingly one signature: **796 are `error[E0002]: Missing ';'`**
— the upstream generator emits Python-style statements without Jac's required
terminator in a minority of records. The rest are long-tail (2 × `Expected 'by', got 'with'`,
one undefined-lambda binding, and a handful of unique parse errors). Nothing here
suggests a harness problem; these records genuinely do not compile.

### Decontamination

Reference sets and the shingles they contributed:

| reference | rows | strict 14-token shingles | relaxed 10-token ident shingles |
|---|---:|---:|---:|
| `04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl` | 1428 | 164,163 | 172,525 |
| `04-cpt-sft/sft_fresh_probe/dataset/dpo/valid.jsonl` | 115 | 11,577 | 11,780 |
| `01-sft-dpo/dataset/eval_holdout/conversion.jsonl` | 150 | 4,946 | 5,779 |
| `01-sft-dpo/dataset/eval_holdout/graph_conversion.jsonl` | 27 | 1,947 | 1,906 |

- **1 strict/relaxed contamination hit**: `is_prime` (relaxed overlap 0.635 vs
  `sft_valid`).
- **100 function-name collisions** audited; **11 dropped** at ident-Jaccard ≥ 0.50.
  All are classic textbook routines — `sieve_of_eratosthenes` (0.63),
  `factorial` (0.50), and four variants of `is_prime` (0.538–0.556). These are
  exactly the functions a mined corpus and a hand-built holdout both contain, so
  dropping them is the right call even though the overlap is coincidental rather
  than copied.
- **Holdout leak check: 0 failures.** No holdout row shares ≥ 0.50 shingle
  overlap with the training pool.

Contamination pressure is very low (1 + 11 of 7,648), which is consistent with a
corpus mined independently of this project's own holdouts.

## Holdout B — fixed at 855

855 rows, carved seed-42 **before any generation**, held out of the training pool
by id-exclusion (re-asserted in `prepare_batches.jac`, which refuses to run on a
leak).

855 is a **deliberate constant**, matched to holdout A and to 06's own holdout B
("Same 855-row size (matched deliberately at holdout-carve time)", 06's final
report §2) so the columns are equal-sized and cross-phase comparison stays
like-for-like. It was **not** rescaled to this corpus's larger pool.
`pipeline.jac` now hard-codes `HOLD_N = 855` and **raises** below a 4,000-row
pool rather than silently shrinking the way 06's `len(pool)//5` fallback would
have — a silently smaller holdout would break comparability with no signal.

## Generation parameters re-derived from this pool

| parameter | 06 | 07 | basis |
|---|---|---|---|
| training pool | 5,474 | **6,781** | triage output |
| `SFT_SHARDS` | 16 (342–343/shard) | **20** (339–340/shard) | 06's 16 would give 424/shard here, outside the 300–400 band one dispatched agent carries |
| body size median | 31 tok / 8 lines | **32 tok / 8 lines** | mean 39.0 / 9.9, p25 21, p75 49 |
| `DPO_MIN_TOKENS` / `_CODE_LINES` | 30 / 5 | **30 / 5 (unchanged)** | median is statistically the same, so the floor transfers |
| rows clearing the DPO floor | 2,870 / 5,474 (52.4%) | **3,614 / 6,781 (53.3%)** | nearly identical proportion — floor **validated**, not merely inherited |
| `DPO_SAMPLE_N` | 1,000 | **1,000 (unchanged)** | deliberately not scaled up; DPO set size is part of the recipe being replicated |

## Named confound for RQ2

07's training pool is **6,781 rows vs 06's 5,474 (1.24×)** and the original
`04-cpt-sft` fresh arm's 8,100 (0.84×). Dataset size and dataset content differ
at once, so the RQ2 comparison is not a clean dataset-quality isolation. This
must be stated in the final report's headline, not buried — 06 hit the same
confound in the opposite direction (its pool was 0.68× the fresh arm's) and
flagged it as unresolved.

## Batches prepared

| track | shards | items | source |
|---|---:|---:|---|
| `sft` | 20 | 6,781 | `candidate_pool.jsonl` (all rows) |
| `holdout_eval` | 3 | 855 | `nitin_holdout.jsonl` (all rows) — **eval only, never trained on** |
| `dpo` | 3 | 1,000 | seeded sample (seed 42) of the 3,614 rows clearing the floor |

The `holdout_eval` track is **new in 07**. 06 froze its holdout without authored
instructions, so the rows had no `messages` field, and its first holdout-B
evaluation "completed" in under 30 seconds having generated nothing (06's final
report §4, incident 4). Authoring those instructions is done up front here, with
the identical prompt builder and register split used for the training set.
