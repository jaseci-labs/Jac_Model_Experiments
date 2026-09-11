# 08-nitin-new2-ds — Design Spec

Status: design of record as written 2026-09-10, before training. The phase
ran a reduced scope (Spectrum SFT only, both holdouts); see
[`../report.md`](../report.md). The DPO,
stock-arm and 12-cell parts below were not run.
Date: 2026-09-10.
Companion to [`CONTEXT_BRIEF.md`](CONTEXT_BRIEF.md) (settled facts) and
[`workflow.md`](workflow.md) (the runbook).

## 1. Purpose

One training battery, three research questions (RQ3 is new relative to
06/07). Direct follow-up to `07-nitin-ds-new-sft`, on a corpus that is not a
newer commit of the same single file but a merge of seven distinct source
types from the same upstream repo.

1. **RQ1 — does Spectrum replicate on a fourth dataset?** See
   `CONTEXT_BRIEF.md` §4.1 for the full prior-result chain.
2. **RQ2 — is this corpus better than 07's?** Both arms score on
   `04-cpt-sft`'s unchanged 855-row holdout (a) (`dataset/holdout_a_shared855.jsonl`),
   directly comparable to 07's and 06's cells.
3. **RQ3 — does real graph-native content change the picture?** 06 and 07
   trained on zero OSP archetypes; 08 trains on 2,537 rows of genuine
   `node`/`edge`/`walker` Jac (osp + farm). Neither holdout in this phase is
   itself graph-native-in-distribution (holdout (a) is jacgen2-derived;
   holdout (b) is reused from 07's py2jac-only content — see §4.2), so RQ3 can
   only be answered indirectly here: does the mixed-content training pool move
   the aggregate pass rate on the existing (non-graph-native) holdouts,
   relative to 07's single-shape pool. A clean graph-native-holdout test is
   explicitly out of scope for this phase (§10).

Two things differ from 07 beyond the dataset: the DPO axis (idiomaticity, not
injected-bug correctness — §3.1) and the merge pipeline itself (multi-source,
with two new filters 07 never ran). Base checkpoint, hyperparameters, seed,
LoRA geometry, eval harness, and holdout (a) are held fixed.

## 2. Source data

See `CONTEXT_BRIEF.md` §1 and §1.1 for the full 7-source table and per-source
verdicts. Summary:

| Field | Value |
|---|---|
| Origin | `https://github.com/chess10kp/jac-data-gen` |
| Commit pin | `c95b7563c16851010396170c269f1fdc5327ef15` (2026-09-01) |
| Sources merged | js2jac (8,674), composer (2,898), farm (1,592), step4_dpo (1,375), osp (943), step4_work (683), farm_handler (2) |
| Pre-dedup total | 17,173 |
| **Clean pool** (`candidate_pool.jsonl`) | **16,167** |

### 2.1 What each source actually contributes, and its verification strength

| source_type | verification | what it teaches |
|---|---|---|
| `js2jac` | 100% `jac check`-gated at the upstream pipeline stage; ORM records additionally cleared a behavioral gate | JS/TS → Jac translation, real-world repos |
| `composer` | `source=="idiomatic"` rows passed a hidden-test gate at generation time; 08 additionally requires `test_count >= 18` (§2.2 below) | Python → Jac translation, single-function, richer Jac idiom |
| `farm`/`farm_handler` | 100% behavioral round-trip (create/list/update/delete) verified against a live graph | **Graph-native**: node archetype + CRUD walker generation from a manifest spec |
| `step4_dpo` | both sides `jac check`-passed and hidden-test-verified at generation time; the pair itself encodes an idiomaticity contrast, not a correctness one | preference signal: idiomatic Jac over mechanically-floor Jac, same function |
| `osp` | `compiler_pass` AND `test_pass` both true on 100% of rows, verified by a model-generated test suite executed per-row | **Graph-native**: NL request → OSP module (node/walker/spawn) from scratch |
| `step4_work` | `coverage` field present, joined against composer's idiomatic rows | Python → Jac translation with an explicit Python source side (`origin_source_text` populated, unlike `composer`) |

### 2.2 Quality-gradient filter — the mechanism, restated with the acceptance numbers

`~/repos/jac-data-gen/docs/PY2JAC_QUALITY_GRADIENT.md` documents a
mutation-strength gate that validated on a 20-record calibration run and was
then silently dropped in mass production, producing a measurable weak tail in
`composer_dataset.jsonl` (median 14 hidden tests vs 21 for the strong-tail
first half). 08 computes a real per-row test count (joined from
`step4*/work/*.json`'s `test_blocks`) rather than trusting chunk provenance,
and applies **`test_count >= 18`**:

| stage | rows |
|---|---:|
| `composer` rows at pin, `source=="idiomatic"` | (subset of 15,144) |
| dropped: `no_work_join_no_test_signal` (no computable count) | 2,850 |
| dropped: `below_test_count_cutoff` (< 18) | 3,108 |
| survive into `composer` source_type | 2,898 |

This is the direct, testable candidate explanation for 07's own finding
(`../../07-nitin-ds-new-sft/report.md`): 07 trained on the full unfiltered pool and did not
beat 06 despite more rows. If 08 does beat 07 on RQ2, the per-slice breakdown
should check whether the win concentrates in `composer`-sourced rows
specifically — that would corroborate the mechanism rather than just the
correlation.

### 2.3 Eval-leakage denylist — applied for the first time

See `CONTEXT_BRIEF.md` §2.1 for the full explanation of what this bug is and
is not. Applied against the numeric-HF-row-id namespace (`composer`,
`step4_work`, `step4_dpo`); confirmed not applicable to the string-id
namespaces (`osp`, `js2jac`, `farm`). Drop counts: 550 SFT-pool rows, 103
DPO-pool pairs.

### 2.4 License filter — `js2jac` only

Kept: MIT, APACHE-2.0, NOASSERTION, CC0-1.0, BSD-2-CLAUSE, UNLICENSE (8,674
rows). Dropped: GPL-3.0, AGPL-3.0, OTHER, blank spdx (1,016 rows,
`license_not_permissive`).

## 3. Decisions locked / changed from 07

| Decision | Choice | Status |
|---|---|---|
| Base checkpoint | `models/qwen-q4` — unchanged across the entire lineage | **Locked** |
| Arms | stock (trailing-16) + spectrum (SNR picks), no CPT arm | **Locked** |
| Spectrum layer selection | Reused verbatim from 07 (sha256-identical chain back to 04-cpt-sft) | **Locked** |
| Script language | All `.jac` | **Locked**, carried from 07 |
| SFT task framing | **Six** task types now, not one — see `dataset-structure.md` — because the corpus genuinely contains six distinct shapes (translation × 3 sources, completion × graph-native, completion × CRUD) | **Changed from 07.** 06/07's single-task-type framing followed from their corpus having exactly one shape; 08's does not |
| DPO axis | **Idiomaticity** (chosen=idiomatic model output, rejected=floor model output, both real, both compiling) | **Changed from 07.** 07's axis was synthetic-bug correctness; not available here because `step4_dpo` pairs are naturally-occurring idiomaticity contrasts, not injectable bugs |
| Eval holdouts | Both: (a) `04-cpt-sft`'s unchanged 855; (b) **07's own holdout, reused byte-identical, not re-carved from 08's pool** | **Changed from 07's carve-fresh approach** — see §4.2 |
| Eval-leakage denylist | Applied | **New in 08** |
| Quality-gradient filter | Applied (`test_count >= 18`) | **New in 08** |
| Training recipe | Copy-adapted from 07, retargeting only data/adapter/results paths | **Locked** |
| Statistical bar | Two-proportion z-test / paired McNemar, p < 0.05 two-sided | **Locked**, same method as every prior phase |

### 3.1 Why the idiomaticity axis, not correctness

07's flagship axis (inherited from 06) required manufacturing a bug — the
corpus had no natural correctness contrast. `step4_dpo` removes that
constraint: two different models independently translated the same Python
function, and the upstream pipeline's own gate classified one output
`idiomatic` and the other `floor`. Both sides are **real model output that
both compile and both pass the hidden-test gate at generation time** — the
contrast is genuinely about Jac-language idiom quality, not a synthetic
defect. This is arguably a *cleaner* DPO signal than 07's (no "assume the bug
is behaviorally meaningful" caveat 07 had to carry), but it teaches a
different thing: prefer idiomatic style, not prefer correct behavior. Do not
describe 08's DPO stage as measuring the same thing 07's did.

## 4. Design — 2 arms × 2 holdouts × 3 stages

### 4.1 The arms

Unchanged from 07: stock (trailing-16, blocks 32-47) via
`run_sft.sh`+`run_dpo_nofuse.sh`; spectrum (16 SNR-picked blocks) via
`run_sft_spectrum.sh`+`run_dpo_spectrum.sh`. Both 281.838M trainable params,
enforced by `--verify-layers`.

### 4.2 The holdouts

**(a) unchanged, reused from every prior phase** —
`dataset/holdout_a_shared855.jsonl` (byte-identical copy of
`04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl`), 1,428 rows / 855
code-graded. See `CONTEXT_BRIEF.md` §4.1 for the full baseline table.

**(b) reused from 07, NOT carved fresh from 08's pool — a deliberate,
non-obvious choice, stated here so it isn't silently assumed to be a fresh
carve.** `dataset/nitin_holdout.jsonl` and `nitin_holdout_eval.jsonl` are
byte-identical copies of 07's files (verified during the build). This keeps
holdout (b) comparable to 07's own holdout-B numbers cell-for-cell. The cost:
holdout (b) is **still 100% py2jac-shaped content** — none of 08's four new
source types (osp, js2jac, farm, step4_dpo) contribute holdout-(b) rows, so
holdout (b) cannot measure in-distribution performance on the *new* material
the way 07's holdout (b) measured in-distribution performance on 07's own
corpus. If a future phase wants an in-distribution multi-source holdout, it
needs its own carve — explicitly out of scope here (§10).

Leakage: verified zero on candidate_pool.jsonl and on the post-split
sft/dpo train+valid files, against both holdouts, via four independent checks
(id, `jac` text hash, `jac_rejected` text hash, holdout-prompt text hash).

### 4.3 The matrix

Same 12-cell structure as 07: 3 stages × 2 arms × 2 holdouts, plus the 6
existing 07 cells and the 6 existing 06 cells as cross-phase references.
Statistical method unchanged (§7).

## 5. Reuse vs new, relative to 07

### 5.1 Reused verbatim (path-retargeted only)

Every training/eval script (`run_sft.sh`, `run_dpo_nofuse.sh`,
`eval_sft_sweep.sh`, `eval_dpo_nofuse.sh` and their spectrum equivalents),
every hyperparameter, the layer selection, the functional harness pointer,
`prep_training_dirs.sh`'s 85/15-seed-42 convention. See
`CONTEXT_BRIEF.md` §6 for the three open items inherited unchanged (holdout
default in DPO runners, the `total==855` gate, the stale "06 incumbent" echo
lines).

### 5.2 New in this phase

- The 7-source merge pipeline (`experiments/08-nitin-new2-ds/scaffold/pipeline.jac`) — not a port, a new
  build, since nothing like it existed before (07's `pipeline.jac` read one
  JSONL).
- The eval-denylist fix (§2.3) and quality-gradient filter (§2.2) — neither
  existed in 06's or 07's pipeline.
- `experiments/08-nitin-new2-ds/scaffold/copy_holdout.jac` — copies 07's holdout verbatim and re-verifies
  leakage against it; 07 had no equivalent since it carved its own holdout
  fresh.
- `experiments/08-nitin-new2-ds/scaffold/release.jac` — pool-schema → trainer-schema splitter; 07's
  equivalent was `collect.jac`, which assumed a single source schema. 08
  needed a new one because it has 7 heterogeneous source schemas to reconcile
  into one trainer-facing format (fenced SFT assistant turns, unfenced DPO
  chosen/rejected — matching 07's wire format exactly).

## 6. Decontamination

Exact and normalized-hash dedup (comments stripped, whitespace collapsed).
This spec originally said 06/07's 14-token shingle near-duplicate check was
carried over; `experiments/08-nitin-new2-ds/scaffold/pipeline.jac` does not contain it. Plus the
eval-denylist pass neither 06 nor 07 ran, plus a global cross-source dedup pass
07 didn't need (07 had one source; 08's `farm_handler` turned out to be a
near-total content subset of `farm`, caught here — 1,006 rows deduped
globally). Full numbers in `corpus-triage-report.md`.

**Stated limitation, carried forward from 07:** the `chosen`/`idiomatic` side
of translation-sourced rows is assumed correct on the strength of a compile +
hidden-test gate at the upstream generation stage, not re-verified here.
`osp` and `farm` carry stronger evidence (model-written test suites executed
per-row for osp; live behavioral round-trip for farm) than `composer`/
`step4_work`/`js2jac` (compile-gate only, plus the test-count filter for
composer). Report this asymmetry if RQ3's graph-native-vs-flat comparison
becomes a headline finding — the graph-native sources are, if anything,
*more* rigorously verified than the flat-function sources, not less.

## 7. Acceptance and decision criteria (pre-registered)

Unchanged methodology from 06/07:

### 7.1 Statistical method

Unpaired two-proportion z-test for marginal rates; paired McNemar wherever
both sides answer identical items (every cross-phase comparison against 07 or
06 on holdout (a), and every arm-vs-arm comparison within a holdout). In
practice 06 and 07 reported only the z-test (no per-row data was saved), so
the cross-phase McNemar comparison against them is not possible.
Significance bar: p < 0.05, two-sided.

### 7.2 Decision rules

**RQ2 — is 08 better than 07?** Paired McNemar of each holdout-(a) cell
against 07's corresponding cell (once 07's own final numbers are read from
`../../07-nitin-ds-new-sft/report.md`; they are
now in `CONTEXT_BRIEF.md` §4.1). Better iff ≥4/6
significant wins with no significant losses; worse iff the mirror holds;
indistinguishable otherwise — a legitimate result, not a failure (07 landed
there against 06 and said so).

Mandatory caveats on any positive RQ2 result: (1) the size confound — 08's
pool is 2.18× 07's and multi-source, so a win is not a clean quality
isolation; (2) per-source-type breakdown of the win, to check whether §2.2's
quality filter is the actual mechanism (a win concentrated in `composer`-
sourced rows corroborates it; a uniform win across all source types suggests
volume alone, not the filter, did the work).

**RQ1 — does Spectrum replicate?** Same ≥4-of-6-significant-wins-no-losses
rule as every prior phase.

**RQ3 — does graph-native content change anything?** No formal pre-registered
statistical test — §4.2 already states neither holdout is graph-native
in-distribution, so RQ3 can only be read qualitatively from whether the
aggregate pass rate moved and from the per-source-type failure breakdown
(`experiments/08-nitin-new2-ds/eval/gen_eval_detail.jac` + `grade_eval_detail.jac`; never run this
phase) showing whether osp/farm-
derived generalization shows up on the existing holdouts at all. State this
limitation plainly in the final report rather than overclaiming a test that
wasn't run.

### 7.3 Power

At n=855 and ~70% base rate, ~4-5pp is roughly the unpaired detection floor
(established in 04-cpt-sft); paired testing detects meaningfully less. No
change from prior phases since holdout (a) is unchanged in size.

### 7.4 Run-level acceptance gates

Unchanged from 07: `--verify-layers` passes at exactly 281.838M trainable
params; `adapter_config.json` rewritten + 256-key assertion before any
spectrum-arm scoring; no all-zero adapter weights; full `iters: 8200` or the
shortfall reported; best-checkpoint provenance re-verified across any resume;
preflight lock clear; base model scored once per holdout as the floor.

## 8. Live monitoring

Unchanged from 06/07 — see `CONTEXT_BRIEF.md` §8.

## 9. Directory layout

See `CONTEXT_BRIEF.md` §9.

## 10. Out of scope

- Re-running the SNR scan (unchanged base model weights).
- Any CPT stage (03's CPT-v2 was rejected: `../../../docs/history/03-cpt-only/cpt-2-results.md`).
- GRPO / RL (05's planned scope, never run: `../../../docs/HISTORY.md`).
- **A graph-native-in-distribution holdout carve.** RQ3 is deliberately
  answered only indirectly this phase (§4.2, §7.2) — building a proper
  in-distribution eval set from the osp/farm content is real scope for a
  follow-up phase, not folded in here.
- Expanding the `mm4_issue_problems.jsonl` prompt pool (508 of 540 prompts
  are ungenerated) — that would mean generating new osp content, not merging
  what already exists.
- Multi-seed repeats per arm — one seed per arm, as in 07.
