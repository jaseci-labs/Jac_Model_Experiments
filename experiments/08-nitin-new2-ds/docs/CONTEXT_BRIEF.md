# 08-nitin-new2-ds — master context brief

Read this FIRST, in full, before doing anything else. It exists so every agent
working this experiment shares the same facts instead of re-deriving them —
do not re-discover what's already answered here; do flag if something here
turns out to be wrong.

> **Status: COMPLETE (2026-09-11).** Spectrum SFT trained (8200/8200, no NaN)
> and evaluated on both holdouts: **70.5% (603/855) on holdout (a), 37.7%
> (322/855) on holdout (b)**. Write-up: `experiments/08-nitin-new2-ds/report.md`.
> Stock SFT also trained cleanly but was never evaluated, and its adapter was
> deleted in the 2026-09-11 repo cleanup. No DPO ran. Every other number in
> this file is a *prior* result (07, 06, or 04-cpt-sft) used as the baseline.
>
> The 07, 06 and 04-cpt-sft directories referred to below were deleted in the
> same cleanup. Their reports are in `../docs/history/`; the holdout (a) file
> lives here as `dataset/holdout_a_shared855.jsonl`; the eval harness is
> `3-eval/eval_functional.jac`. How to run a new experiment from this tree:
> `../docs/PLAYBOOK.md`.
>
> Dataset: source repo re-pinned (§1), 7-source merge + eval-denylist fix +
> quality-gradient filter + license filter (§2), triage funnel
> (17,173 pre-dedup → 16,167 clean pool → 14,792 SFT release + 1,375 DPO
> release, 0 leaks; see `experiments/08-nitin-new2-ds/docs/corpus-triage-report.md`), holdout (b)
> reused verbatim from 07 for comparability, 85/15 splits built.
>
> **IMPORTANT — scope cut, superseding the original 12-cell design
> (2026-09-11, two successive explicit user calls, second overrides the
> first):** this phase now trains and reports **spectrum-arm SFT only**.
> Concretely:
> - Stock SFT DID run to completion (clean, no NaN) — its adapter/results
>   are kept on disk as an artifact, but it is **NOT evaluated and NOT in
>   the report**. Do not run `3-eval/eval_sft_sweep.sh` for this phase.
> - **DPO is skipped entirely, both arms.** Neither
>   `2-train/stock/run_dpo_nofuse.sh` nor `2-train/dpo/run_dpo_spectrum.sh`
>   runs this phase. Do not launch either without asking first.
> - Pipeline is: spectrum SFT (§6) → eval spectrum SFT only, both holdouts
>   (§5.1/§5.2, spectrum scripts only) → report.
> - Consequence: RQ1 (arm-vs-arm) and the DPO-axis analysis (§2.4) are both
>   **out of scope this phase** — there is nothing to compare spectrum
>   against, and no DPO-stage numbers at all. The report is a single-arm,
>   SFT-only result compared against 07's/06's SFT-stage numbers (§4.1),
>   not the full 12-cell matrix Stage 6 originally specified. State this
>   plainly in the report rather than silently produce a partial matrix.

## 0. What this experiment is

Follow-up to `07-nitin-ds-new-sft` (deleted; report at
`../docs/history/07-nitin-ds-new-sft/07-final-comparison.md`), but **not a clean
single-variable replication** the way 06→07 was. Nitin's dataset-generation
repo (`jac-data-gen`) shipped a large volume of *genuinely new* work between
07's pin and now — not more rows of the same file, but four new source types
(OSP idiomize gold slice, js2jac translation corpus, step4 DPO pairs, step4
work-derived SFT pairs) on top of the original py2jac corpus. This phase folds
all of it in. Three research questions, building on 06/07's methodology:

1. **RQ1 — does Spectrum (SNR / Marchenko-Pastur layer selection) replicate on
   a FOURTH dataset?** 04-cpt-sft found a significant win at every stage; 06
   did not replicate it; 07's result is the most recent prior (see §4.1). This
   is the next replication in that chain.
2. **RQ2 — is this multi-source corpus better than 07's single-source one?**
   Measured cell-for-cell against 07's numbers on the *identical* shared
   holdout (a), the same instrument 07 used against 06.
3. **RQ3 — new, not asked by 06/07 — does adding real graph-native (OSP) and
   CRUD-walker (farm) content change what the model learns, beyond a
   same-shape-more-data effect?** 06 and 07 both explicitly verified their
   corpus was flat Python-shaped functions with **zero** OSP archetypes and
   warned not to describe it as graph-native. **That is no longer true here.**
   943 `osp` rows and 1,594 `farm`/`farm_handler` rows contain real
   `node`/`edge`/`walker`/`spawn`/`with entry` Jac — see §1.1. This is a
   genuine framing change, not a continuation of "same shape, better Jac."

**Named confound, stated up front rather than discovered later:** unlike
06→07 (same task shape, different row quality/count), 08 changes dataset
**content type** (multi-source, multi-task, partially graph-native) and
**size** (16,167 vs 07's 6,781 training-pool rows) simultaneously. RQ2's
comparison against 07 is therefore not a clean content-quality isolation the
way 07-vs-06 was — say so in the headline, don't bury it.

## 1. Source data — the corpus, now 7 sources deep

**RESOLVED 2026-09-08/09.** Same repo as 06/07, three commits further:

| Field | Value |
|---|---|
| Origin repo | `https://github.com/chess10kp/jac-data-gen` (same as 06/07) |
| **Commit pin** | **`c95b7563c16851010396170c269f1fdc5327ef15`** — "data(osp): test-verified gold slice (1034 PASS) + mm4 issue pool", 2026-09-01 |
| 07's pin, for contrast | `11fa3f45a0a349337ae4c355708a7e4974b54a36` (2026-08-17) |
| 06's pin, for contrast | `7c25aff3110f526eec59e0123ffe6c0c152cce91` (2026-08-10) |
| Local clone | `~/repos/jac-data-gen` (INTERNAL disk) |
| Diff vs 07's pin | 16,069 files changed, +324,870/−9,368 lines — **not** an incremental update to the file 07 read |

The old `py2jac_dataset_idiomatic.jsonl` (07's only source) barely moved
(9,367→9,371 records) at this pin. The real update is four sources that did
not exist at 07's pin. `1-scaffold/pipeline.jac` reads all seven; its `COMMIT`
constant is the single source of truth.

### 1.1 The 7 sources — what each one is and why it's in or out

| source_type | rows kept | task_kind | what it is |
|---|---:|---|---|
| `js2jac` | 8,674 | translation | JS/TS → Jac translation pairs, `scripts/js2jac_dataset/js2jac_dataset.jsonl` (the 9,828-row master, not the smaller `data/` subset 07 would have found) |
| `composer` | 2,898 | translation | `data/composer_dataset.jsonl` (15,144 rows) — the superset of 07's py2jac file, quality-filtered (§2.2) |
| `farm` | 1,592 | completion | `data/farm_dataset.jsonl` — Jac node archetype + CRUD walkers, **behaviorally round-trip verified** (100% `gate.reason=="round-trip passed"`), prompt synthesized from `manifest` |
| `step4_dpo` | 1,375 | pair_dpo | cross-referenced `step4{b,c,d,e}/dataset_{sonnet,composer}.jsonl` pairs where the two models' `source` tier disagreed (idiomatic vs floor) — the only preference data in the whole repo |
| `osp` | 943 | completion | `data/osp_dataset_pass.jsonl` — chat-format, **model-generated-test-verified, 100% PASS** (`compiler_pass` + `test_pass` both true on every row), fences stripped. **This is real OSP/graph-native Jac** — see the warning below |
| `step4_work` | 683 | translation | `data/step4*/work/*.json` — Python source + idiomatic-Jac target, joined by id against `composer_dataset.jsonl`'s idiomatic rows, used where no DPO pairing existed for that id |
| `farm_handler` | 2 | completion | `data/farm_handler_dataset.jsonl` — 983 rows at source, but content-identical to `farm_dataset.jsonl` (confirmed subset); 981 dropped as `global_duplicate_content`, 2 survive with no farm counterpart |
| **total** | **16,167** | | after all filters, dedup, and leakage checks (§2) |

**Skipped entirely** (verdict from investigation, not laziness):
- `data/golden_client_jac/` (6,210 bare `.jac` files) — no paired prompt JSONL exists (deliberately gitignored upstream); this project's own CPT-v2 result (`../docs/history/03-cpt-only/cpt-2-results.md`) already rejected further CPT-only training. Would need a prompt-pairing pass to be usable at all.
- `data/graph_targets/` — gitignored discovery metadata upstream, not in the repo at this pin. Its only committed descendant, the 540-row `mm4_issue_problems.jsonl` prompt pool, has **no Jac target** (32/540 already appear as `osp` prompts; the other 508 are an input queue, not training data).

### 1.2 Warning that inverts 06/07's warning — read this before writing anything comparing 08 to 07

06's `CONTEXT_BRIEF.md` and 07's both say, verbatim: *"do not describe this
data as graph-native or OSP — 0 archetype declarations, 0 `walker` keyword, 0
`with entry` blocks."* That was true of 06 and 07's single source. **It is
false for 08.** Verified: all 943 `osp` rows use `node`/`walker`/`has`/`spawn`
Jac (1,031/943 have `walker`, 978/943 have `++>`, 245 use typed `edge`); the
1,594 `farm`/`farm_handler` rows define `node` archetypes with CRUD `walker`s.
That is **2,537 of 16,167 rows (15.7%)** of genuinely graph-native content —
the first time this experiment lineage has trained on real OSP material
rather than flat Python-shaped functions. Do not import 06/07's "same shape"
framing into 08's report without re-checking it; it does not hold here.

## 2. Merge pipeline — denylist fix, quality filter, license filter

**RESOLVED 2026-09-09/10** by `1-scaffold/pipeline.jac` (verified end-to-end by
an independent re-check after a mid-run drive disconnect — see §11).

### 2.1 Eval-leakage denylist — applied for the first time in this lineage

`jac-data-gen` ships `evals/function/v1/denylist_ids.txt` (1,404
cluster-expanded ids) specifically to keep training data disjoint from its own
sealed eval set (`evals/function/v1/`, `status: candidate_unvalidated`).
**Neither 06 nor 07 applied it.** Precision on what that means: it does **not**
invalidate 06's or 07's reported numbers in *this* project — their headline
results are scored against this project's own carved holdouts (a) and (b),
never against `jac-data-gen`'s sealed eval set. But their *released training
data* does carry eval-set overlap (measured: composer_dataset 5.1%,
py2jac_dataset_idiomatic 7.8%) — a real hygiene gap if that eval set is ever
used directly, and worth fixing regardless. **08 applies it**: 550 SFT-pool
rows and 103 DPO-pool rows dropped for `eval_denylist` (§2.2's ids are only
checked against the numeric HF-row-id namespace used by `composer`,
`step4_work`, `step4_dpo` — `osp`/`js2jac`/`farm` use unrelated string-id
namespaces and were confirmed, not assumed, to not collide).

### 2.2 Quality-gradient filter on `composer_dataset.jsonl`

`docs/PY2JAC_QUALITY_GRADIENT.md` in `jac-data-gen` documents that a
mutation-strength gate (kill ≥80% of injected mutants) was validated on a
20-record calibration run, then silently dropped in mass production — so
`composer_dataset.jsonl`'s later rows are a measurably weaker tail (median 14
hidden tests vs 21 for the first-half/strong slice). Rather than a chunk
cutoff, 08 computes a **per-row hidden-test count** by joining each id against
its `step4*/work/*.json` entry (`test_blocks` field) and keeps only
`source=="idiomatic"` rows with **`test_count >= 18`** — a threshold between
the documented weak-tail median (14) and strong-tail median (21). Verified in
every surviving `composer`/`step4_work` row's `quality_signals.test_cutoff`
field. Rows with no joinable `work/*.json` (no computable test count) are
excluded rather than assumed good: 2,850 rows, logged as
`no_work_join_no_test_signal`. 3,108 more rows were dropped for scoring below
18. This is the plausible root-cause fix for the finding in 07's own
report (`../docs/history/07-nitin-ds-new-sft/07-final-comparison.md`) that 07 (larger pool, weaker average
quality) underperformed 06 despite more rows.

### 2.3 License filter on `js2jac`

Kept genuinely permissive licenses only: MIT (8,351), APACHE-2.0 (170),
NOASSERTION (6), CC0-1.0 (4), BSD-2-CLAUSE (33), UNLICENSE (110) — 8,674
total. Dropped GPL-3.0/AGPL-3.0/OTHER/blank-spdx rows as `license_not_permissive`
(1,016 rows).

### 2.4 What "DPO axis" means here — different from 07, not inherited

07's DPO axis (inherited from 06) was **synthetic correctness**: take a known-
good file, LLM-author a variant with one subtly injected bug. 08's axis is
different and **not synthetic**: `step4_dpo` pairs come from two different
models (sonnet, composer) independently translating the *same* Python
function, where one model's output was later classified `idiomatic` and the
other's `floor` by the upstream pipeline's own gate. **Axis: idiomaticity**,
not correctness — both sides are real model output, both compile, the
difference is Jac-language idiom quality, not an injected defect. Document
this distinction in any report; conflating it with 07's bug-injection axis
would misdescribe what the DPO training signal actually teaches.

## 3. Base checkpoint — unchanged, explicit standing instruction

`models/qwen-q4` (registry label "Qwen · BASE") — the SAME base checkpoint
used by 04-cpt-sft, 06, and 07. Holding the base fixed across every phase in
this lineage is what keeps the cross-phase dataset comparison clean.

**Spectrum layer selection: reuse verbatim, do NOT re-run the SNR scan.**
`2-train/spectrum/spectrum_layers.json` is copied unchanged
from 07 (itself sha256-identical to 06's and 04-cpt-sft's) —
`[0, 22, 23, 27, 30, 34, 36, 37, 38, 39, 41, 42, 43, 44, 45, 47]`. Verify the
sha256 before trusting it rather than re-asserting from memory.

## 4. Eval — both holdouts, four result cells per stage

Same convention as 06/07. Per stage (SFT-final, DPO-best, DPO-final), for
EACH arm (stock, spectrum), eval against:

**(a) The existing shared holdout — reuse UNCHANGED, do not regenerate:**
`dataset/holdout_a_shared855.jsonl` — a byte-identical copy of 04-cpt-sft's
`sft_fresh_probe/dataset/sft/valid.jsonl` (that directory is deleted), 1,428
rows, 855 code-graded. Makes every number directly comparable to 06's and
07's cells.

**(b) 07's own holdout, reused verbatim, NOT regenerated from 08's pool.**
`dataset/nitin_holdout.jsonl` and `nitin_holdout_eval.jsonl` were **copied
byte-identically from `07-nitin-ds-new-sft/dataset/`**, not carved fresh from
08's multi-source pool. This was a deliberate choice (§9 of the build): it
keeps holdout (b) comparable to 07's own holdout-B numbers, at the cost of
holdout (b) no longer being in-distribution with 08's *new* source types
(osp/js2jac/farm never contribute holdout-(b) rows — it is still 100%
py2jac-shaped content). Read holdout (b) accordingly: it measures "did 08's
training help on 07's own task shape," not "did 08's training help across all
of 08's task shapes." There is no in-distribution holdout across all 7
sources in this phase — flag this as a real gap if RQ3 needs one.

**Leakage: verified zero.** Four independent surfaces checked (numeric HF-row
id, normalized `jac` text hash, normalized `jac_rejected` text hash, normalized
holdout-prompt text) against the pool as written to disk, both pre-split and
post-split. Note: `pipeline.jac` and `copy_holdout.jac` check against holdout
(b) only. Holdout (a) is not checked by any 08 script; an exact prompt/target
match of `dataset/sft/{train,valid}.jsonl` against it was run afterwards
(2026-09-11) and found 0 hits. 530 candidate rows and 117 DPO pairs were caught and dropped by
this check before it reached zero (`holdout_leak` in the rejected-reason
breakdown, §5 of `experiments/08-nitin-new2-ds/docs/corpus-triage-report.md`) — the leak-check did
real work, it did not just confirm an empty set.

### 4.1 The baselines this phase is measured against

All of the following are **prior results**, not this phase's.

**07-nitin-ds-new-sft, holdout A (the shared 855)** —
`../docs/history/07-nitin-ds-new-sft/07-final-comparison.md` §1. 07 did
**not** beat 06 (0/6 significant, all 6 point estimates favor 06) despite a
larger pool; that finding is the direct motivation for §2.2's quality filter.

| Stage | Stock | Spectrum |
|---|---|---|
| Base (untrained) | 10.5% (90/855) | 10.5% (90/855) |
| SFT-final | 68.9% (589/855) | 72.4% (619/855) |
| DPO-best | 68.2% (583/855) | 73.0% (624/855) |
| DPO-final | 66.3% (567/855) | 71.0% (607/855) |

07 on its own holdout (b), which 08 reuses: SFT-final 98.2% (840/855) for both
arms.

**06-nitin-ds-sft, holdout A** —
`../docs/history/06-nitin-ds-sft/2026-08-final-comparison.md` §1:

| Stage | Stock | Spectrum |
|---|---|---|
| Base (untrained) | 10.5% (90/855) | 10.5% (90/855) |
| SFT-final | 72.4% (619/855) | 74.4% (636/855) |
| DPO-best | 72.4% (619/855) | 73.8% (631/855) |
| DPO-final | 68.2% (583/855) | 74.4% (636/855) |

**04-cpt-sft fresh arm, holdout A** — SFT stock 597/855, spectrum 639/855;
DPO-best stock 597, spectrum 634; DPO-final stock 531, spectrum 622.

**Training-pool sizes, the size confound restated:** 04-cpt-sft fresh 8,100
SFT rows; 06: 5,474; 07: 6,781; **08: 14,792 SFT + 1,375 DPO** — 2.18× 07's
SFT pool, and multi-source rather than single-source. This is the biggest pool
in the lineage by a wide margin; report it as a headline confound exactly like
07 did for its own 1.24× jump over 06.

### 4.2 Eval sweep sizing — unchanged from 06/07

Interim/sweep checkpoints eval against the **first 100 rows** of the holdout
(`SUBSET=100` → `JAC_EVAL_LIMIT`, not a random slice; on holdout (a) those 100
rows are all `code_gen`); final checkpoints (SFT-final, DPO-best, DPO-final)
always eval on BOTH holdouts at full size. Never substitute the sweep subset
for a headline number.

## 5. Everything is Jac — carried forward from 07

All pipeline and driver scripts in this phase are `.jac`. `.sh` wrappers stay
`.sh` but invoke `jac run <driver>.jac <flags>`. Use `with entry:__main__ { }`,
never a bare `with entry { }`.

## 6. Training — copy-adapted from 07, paths retargeted only

`2-train/stock` and `2-train` are 07's trees with every
`07-nitin-ds-new-sft` path reference substituted for `08-nitin-new2-ds`.
Hyperparameters unchanged: `iters: 8200`, `batch_size: 1`,
`learning_rate: 2.0e-5` cosine (warmup 820), `num_layers: 16`, LoRA rank 16 /
scale 2.0 / dropout 0.05, `max_seq_length: 3072`, `seed: 42`,
`mask_prompt: true`, `grad_checkpoint: true`. Verified: `jac check` clean on
all 5 `.jac` files, zero leftover `07-nitin-ds-new-sft` string references,
`dpo_fixed_train.jac` byte-identical across both probes (md5 matches 07's).

**Three open items inherited unchanged from 07, not yet re-decided:**

1. Both DPO run scripts default `HOLDOUT` to
   `dataset/holdout_a_shared855.jsonl` (holdout (a)) for the in-loop
   collapse-gate baseline lookup — same as 07. Confirm this is still wanted
   before the first launch.
2. That lookup filters on `total == 855`; pointing a DPO training run at
   holdout (b) instead would silently drop the collapse gate to its 30%
   absolute floor. Preserved verbatim from 07, comment intact.
3. `3-eval/eval_sft_spectrum.sh:85` and `eval_dpo_spectrum.sh:85`
   still echo **06** (636/855, 74.4%) as "the incumbent" — inherited from 07
   unchanged. 08's natural comparison baseline is 07, not 06; update these
   echo lines once 07's actual final numbers are confirmed from its report,
   or leave as informational only (they don't affect scoring).

Three corrections inherited from 06/07, still authoritative: no
`spectrum/configs/dpo_spectrum.yaml` (env-var defaults in the runner); the
functional harness is `3-eval/eval_functional.jac` (moved here from
04-cpt-sft's `sft_cptv2_probe/jacgen/` when 04 was deleted; grading logic
unchanged) — never change how it grades; the stock DPO runner is
`run_dpo_nofuse.sh`, never `run_dpo.sh` (fuse re-quantizes and discards the
SFT delta).

**Preflight hazard, unchanged:** `pgrep -f "jac start"; pgrep -f mlx_lm` must
return nothing before any launch.

## 7. Decontamination — reused shingle machinery, extended

`1-scaffold/pipeline.jac` does exact and normalized-hash dedup (comments
stripped, whitespace collapsed) per source and across sources, plus the
eval-denylist pass (§2.1) that neither 06 nor 07 ran. It does **not** carry
06/07's 14-token-shingle near-duplicate check (verified by reading the script
2026-09-11; earlier drafts of this brief said it did). Global cross-source
dedup caught 1,006 rows — mostly `farm_handler` being a near-total content
subset of `farm` (§1.1).

**No silent caps.** Every drop is counted and attributed by reason in
`experiments/08-nitin-new2-ds/docs/corpus-triage-report.md`.

## 8. Live monitoring — unchanged convention

The SFT runners' watchdog regenerates PNGs (train/val loss, LR, throughput,
memory) into the arm's results dir on every poll via
`3-eval/plot_metrics.jac`. `3-eval/plot_progress.jac` regenerates loss and
eval-curve PNGs from `train.log` + `metrics_functional.jsonl` on demand,
conventionally written to `<results dir>/plots/`.

## 9. Directory layout

```
experiments/08-nitin-new2-ds/   (layout after the 2026-09-11 repo flatten)
  report.md                 <- final comparison report
  adapter/                  <- final spectrum SFT adapter (safetensors gitignored)
  results/                  <- train.log, verify/key-assertion logs, base/final evals, plots; holdoutB/
  docs/
    CONTEXT_BRIEF.md        <- this file
    README.md  spec.md  workflow.md  dataset-structure.md  corpus-triage-report.md
    funnel/                 <- stats.json, release.json
  dataset/
    holdout_a_shared855.jsonl <- holdout (a), 1,428 rows / 855 code-graded, copy of 04's
    candidate_pool.jsonl    <- 16,167 rows, clean/deduped/decontaminated/leak-checked
    nitin_holdout.jsonl     <- REUSED VERBATIM from 07 (855 rows), not carved from 08's pool
    nitin_holdout_eval.jsonl<- same, with authored instructions (07's, byte-identical)
    sft_train.jsonl (14,792)  dpo_train.jsonl (1,375)
    sft/{train,valid}.jsonl (12,570 / 2,217 after the NaN guard)  dpo/{train,valid}.jsonl (1,169 / 206)
    rejected/{sft,dpo}/     <- every drop, with a reason (9,404 sft / 224 dpo)

repo root (shared pipeline, run with EXP=experiments/08-nitin-new2-ds, the default):
  1-scaffold/               <- pipeline.jac, copy_holdout.jac, release.jac, prep_training_dirs.sh, nan_guard.jac
  2-train/                  <- run_sft_spectrum.sh, spectrum/, configs/; stock/ (trailing-16 arm), dpo/
  3-eval/                   <- eval_functional.jac, eval_*.sh, plot_*.jac, gen/grade_eval_detail.jac, grade_reference.jac
```

Both arms share ONE `dataset/` — they train on identical data by design.

**Adapter names, matching the lineage's `-nitin` convention** (kept as-is,
not renamed to `-nitin2`, since the scaffold was a mechanical path
substitution from 07 — revisit if a distinct adapter name is wanted before
first launch):

| Arm | SFT adapter | DPO adapter | DPO-best |
|---|---|---|---|
| stock | `adapters/sft-on-nitin` (now `stock/adapter`) | `adapters/dpo-on-sft-nitin-nofuse` (now `stock-dpo/adapter`) | `…-nofuse-best` (now `stock-dpo/adapter-best`) |
| spectrum | `adapters/sft-on-nitin-spectrum` (now `adapter`) | `adapters/dpo-on-sft-nitin-spectrum` (now `dpo/adapter`) | `…-spectrum-best` (now `dpo/adapter-best`) |

Only `experiments/08-nitin-new2-ds/adapter` exists on disk now.

## 10. Sequencing (respect the dependency order)

As planned (12 cells), then as actually run after the 2026-09-11 scope cut:

```
[DONE] pin corpus @ c95b7563 -> 7-source merge -> denylist + quality + license filters
  -> dedup + decontam -> holdout(b) reused from 07 -> leak-check -> 85/15 splits
[DONE] stock SFT (trained, never evaluated, adapter since deleted)
[DONE] spectrum SFT -> eval on holdout (a) and (b) -> experiments/08-nitin-new2-ds/report.md
[CUT ] DPO x2 arms, stock-arm eval, 12-cell matrix
```

## 11. Failure modes already paid for — do not rediscover these

| Symptom | Cause |
|---|---|
| DPO collapses to ~2–12% pass | `mlx_lm.fuse` on an int4 model re-quantizes and discards the SFT LoRA delta |
| Spectrum adapter loads but scores like a partial model | `adapter_config.json` not rewritten; `load_weights(strict=False)` silently drops out-of-slice blocks |
| Holdout-B eval "finishes" in 30 seconds | holdout rows have no `messages` field — nothing was ever generated (06 §4 incident 4) |
| Best-checkpoint tracker resets after a crash-resume | the watchdog's "best" comparison does not persist across a restart |
| Bus error / `tee: Input/output error` mid-run | the external drive dropping off the bus — **hit again during 08's own dataset build** (2026-09-09): `.git` and `dataset/` both went `Input/output error` mid-pipeline-run; drive returned healthy (SMART "Verified") after a physical reseat; the partial `candidate_pool.jsonl` on disk was re-verified byte-by-byte before trusting it (funnel arithmetic closed exactly, zero duplicate ids, zero malformed records) rather than assumed good. Verify every artifact after a remount before resuming, every time — this is now the third occurrence in this lineage (06 incidents 1/3, and this one) |
| DPO OOM at 37GB+ free RAM | macOS GPU wired-memory ceiling, not system RAM; policy + reference both resident |
| Behavioral gate always passes | `gate_class="behavioral"` with an empty `expected_output` |
| Script runs as a side effect of being imported | `with entry { }` instead of `with entry:__main__ { }` |
| Cross-directory Jac import dies at runtime but passes `jac check` | relative import with no known parent package — inline the shared logic |
| A silently-partial merge pipeline looks complete | verify funnel arithmetic closes end-to-end (stage-by-stage subtraction, not just a final total) before trusting any multi-source merge output, especially one resumed after an interruption |
| SFT train loss goes `nan` partway through and never recovers, `mlx_lm.lora` does NOT crash or exit non-zero | `mask_prompt: true` + a row whose PROMPT ALONE exceeds `max_seq_length` (3072): truncation keeps only prompt tokens, zero unmasked completion tokens survive, cross-entropy over an empty target is NaN, and the NaN gradient poisons Adam's momentum/variance state permanently — every subsequent step stays NaN. Hit live during 08's stock-arm SFT launch (2026-09-10): 5 `js2jac` rows (full_len up to 7521, 2.4x the limit) had prompt-alone length ≥3072; first hit at iter ~2300 (matches the `[WARNING] ... longest sentence 7521 ... truncated` line immediately prior). `mlx_lm.lora`'s own per-report loss line is the only signal — no exception, no stall (log keeps growing), so the watchdog's stall/OOM detectors both miss it; must explicitly grep segment logs for `nan` too. Fix applied: scan `dataset/sft/{train,valid}.jsonl` with the real tokenizer (`mlx_lm.tokenizer_utils.load`), compute `len(apply_chat_template(msgs[:last_assistant_idx], add_generation_prompt=True))` per row, drop any row where that's ≥ `max_seq_length`. 3 dropped from train, 2 from valid, all `js2jac` (the `*.jsonl.pre-nanfix.bak` backups made at the time were removed in the 2026-09-11 cleanup). This check is now `1-scaffold/nan_guard.jac`, run automatically by `prep_training_dirs.sh`. Check `dataset/dpo/{train,valid}.jsonl` separately before any DPO stage (smaller `DPO_MAXLEN=512`, not verified clean; `nan_guard.jac` only reads `messages`, so it does not cover DPO rows). |
