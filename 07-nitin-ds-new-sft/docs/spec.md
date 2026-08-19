# 07-nitin-ds-new-sft — Design Spec

Status: scaffolded, pre-corpus-pin.
Date: 2026-08-18.
Companion to [`../CONTEXT_BRIEF.md`](../CONTEXT_BRIEF.md) (settled facts) and
[`workflow.md`](workflow.md) (the runbook). This file is the umbrella: what
gets built, from what, measured how, and what result counts as a win.

## 1. Purpose

One training battery, two independent research questions. Direct follow-up to
`06-nitin-ds-sft`, on the **successor corpus** from the same author.

1. **RQ1 — does Spectrum replicate on a third dataset?** `04-cpt-sft`'s fresh
   arm found SNR-based layer selection beat `mlx_lm`'s default trailing-16 slice
   at every stage, significantly. `06-nitin-ds-sft` ran the identical probe on an
   independently-sourced dataset and got a mixed result — 1 significant win
   (DPO-final, holdout A, +6.2pp), 1 significant **loss** (SFT-final, holdout
   B, −1.4pp), 4 nulls — clearing the ≥4-of-6 bar on neither holdout. 06's
   honest reading was that Spectrum's advantage is **not dataset-invariant**,
   and that a third replication is needed before settling either direction.
2. **RQ2 — is the new dataset better than 06's?** Both arms are scored on
   `04-cpt-sft`'s **unchanged** 855-row code-graded holdout, which makes each
   cell directly comparable to 06's number for the same arm and stage — and,
   through 06, to the original jacgen2 fresh-arm number too. That is a three-way
   dataset comparison on one fixed instrument.

Only two things differ from 06: the **dataset**, and the fact that all scripts
are now **Jac** rather than a Jac/Python mix (a syntax change with no intended
semantic effect — see `CONTEXT_BRIEF.md` §5). Base checkpoint, hyperparameters,
seed, LoRA geometry, eval harness, and holdout (a) are held fixed. That is what
makes this a replication rather than a confounded new experiment.

## 2. Source data

| Field | Value |
|---|---|
| Origin | `https://github.com/chess10kp/jac-data-gen` — **the same repo 06 used, at a newer commit** |
| Commit pin | **`11fa3f45a0a349337ae4c355708a7e4974b54a36`** ("Add idiomatic py2jac corpus (9,367 records)", 2026-08-17) |
| 06's pin | `7c25aff3110f526eec59e0123ffe6c0c152cce91` (2026-08-10) |
| Local clone | `~/repos/jac-data-gen` (internal disk) |
| Shape on disk | `data/py2jac_dataset_idiomatic.jsonl` — **one JSONL**, not loose `.jac` files |
| Record count | **9,367** (06: 7,627) |
| Record schema | `{id: int, entrypoint: str, source: str, jac: str}` |
| Per-record content | one docstring + one top-level `def name(params) -> type { ... }` |
| Metadata | `source` is `"idiomatic"` on all rows; no split, no labels, no provenance |

Cite the commit pin everywhere: in `origin` on every emitted example
(`<repo>:<commit>:<path>`), in the triage report, and in the final comparison
report. The placeholders live in `scripts/pipeline.jac` (`CORPUS`, `SRC`,
`SRC_REL`, `COMMIT`) and `scripts/compile_check.jac` (`SRC`) — those are the
single source of truth the pipeline reads. `origin` is
`jac-data-gen:11fa3f45…:data/py2jac_dataset_idiomatic.jsonl#id=<id>`.

**I/O note.** The project's `/Volumes/ExtremePro/...` external drive measured
~4.3 MB/s — a genuine hardware/contention bottleneck confirmed via raw `dd`,
not an app bug. Bulk operations over the corpus (compile-checking, shingling,
dedup) run against the internal clone; `scripts/compile_check.jac` writes its
results to `/tmp/nitin_new_triage/`. Only final JSONL releases, plots, and
reports are written to the external project tree.

### 2.1 What this corpus actually is — UNVERIFIED

06's corpus was verified by sampling to be **flat Python-to-Jac function
conversions** — dicts, lists, loops, early returns, string handling, with no
`node`, `edge`, `walker`, or OSP traversal anywhere. Everything downstream in
06 followed from that.

**ANSWERED 2026-08-18 by a full-corpus scan** (all 9,367 records, not a sample) —
see `../CONTEXT_BRIEF.md` §1.1 for the tables. Result: **same shape as 06's**.
Every record has exactly one top-level `def`; there are **zero** OSP archetype
declarations (`node X {` / `edge X {` / `walker X {`), zero `walker` keywords and
zero `with entry` blocks. The 92 records mentioning node/edge/root/here are all
incidental — docstring prose and identifiers inside tree/graph *algorithms*.

So every decision in §3 stands as inherited: the corpus maps onto exactly one
existing task type, `python_to_jac_function` (category `conversion`); it cannot
support the flagship idiomatic DPO axis; and it must not be described as
"graph-native" anywhere.

**What "idiomatic" actually means here** — richer *Jac language* idiom inside the
same flat-function shape, not a new paradigm: 9,359/9,367 carry an explicit `->`
return type, 2,566 use union types, 479 use `isinstance` narrowing, 143 use
`match`, 120 use `lambda`. That is precisely the RQ2 contrast: **same task shape,
better-written Jac**.

## 3. Decisions locked / inherited

| Decision | Choice | Status |
|---|---|---|
| Base checkpoint | `models/qwen-q4` ("Qwen · BASE") — the same checkpoint 06 and `04-cpt-sft`'s fresh arm used. Not `qwen-cpt-v1`, not any tuned checkpoint. | **Locked** (standing user instruction; also the only choice that keeps cross-phase comparison clean) |
| Arms | Two: **stock** (trailing-16, `mlx_lm` default blocks 32–47) and **spectrum** (SNR picks). No CPT arm. | **Locked** |
| Spectrum layer selection | Reuse `spectrum_probe/spectrum/configs/spectrum_layers.json` verbatim — sha256-identical to 06's and to `04-cpt-sft`'s. Do not re-run `snr_scan` / `layer_select`: the SNR scan is a property of the base model's *weights*, not the training data, and the base is unchanged. | **Locked** |
| Script language | **All `.jac`.** `.sh` wrappers invoke `jac run <driver>.jac`. | **Locked** (explicit user instruction for this phase) |
| SFT task framing | Reverse-instruction: one NL instruction authored from each file's docstring + signature; completion = the file's Jac code verbatim. | **Inherited from 06, not reconfirmed** — depends on §2.1 |
| DPO axis | **Correctness only** (`dpo-plan.md` §2.3). `chosen` = the file's Jac as-is; `rejected` = the same function with one subtly-introduced logic bug. See §3.1. | **Inherited from 06, not reconfirmed** — depends on §2.1 |
| DPO axes declined | `graph_native`, `idiomatic`, `auth_security`, `typing` — none were expressible in 06's corpus. | **Inherited**; reopens if §2.1 finds graph-native material |
| Eval holdouts | Both. (a) `04-cpt-sft`'s existing 855-row code-graded holdout, reused **unchanged**; (b) a new holdout carved from this corpus, frozen before any generation, **with authored instructions** (§4.2). Neither optional. | **Locked** |
| Training recipe | Copy-adapted from 06, retargeting **only** data / adapter / results paths. `iters: 8200`, `batch_size: 1`, `learning_rate: 2.0e-5` cosine (warmup 820), `num_layers: 16`, LoRA rank 16 / scale 2.0 / dropout 0.05, `max_seq_length: 3072`, `seed: 42`, `mask_prompt: true`, `grad_checkpoint: true` — verbatim. | **Locked** |
| Generation transport | Batch handoff via dispatched Claude Code Agents (no API key in this environment). Reverse-instruction authoring → **Opus** (bulk). Buggy-variant authoring → **Fable** (precision). | **Locked** |
| Statistical bar | Two-proportion z-test, p < 0.05 two-sided; paired McNemar where both sides answer identical items. Same method as every prior comparison in this project. Pre-registered in §7. | **Locked** |

### 3.1 Why the correctness axis (inherited reasoning)

`dpo-plan.md`'s flagship axis is chosen = graph-native, rejected =
Python-shaped. It did not fit 06's content: a pure utility function has nothing
to graph-ify, so a "rejected" side would have to be manufactured out of
nothing, and the preference signal would be about the generator's imagination
rather than about Jac.

The correctness axis is the one axis a corpus of flat functions genuinely
supports:

- **chosen** — the file's Jac code as-is (treated as correct/functional; the
  triage compile gate is the evidence, and its limits are stated in §6).
- **rejected** — the same function with exactly one subtly-introduced bug:
  off-by-one, flipped comparison operator, swapped branch bodies, mutated
  default argument, inverted accumulator, dropped edge-case guard.
- **Hard requirement: both sides must still compile** (`jac run` exit 0). The
  bug must be a *behavioural* divergence, not a syntax break. A pair whose
  rejected side fails to compile teaches the model to prefer "parses" over "is
  correct" — it gets rejected at collect time with the reason logged.

Documented as a deliberate scope narrowing, not an oversight. If §2.1 finds
graph-native material in the new corpus, revisit this before Stage 2.

## 4. Design — 2 arms × 2 holdouts × 3 stages

### 4.1 The arms

| Arm | LoRA target blocks | Runner lineage |
|---|---|---|
| **stock** | trailing 16 (32–47), `mlx_lm` default | `run_sft.sh` + `run_dpo_nofuse.sh` |
| **spectrum** | `[0, 22, 23, 27, 30, 34, 36, 37, 38, 39, 41, 42, 43, 44, 45, 47]` (11/16 overlap with trailing-16) | `run_sft_spectrum.sh` + `run_dpo_spectrum.sh` |

Both arms are the same 16-block *capacity* — **281.838M trainable parameters**,
enforced by `spectrum_lora_layers.jac --verify-layers` as a hard preflight gate.
A differing count means the selection changed capacity rather than placement,
which invalidates the comparison outright. Keep the gate; never `SKIP_VERIFY=1`
past it.

### 4.2 The holdouts

**(a) The shared holdout — reused unchanged, never regenerated.**

`04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl` — 1,428 rows,
**855 code-graded** (the `behavioral` + `compile_only` gate classes; the
remaining 573 are `prose_lexical` `explanation`/`documentation` rows the
functional harness does not score). Verified composition:

| slice | n |
|---|---:|
| conversion / behavioral | 322 |
| code_gen / compile_only | 227 |
| trajectory / compile_only | 113 |
| code_gen / behavioral | 86 |
| trajectory / behavioral | 70 |
| debug / behavioral | 33 |
| debug / compile_only | 3 |
| migration / behavioral | 1 |
| **code-graded total** | **855** |

`sft_fresh_probe/dataset/dpo/valid.jsonl` (115 rows) is **not** a scored eval
set — nothing reads it for scoring; it is `mlx_lm_lora`'s training-time
validation split, consumed implicitly via `--data .../dataset/dpo`. All three
stages are scored on the same 855 rows.

**The baselines this makes available** (all n = 855, all *prior* results):

| Stage | 06 stock | 06 spectrum | 04-cpt-sft fresh stock | 04-cpt-sft fresh spectrum |
|---|---|---|---|---|
| Base (untrained) | 10.5% (90) | 10.5% (90) | — | — |
| SFT-final | 72.4% (619) | 74.4% (636) | 69.8% (597) | 74.7% (639) |
| DPO-best | 72.4% (619) | 73.8% (631) | 69.8% (597) | 74.2% (634) |
| DPO-final | 68.2% (583) | 74.4% (636) | 62.1% (531) | 72.7% (622) |

Sources: `06-nitin-ds-sft/docs/reports/2026-08-final-comparison.md` §1 and
`04-cpt-sft/docs/reports/2026-08-spectrum-vs-stock-comparison.md` §1.

**(b) New holdout carved from this corpus** — `dataset/nitin_holdout.jsonl`.

- Frozen **before** any generation, held out of the training pool entirely —
  enforced by id-exclusion at manifest build (`prepare_batches.jac` asserts it),
  not assumed.
- **Size: exactly 855 — a deliberate constant matched to holdout (a) and to 06**
  ("Same 855-row size (matched deliberately at holdout-carve time)", 06's final
  report §2). `scripts/pipeline.jac` hard-codes `HOLD_N = 855` and **raises** if
  the clean pool is under 4,000 rows rather than silently carving a smaller
  holdout the way 06's fallback would have. Do not rescale 855 to the new
  corpus's size; equal-sized holdout columns are what keep the cross-phase
  comparison like-for-like.
- **It must carry authored instructions.** 06 froze its holdout from the raw
  candidate pool, so the rows had `jac_code` + `docstring` but no `messages`
  field — and `eval_functional.jac` had nothing to prompt the model with. The
  first holdout-B eval "completed" in under 30 seconds having never run. 06
  recovered by reverse-authoring instructions for all 855 rows into a separate
  `nitin_holdout_eval.jsonl`. **This phase plans that pass up front**;
  `scripts/gen_eval_detail.jac` already points holdout "B" at
  `dataset/nitin_holdout_eval.jsonl`.
- Tests in-distribution performance on this corpus's own distribution.
  Comparable **stock vs spectrum within itself**; *not* comparable to (a)'s
  numbers, and not comparable to 06's holdout B either (different item set).
  Any report lining them up side by side must say so.

**Expect a ceiling on (b).** 06's holdout B ran at 96–99% for both arms after
training — a single category, `compile_only` gate, essentially the same
distribution as its own training pool. That ceiling compresses the arm-vs-arm
difference and makes (b) a weak discriminator. Read (b) accordingly, and let
(a) carry the decision if (b) saturates again.

### 4.3 The matrix

3 stages (SFT-final, DPO-best, DPO-final) × 2 arms × 2 holdouts = **12 result
cells**, plus the 6 existing 06 cells and the 6 `04-cpt-sft` fresh-arm cells as
cross-phase references.

|  | holdout (a) — 855 shared | holdout (b) — new corpus, n = TBD |
|---|---|---|
| SFT · stock | cell A1 | cell B1 |
| SFT · spectrum | cell A2 | cell B2 |
| DPO-best · stock | cell A3 | cell B3 |
| DPO-best · spectrum | cell A4 | cell B4 |
| DPO-final · stock | cell A5 | cell B5 |
| DPO-final · spectrum | cell A6 | cell B6 |

Three families of comparison:

- **Within-column, arm-vs-arm** (A2−A1, A4−A3, A6−A5; B2−B1, B4−B3, B6−B5) —
  research question 1, does Spectrum replicate. Paired McNemar: both arms answer
  identical items.
- **Cross-phase, same cell** (A1 vs 06's 619/855, A2 vs 636/855, …) — research
  question 1, is the new dataset better. Also paired McNemar: same 855 items,
  same harness, different training data. Report the 04-cpt-sft column alongside
  as the three-way lineage view.
- **Column (a) vs column (b)** — context only. Different item sets, different
  difficulty; a delta here is descriptive, never a test.

## 5. Reuse vs new, relative to 06

Everything expensive already exists. The inventory is explicit so nobody
rebuilds a battle-tested script "cleanly".

### 5.1 Reused (copied from 06, paths retargeted, ported to Jac, logic unchanged)

| Artifact | This phase | Retarget |
|---|---|---|
| SFT runner, stock | `stock_probe/run_sft.sh` | data/adapter/results paths |
| SFT runner, spectrum | `spectrum_probe/run_sft_spectrum.sh` | data/adapter/results paths |
| DPO runner, stock | `stock_probe/run_dpo_nofuse.sh` — **not** `run_dpo.sh`, see §5.3 | data/adapter/results paths |
| DPO runner, spectrum | `spectrum_probe/run_dpo_spectrum.sh` | data/adapter/results paths |
| Eval, SFT stock | `stock_probe/eval_sft_sweep.sh` | adapter/results/holdout paths |
| Eval, SFT spectrum | `spectrum_probe/eval_sft_spectrum.sh` | adapter/results/holdout paths |
| Eval, DPO stock | `stock_probe/eval_dpo_nofuse.sh` | adapter/results/holdout paths |
| Eval, DPO spectrum | `spectrum_probe/eval_dpo_spectrum.sh` | adapter/results/holdout paths |
| DPO driver, stock | `{stock,spectrum}_probe/dpo_fixed_train.jac` | nothing (two byte-identical copies; the spectrum driver imports its sibling from PROBE_DIR, so the second copy is required, not incidental) |
| Spectrum SFT driver | `spectrum_probe/spectrum/spectrum_lora_layers.jac` | nothing |
| Spectrum DPO driver | `spectrum_probe/spectrum/dpo_spectrum_train.jac` | nothing |
| Adapter-config rewriter | `spectrum_probe/spectrum/adapter_config_fix.jac` | nothing |
| Layer selection | `spectrum_probe/spectrum/configs/spectrum_layers.json` | **nothing — byte-identical, sha256-verified against 06** |
| SFT hyperparameters | `stock_probe/configs/sft.yaml`, `spectrum_probe/spectrum/configs/sft_spectrum.yaml` | `data`, `adapter_path` only |
| DPO hyperparameters | `{stock,spectrum}_probe/configs/dpo_lora.yaml` + the env-var defaults in the DPO runners | nothing |
| Functional harness | `04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac` | nothing — fully env-var driven (`JAC_EVAL_*`, `JAC_HOLDOUT`). **Point at it, do not copy it.** |
| Triage + decontam | `scripts/pipeline.jac`, `scripts/compile_check.jac` | corpus coordinates (`[TODO]`s), `/tmp/nitin_new_triage/`, version + run tags |
| Batch handoff | `scripts/prepare_batches.jac`, `scripts/collect.jac` | paths; shard counts are `[TODO]` until the pool size is known |
| Train/valid split | `scripts/prep_training_dirs.sh` | paths (85/15, seed 42 unchanged) |
| Failure analysis | `scripts/gen_eval_detail.jac`, `scripts/grade_eval_detail.jac`, `scripts/grade_reference.jac` | holdout paths |
| Live plots | `scripts/plot_progress.jac` | nothing — `--train-log` / `--out` driven |

### 5.2 New in this phase (the only things actually written)

- The Jac port of every `.py` driver and pipeline script (syntax only —
  `CONTEXT_BRIEF.md` §5).
- Corpus triage + dedup + decontam + holdout carve against the **new** corpus.
- Reverse-instruction SFT-pair authoring, **including the holdout rows** so
  holdout (b) is evaluable (§4.2).
- Buggy-variant DPO-pair authoring with the dual compile gate.
- The comparison report.

### 5.3 Corrections carried forward from 06 (verified on disk there)

1. **`spectrum/configs/dpo_spectrum.yaml` does not exist.** DPO-spectrum
   hyperparameters are env-var defaults inside `run_dpo_spectrum.sh`
   (`DPO_ITERS=250`, `DPO_LR=1e-6`, `DPO_BETA=0.1`, `DPO_MAXLEN=512`) plus the
   shared `configs/dpo_lora.yaml` (rank 16 / scale 2.0 / dropout 0.05,
   `fuse: false`). Copying the runner carries them.
2. **There is exactly one copy of the functional harness in the repo:**
   `04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac`. Every eval script
   here points at it. Nothing to diff, nothing to copy.
3. **The stock DPO runner is `run_dpo_nofuse.sh`, not `run_dpo.sh`.**
   `run_dpo.sh`'s own results file reads **12% (107/855)** — the collapsed
   `mlx_lm.fuse` path: fusing an SFT LoRA delta into int4-quantized weights
   re-quantizes afterward and silently discards the delta. `run_dpo.sh` is
   deliberately absent from this tree.

Not needed here, stated so nobody goes looking: `merge_frozen_keys.py` exists
only under `sft_cptv2_probe/` and merges frozen CPT-v2 keys into a trained
adapter. There are no frozen prior keys on a fresh base, so neither arm uses it
and the `mx.load` in-place-truncation bug cannot occur on this code path.

## 6. Decontamination — required, machinery already ported

`scripts/pipeline.jac` carries `jacgen2/decontam_v2.jac`'s algorithm: 14-token
shingles over comment-stripped, whitespace-tokenized code; contaminated at
≥ 0.5 shingle overlap; plus a relaxed identifier-only 10-shingle pass and a
function-name-collision audit dropping at ident-Jaccard ≥ 0.5.

Three passes, all mandatory:

1. **Within-corpus dedup.** Exact normalized-token hash first, then greedy
   shingle near-dup. Large mined corpora cluster hard around common utility
   patterns.
2. **Training pool vs existing holdouts.** Reference set as shipped:
   `04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl`,
   `.../dataset/dpo/valid.jsonl`, and
   `01-sft-dpo/dataset/eval_holdout/{conversion,graph_conversion}.jsonl`.
   `[TODO: decide whether 06's `dataset/nitin_holdout.jsonl` joins this list —
   it is only a *required* target if 07 also reports numbers on 06's holdout.
   Left off by default so the triage logic stays mechanically identical to 06's;
   the TODO is inline in `scripts/pipeline.jac`.]`
3. **New holdout vs everything.** `nitin_holdout.jsonl` is leak-checked against
   the remaining training pool (otherwise it is *labeled* a holdout rather than
   being one) and against the (a) holdouts — a cross-holdout near-dup makes the
   two holdout columns statistically non-independent, which would quietly
   invalidate treating them as two separate tests.

**No silent caps.** Every drop is counted and attributed by reason in
`docs/reports/corpus-triage-report.md` and written under `dataset/rejected/`.

**Stated limitation.** The `chosen` side of every DPO pair is assumed correct on
the strength of a compile gate plus the corpus author's transpiler. Nothing
proves behavioural equivalence to the original Python — this corpus ships no
pinned expected outputs. A transpiler bug present in the source would be trained
*toward*, on both arms equally. That affects absolute pass rates, not the
stock-vs-spectrum contrast, and it is a real caveat on research question 2.

**Second limitation, observed in 06 and likely to recur.** Several source
functions in 06's corpus had code contradicting their own docstring
(pre-existing upstream, not introduced by the pipeline). Fill agents resolved
these inconsistently — some wrote instructions matching the documented *intent*,
others the actual *code* — real label noise in both the training set and the
holdout. `[TODO: check whether the successor corpus fixed this; if not, give the
fill agents an explicit tie-break rule up front rather than discovering the
inconsistency afterwards.]`

## 7. Acceptance and decision criteria (pre-registered)

Pre-registered means: fixed now, before any number exists. Do not adjust the bar
after seeing results.

### 7.1 Statistical method

Unpaired baseline, for reporting marginal rates:

```
z = (p1 − p2) / sqrt( p_pool · (1 − p_pool) · (1/n1 + 1/n2) )
p_pool = (x1 + x2) / (n1 + n2)
```

Wherever both sides answered **identical items** — every comparison in §4.3's
first two families — use **paired McNemar** on the discordant-pair counts and
report the paired result as decision-relevant. Pairing roughly halves the
detectable-effect threshold versus comparing marginal rates. Report both when
they disagree, and say which one the decision used.

Significance bar: **p < 0.05**, two-sided.

### 7.2 Decision rules

**Research question 2 — is the new dataset better than 06's?**

Direct paired McNemar of each holdout-(a) cell against its 06 counterpart, on
the identical 855 items:

| This phase | vs | 06-nitin-ds-sft |
|---|---|---|
| A1 SFT stock | vs | 619/855 (72.4%) |
| A2 SFT spectrum | vs | 636/855 (74.4%) |
| A3 DPO-best stock | vs | 619/855 (72.4%) |
| A4 DPO-best spectrum | vs | 631/855 (73.8%) |
| A5 DPO-final stock | vs | 583/855 (68.2%) |
| A6 DPO-final spectrum | vs | 636/855 (74.4%) |

The new dataset is **better** iff a **majority (≥ 4 of 6)** are significant wins
at p < 0.05 with no significant losses; **worse** iff the mirror holds;
**indistinguishable** otherwise. "Indistinguishable" is the expected base rate
and is a perfectly good result — 06 landed there against jacgen2 and said so.

Report the same six cells against the `04-cpt-sft` fresh-arm numbers too, as the
three-way lineage view. It costs nothing and it is the only way to see whether
successive corpora are actually accumulating.

Two mandatory caveats on any positive answer:

- Holdout (a) is 322/855 `conversion` rows — 38% of the graded set is (if §2.1
  holds) literally this corpus's own task shape. A win concentrated entirely in
  the `conversion` slice is a **specialization** result, not a general one.
  Report the per-slice breakdown alongside the headline, using the harness's own
  category/gate_class rows.
- Training-set sizes differ (04-cpt-sft fresh: 8,100 SFT / 654 DPO; 06: 5,474
  SFT; this phase: TBD). If this phase's pool is materially different, that is a
  **named confound stated in the headline**, not a footnote. 06 hit exactly this
  and flagged it as unresolved.

**Research question 1 — does Spectrum replicate?**

Six arm-vs-arm comparisons (3 stages × 2 holdouts). Spectrum **replicates** iff
a **majority (≥ 4 of 6)** are individually significant wins for spectrum at
p < 0.05, with sign consistency — a significant *loss* in any cell counts
against and is reported loudly rather than averaged away.

- 4+ significant wins, no significant losses → replicated on a third dataset.
  That would materially rehabilitate the 04-cpt-sft finding that 06 failed to
  reproduce.
- 1–3 significant wins → **partial / not replicated.** Combined with 06's
  result, the honest reading becomes that the fresh-arm effect was largely
  dataset-specific.
- 0 significant wins → not replicated, twice running. At that point the
  04-cpt-sft result should be treated as a single-dataset finding, and the
  project should stop paying the Spectrum arm's cost by default.
- Any significant loss → the asymmetry gets its own section, with any hypothesis
  clearly labeled speculation rather than a mechanism this design tests.

A single-metric win is never sufficient. Note also that 06's holdout B saturated
at 96–99%, which compressed every arm-vs-arm delta there; if (b) saturates again,
say so and weight the (a) column accordingly rather than reading a null on a
ceilinged holdout as evidence of no effect.

### 7.3 Power

At n = 855 and a base rate near 70%, the 04-cpt-sft comparisons detected +4.9pp
at p = 0.023 unpaired — so ~4–5pp is roughly the unpaired floor at this n, and
paired testing detects meaningfully less. Holdout (b)'s floor cannot be stated
until triage reports its real size; compute and **publish it** rather than
letting a null on a small or ceilinged holdout read as evidence of no effect.

### 7.4 Run-level acceptance gates

Before any cell is reported, each run must clear:

- `--verify-layers` preflight passes: trainable params **exactly 281.838M** on
  the spectrum arm.
- `adapter_config.json` rewritten before scoring, key assertion passes:
  **256 keys** (16 per block × 16 blocks). Without it, `load_adapters` rebuilds
  LoRA on blocks 32–47 from the adapter's own `num_layers: 16` and
  `load_weights(strict=False)` silently drops the out-of-slice picks — the same
  failure class as the fuse bug, and it fails *quietly*.
- No all-zero adapter weights. Check before eval, not after a confusing result.
- Training completed the full `iters: 8200`, or the shortfall is reported with
  its cause.
- **Best-checkpoint provenance verified across any resume.** 06's DPO watchdog
  resumed correctly after a mid-run crash but its "best snapshot" comparison did
  not persist, so it named a worse checkpoint as best. Diff `runs_pct` across
  every segment of `train.log` before trusting `.best_step`.
- Preflight lock respected: `pgrep -f "jac start"; pgrep -f mlx_lm` returns
  nothing before launch.
- The base model scored once per holdout (`JAC_EVAL_ADAPTER=""`) as the floor.

## 8. Live monitoring

Loss curves and periodic eval-subset pass-rate curves are captured **as runs
progress**. The runners write `train.log` and periodic checkpoints
(`save_every: 820`, `steps_per_eval: 500`), and `run_sft_spectrum.sh`'s watchdog
invokes `plot_metrics.jac` every `EVAL_EVERY` seconds.
`scripts/plot_progress.jac` regenerates PNGs on demand into

```
model-experiments/07-nitin-ds-new-sft/<arm>_probe/results/<stage>/plots/
```

It is one-shot by design: the orchestrating session calls it on a loop and
publishes the results as an Artifact dashboard. Subagents only keep the PNGs
current and in place.

## 9. Directory layout

See `../CONTEXT_BRIEF.md` §9 for the full tree. `dataset/` follows this
project's convention of being gitignored (large generated artifacts, regenerable
from the scripts, not source) — except the triage and comparison reports, which
are source.

## 10. Out of scope

- Re-running the SNR scan (§3 — the answer is on disk and unchanged).
- Any CPT stage. `03-cpt-only/docs/cpt-2/analysis.md` recommends against it, and
  `04-cpt-sft` already measured Spectrum-on-CPT.
- GRPO / RL. That is `05-cpt-sft-grpo/`'s scope.
- Synthesizing graph-native / OSP tasks the corpus does not contain. If §2.1
  finds such material natively, that is in scope; manufacturing it is not.
- Multi-seed repeats per arm. `04-cpt-sft`'s protocol calls for 2–3 seed repeats
  to estimate training-noise σ; this phase runs one seed per arm (four training
  runs, sequential, one 48GB box). Consequence, stated rather than omitted:
  deltas are tested against sampling noise only, not against training-seed
  noise. A borderline single-cell result should be read more cautiously than the
  same p-value in a seed-replicated design.
