# 06-nitin-ds-sft — Design Spec

Status: approved, pre-implementation.
Date: 2026-08-10.
Companion to [`../CONTEXT_BRIEF.md`](../CONTEXT_BRIEF.md) (settled facts) and
[`workflow.md`](workflow.md) (the runbook). This file is the umbrella: what
gets built, from what, measured how, and what result counts as a win.

## 1. Purpose

One training battery, two independent research questions:

1. **Does Spectrum replicate?** `04-cpt-sft`'s fresh arm found SNR-based
   layer selection beat `mlx_lm`'s default trailing-16 slice at every stage,
   significantly, on the jacgen2 dataset — and found nothing on the CPT-v2
   arm. That is a single-dataset result. Running the identical probe on an
   independently-sourced dataset, against the *same* base checkpoint, tests
   whether the effect belongs to the method or to that dataset.
2. **Is this dataset better?** Both arms are also scored on `04-cpt-sft`'s
   **unchanged** 855-row code-graded holdout, which makes each cell directly
   comparable to the existing fresh-arm number for the same arm and stage.

Only two things differ from `04-cpt-sft`'s fresh arm: the **dataset**, and
(for one arm) the **LoRA target-block selection**. Base checkpoint,
hyperparameters, seed, eval harness, and holdout are held fixed. That is what
makes this a replication rather than a confounded new experiment.

## 2. Source data

| Field | Value |
|---|---|
| Origin | `https://github.com/chess10kp/jac-data-gen` |
| Commit pin | **`7c25aff3110f526eec59e0123ffe6c0c152cce91`** |
| Local clone | `~/repos/jac-data-gen` (internal disk — see the I/O note below) |
| Shape on disk | `data/jac_outputs/*.jac`, **7,627 files**, verified by count |
| Per-file content | one docstring + one `def name(params) -> type { ... }` |
| Metadata | none — no split, no labels, no provenance, single commit |

Cite the commit pin everywhere: in `origin` on every emitted example
(`jac-data-gen:7c25aff3110f526eec59e0123ffe6c0c152cce91:<path>`), in the
triage report, and in the final comparison report. There is no other version
identifier on this corpus.

**I/O note (`CONTEXT_BRIEF.md` §1).** The project's own
`/Volumes/ExtremePro/...` external drive measured ~4.3 MB/s this session — a
genuine hardware/contention bottleneck confirmed via raw `dd`, not an app bug.
Bulk operations over the corpus (compile-checking 7,627 files, shingling,
dedup) run against the internal clone. Only final JSONL releases, plots, and
reports are written to the external project tree.

### 2.1 What this corpus actually is

Verified by sampling, and this is the framing for every downstream document:
these are **flat Python-to-Jac function conversions** — dicts, lists, loops,
early returns, string handling. No `node`, no `edge`, no `walker`, no OSP
traversal appears in the sampled files. Representative examples named in the
brief: `expand_comma(value: str)`, `Distance_modulus_to_distance(dm, absorption)`.

Consequences, all of them already decided:

- The corpus maps onto exactly one existing task type,
  `python_to_jac_function` (category `conversion`), not a new taxonomy.
  It is the same *shape* as material `04-cpt-sft` already trains on — which
  is precisely what makes the head-to-head dataset comparison meaningful, and
  also what caps how much novelty can honestly be claimed for it.
- It cannot support the flagship idiomatic DPO axis (§3), because these
  functions have no natural graph-native rewrite to prefer.
- Do **not** describe this phase's data as "idiomatic graph-native Jac"
  anywhere. It would be an overclaim against evidence already gathered.

## 3. Decisions locked (do not re-litigate without a new brainstorming pass)

| Decision | Choice |
|---|---|
| Base checkpoint | `models/qwen-q4` ("Qwen · BASE") — the **same** checkpoint `04-cpt-sft`'s fresh arm used, confirmed in `sft_fresh_probe/spectrum/configs/sft_spectrum.yaml` (`model: "models/qwen-q4"`). Not `qwen-cpt-v1`, not any SFT/DPO-tuned checkpoint. Explicit user instruction; also the only choice that keeps the comparison to existing fresh-arm numbers clean. |
| Arms | Two: **stock** (trailing-16, `mlx_lm` default blocks 32–47) and **spectrum** (SNR picks). No CPT arm — `03-cpt-only/docs/cpt-2/analysis.md` recommends skipping further CPT, and `04-cpt-sft` already measured Spectrum-on-CPT. |
| Spectrum layer selection | **Reuse `04-cpt-sft/sft_fresh_probe/spectrum/configs/spectrum_layers.json` verbatim.** Do not re-run `snr_scan.py`/`layer_select.py`. The SNR scan is a property of the base model's *weights*, not the training data; the base checkpoint is identical, so the selection is identical. Re-running would burn hours for a byte-identical answer. |
| SFT task framing | Reverse-instruction: one NL instruction authored from each file's docstring + signature, target completion = the file's Jac code. Matches the existing `code_gen`/`python_to_jac_function` reverse-authoring pattern (`01-sft-dpo/sft_dpo/jacgen2/`). |
| DPO axis | **Correctness only** (`dpo-plan.md` §2.3's correct-vs-subtly-wrong pattern). `chosen` = the file's Jac as-is; `rejected` = an LLM-authored variant carrying one subtly-introduced logic bug. See §3.1. |
| DPO axes explicitly declined | `graph_native`, `idiomatic`, `auth_security`, `typing` — none are expressible in this corpus. Recorded as a scope decision, not an oversight. |
| Eval holdouts | Both. (a) `04-cpt-sft`'s existing 855-row code-graded holdout, reused **unchanged**; (b) a new holdout carved from this corpus, frozen before any generation. Neither is optional. |
| Training recipe | Copy-adapted from `04-cpt-sft/sft_fresh_probe/`, retargeting **only** data paths, adapter paths, and results paths. `iters: 8200`, `batch_size: 1`, `learning_rate: 2.0e-5` with cosine decay (warmup 820), `num_layers: 16`, LoRA rank 16 / scale 2.0 / dropout 0.05, `max_seq_length: 3072`, `seed: 42`, `mask_prompt: true`, `grad_checkpoint: true` — all verbatim. |
| Generation transport | Batch-handoff via dispatched Claude Code Agents (no API key in this environment). Reverse-instruction authoring → **Opus** (bulk). Buggy-variant authoring → **Fable** (precision). Matches this project's per-category convention (`04-cpt-sft/docs/spec.md` §4.1). |
| Statistical bar | Two-proportion z-test, p < 0.05, paired McNemar where both arms answer identical items. Same method as every prior comparison in this project. Pre-registered in §7. |

### 3.1 Why the correctness axis, spelled out

`dpo-plan.md`'s flagship axis is chosen = graph-native, rejected =
Python-shaped. It does not fit this content: a pure utility function like
`expand_comma(value: str)` has nothing to graph-ify, so a "rejected" side
would have to be manufactured out of nothing, and the preference signal would
be about the generator's imagination rather than about Jac.

The correctness axis is the one axis this corpus genuinely supports:

- **chosen** — the file's Jac code as-is (treated as correct/functional;
  triage's compile gate is the evidence for that assumption, and its limits
  are stated in §6).
- **rejected** — the same function with exactly one subtly-introduced bug:
  off-by-one, wrong comparison operator, swapped branch, mutated default,
  inverted accumulation, and so on.
- **Hard requirement: both sides must still compile** (`jac run` exit 0). The
  bug must be a *behavioral* divergence — differing output on some input —
  not a syntax break. A pair where the rejected side fails to compile teaches
  the model to prefer "parses" over "is correct", which is not the signal
  wanted, and gets rejected at collect time with the reason logged.

This is a deliberate scope narrowing to keep the pipeline honest and
tractable. It should be documented as such in the final report, not quietly
omitted.

## 4. Design — 2 arms × 2 holdouts × 3 stages

### 4.1 The arms

| Arm | LoRA target blocks | Runner lineage |
|---|---|---|
| **stock** | trailing 16 (32–47), `mlx_lm` default | `sft_fresh_probe/run_sft.sh` + `run_dpo_nofuse.sh` |
| **spectrum** | `[0, 22, 23, 27, 30, 34, 36, 37, 38, 39, 41, 42, 43, 44, 45, 47]` (11/16 overlap with trailing-16) | `sft_fresh_probe/run_sft_spectrum.sh` + `run_dpo_spectrum.sh` |

Both arms are the same 16-block *capacity* — 281.838M trainable parameters,
enforced by `spectrum_lora_layers.py --verify-layers` as a hard preflight
gate. A count that differs means the selection changed capacity rather than
placement, which would invalidate the comparison outright. Keep the gate.

### 4.2 The holdouts

**(a) Existing "mine" holdout — reused unchanged, never regenerated.**

- `04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl` — 1,428 rows,
  **855 code-graded** (the `behavioral` + `compile_only` gate classes; the
  remaining 573 are `prose_lexical` `explanation`/`documentation` rows the
  functional harness does not score). Verified composition:

  | slice | n |
  |---|---|
  | conversion / behavioral | 322 |
  | code_gen / compile_only | 227 |
  | trajectory / compile_only | 113 |
  | code_gen / behavioral | 86 |
  | trajectory / behavioral | 70 |
  | debug / behavioral | 33 |
  | debug / compile_only | 3 |
  | migration / behavioral | 1 |
  | **code-graded total** | **855** |

- `04-cpt-sft/sft_fresh_probe/dataset/dpo/valid.jsonl` — 115 rows,
  `prompt`/`chosen`/`rejected`. **Correction to `CONTEXT_BRIEF.md` §4(a):**
  this file is *not* the DPO-stage eval set. Grepping the fresh probe's
  scripts shows nothing reads it directly — it is `mlx_lm_lora`'s validation
  split, consumed implicitly by `--data .../dataset/dpo`. Every headline DPO
  number in the comparison report was produced by `eval_dpo_nofuse.sh` /
  `eval_dpo_spectrum.sh` scoring the **SFT** holdout (855). This phase keeps
  that convention: `dataset/dpo/valid.jsonl` is a training-time validation
  split, and all three stages are scored on the same 855 rows.

- **The comparison baseline this makes available** (from
  `04-cpt-sft/docs/reports/2026-08-spectrum-vs-stock-comparison.md` §1, all
  n = 855):

  | Stage | Stock | Spectrum | Δ | p |
  |---|---|---|---|---|
  | SFT | 69.8% (597/855) | 74.7% (639/855) | +4.9pp | 0.023 |
  | DPO-best | 69.8% (597/855, step 20) | 74.2% (634/855) | +4.3pp | 0.046 |
  | DPO-final | 62.1% (531/855, step 250) | 72.7% (622/855) | +10.6pp | <0.0001 |

**(b) New holdout carved from this corpus** — `dataset/nitin_holdout.jsonl`.

- Frozen **before** any SFT-pair generation happens, and held out of the
  training pool entirely — enforced by id-exclusion at manifest build, not
  assumed.
- Size: whatever the clean pool comfortably supports after triage. Aim for
  the same order of magnitude as (a) (several hundred code-graded rows).
  **Do not force a target count.** If the clean pool doesn't support it,
  report the true number and recompute the detectable-effect floor (§7.3).
- Tests in-distribution performance on this corpus's own distribution.
  Comparable **stock vs spectrum within itself**; *not* comparable to (a)'s
  prior numbers, and any report that lines them up side by side must say so.

### 4.3 The matrix

3 stages (SFT, DPO-best, DPO-final) × 2 arms × 2 holdouts = **12 result
cells**, plus the 6 existing `04-cpt-sft` fresh-arm cells as the cross-phase
reference.

|  | holdout (a) — 855 shared | holdout (b) — Nitin, n = TBD |
|---|---|---|
| SFT · stock | cell A1 | cell B1 |
| SFT · spectrum | cell A2 | cell B2 |
| DPO-best · stock | cell A3 | cell B3 |
| DPO-best · spectrum | cell A4 | cell B4 |
| DPO-final · stock | cell A5 | cell B5 |
| DPO-final · spectrum | cell A6 | cell B6 |

Three families of comparison come out of this:

- **Within-column, arm-vs-arm** (A2−A1, A4−A3, A6−A5; B2−B1, B4−B3, B6−B5) —
  research question 1, does Spectrum replicate. Paired McNemar: both arms
  answer identical items.
- **Cross-phase, same cell** (A1 vs 597/855, A2 vs 639/855, …) — research
  question 2, is this dataset better. Also paired McNemar: same 855 items,
  same harness, different training data.
- **Column (a) vs column (b)**, reported as context only. Different item
  sets, different difficulty; a delta here is descriptive, never a test.

## 5. Reuse vs new, relative to 04-cpt-sft

Everything expensive already exists. The reuse inventory is deliberately
explicit so nobody rebuilds a battle-tested script "cleanly".

### 5.1 Reused verbatim (copy, retarget paths, change nothing else)

| Artifact | Path under `04-cpt-sft/sft_fresh_probe/` | Retarget |
|---|---|---|
| SFT runner, stock | `run_sft.sh` | data/adapter/results paths |
| SFT runner, spectrum | `run_sft_spectrum.sh` | data/adapter/results paths |
| DPO runner, stock | `run_dpo_nofuse.sh` — **not** `run_dpo.sh`, see §5.3 | data/adapter/results paths |
| DPO runner, spectrum | `run_dpo_spectrum.sh` | data/adapter/results paths |
| Eval, SFT spectrum | `eval_sft_spectrum.sh` | adapter/results/holdout paths |
| Eval, DPO stock | `eval_dpo_nofuse.sh` | adapter/results/holdout paths |
| Eval, DPO spectrum | `eval_dpo_spectrum.sh` | adapter/results/holdout paths |
| Functional harness | `sft_cptv2_probe/jacgen/eval_functional.jac` | nothing — fully env-var driven (`JAC_EVAL_*`, `JAC_HOLDOUT`) |
| Spectrum driver | `spectrum/spectrum_lora_layers.py` | nothing |
| DPO spectrum driver | `spectrum/dpo_spectrum_train.py` | nothing |
| Adapter-config rewriter | `spectrum/adapter_config_fix.py` | nothing |
| Layer selection | `spectrum/configs/spectrum_layers.json` | **nothing — reuse byte-identical** |
| SFT hyperparameters | `configs/sft.yaml`, `spectrum/configs/sft_spectrum.yaml` | `data`, `adapter_path` only |
| DPO hyperparameters | `configs/dpo_lora.yaml` + the env-var defaults in the DPO runners | nothing |
| Decontamination | `01-sft-dpo/sft_dpo/jacgen2/decontam_v2.jac` (14-token shingles, ≥0.5 overlap) | port the shingle machinery into a standalone script if the old pipeline's layout gets in the way; do not reinvent shingling |
| Loss-curve plotting | `01-sft-dpo/sft_dpo/jacgen/plot_metrics.jac` | `JAC_TRAIN_LOG` / `JAC_PLOT_DIR` env vars |

### 5.2 New in this phase (the only things actually written)

- Corpus triage + dedup + decontam + holdout carve (concurrent agent).
- Reverse-instruction SFT-pair authoring: batch handoff + collect/gate.
- Buggy-variant DPO-pair authoring: batch handoff + dual compile gate + collect.
- A lightweight watcher that regenerates loss / pass-rate PNGs while runs are
  live (§8).
- The comparison report.

### 5.3 Corrections to `CONTEXT_BRIEF.md` §5, found by checking the tree

Three things in the brief's reuse list do not match what is on disk. All
three are cheap to get wrong and expensive to discover mid-run.

1. **`spectrum/configs/dpo_spectrum.yaml` does not exist.** The fresh probe's
   `spectrum/configs/` contains exactly `sft_spectrum.yaml` and
   `spectrum_layers.json`. DPO-spectrum hyperparameters are not in a YAML at
   all — they are env-var defaults inside `run_dpo_spectrum.sh`
   (`DPO_ITERS=250`, `DPO_LR=1e-6`, `DPO_BETA=0.1`, `DPO_MAXLEN=512`) plus
   the shared `configs/dpo_lora.yaml` (rank 16 / scale 2.0 / dropout 0.05,
   `fuse: false`). Copy the runner and you carry the hyperparameters with it.
   (The brief hedged this correctly with "if it exists, check" — this is the
   answer.)
2. **`sft_fresh_probe/jacgen/eval_functional.jac` does not exist.** That
   directory holds only `make_dashboard.jac`. There is exactly **one** copy of
   the functional harness in the repo,
   `04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac`, and the fresh
   arm's own eval scripts already point at it. So the brief's "diff them
   before picking" is moot — there is nothing to diff, use the cptv2 copy.
3. **The stock DPO runner is `run_dpo_nofuse.sh`, not `run_dpo.sh`.** The
   fresh-arm stock DPO numbers quoted in §4.2 (597/855 best, 531/855 final)
   come from `results/dpo-nofuse/`. `run_dpo.sh`'s own output,
   `results/dpo/final_best.txt`, reads **12% (107/855)** — that is the
   collapsed `mlx_lm.fuse` path documented in
   `run_dpo_nofuse.sh`'s header and in comparison report §3: fusing an SFT
   LoRA delta into int4-quantized weights re-quantizes afterward and silently
   discards the delta. Copying `run_dpo.sh` would faithfully reproduce a known
   broken recipe. Use `run_dpo_nofuse.sh` + `eval_dpo_nofuse.sh`.

One thing that is *not* needed here, worth stating so nobody goes looking:
`merge_frozen_keys.py` exists only under `sft_cptv2_probe/spectrum/`. It
merges frozen CPT-v2 keys into a trained adapter — there are no frozen prior
keys on a fresh base, so neither arm here uses it, and the `mx.load`
in-place-truncation bug from comparison report §3.1 cannot occur on this
phase's code path.

## 6. Decontamination — required, reuse the existing machinery

Reuse `jacgen2/decontam_v2.jac`'s algorithm: 14-token shingles over
comment-stripped, whitespace-tokenized code; a candidate is contaminated when
≥ 0.5 of its shingles appear in the reference set. Port it to a standalone
script if coupling to the old pipeline's file layout gets in the way — do not
rebuild shingling from scratch.

Three passes, all mandatory:

1. **Within-corpus dedup.** Large Python-mined corpora cluster hard around
   common utility patterns. Exact-hash first, then near-dup by shingle
   overlap.
2. **Training pool vs existing holdouts.** Reference set =
   `04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl` +
   `.../dataset/dpo/valid.jsonl` +
   `01-sft-dpo/dataset/eval_holdout/{conversion,graph_conversion}.jsonl`.
   Any match above threshold is dropped from training.
3. **New holdout vs everything.** `nitin_holdout.jsonl` must be
   decontaminated against the training pool (otherwise it is labeled a
   holdout rather than being one) **and** against the (a) holdouts — a
   Nitin-holdout item that is a near-dup of an (a)-holdout item makes the two
   holdout columns statistically non-independent, which would quietly
   invalidate treating them as two separate tests.

**No silent caps.** Every drop is counted and attributed by reason in
`docs/reports/corpus-triage-report.md`: compile-gate failures, exact dups,
near-dups, contamination hits per reference set. A pool that shrinks a lot is
a finding to report, not a number to quietly round up.

**Stated limitation.** The `chosen` side of every DPO pair is assumed correct
on the strength of a compile gate (`jac run` exit 0) plus the corpus author's
transpiler. Nothing here proves behavioral equivalence to the original Python
— there are no pinned expected outputs in this corpus, unlike
`04-cpt-sft`'s seed pool. A transpiler bug present in the source corpus would
be trained *toward*, on both arms equally. This affects absolute pass rates,
not the stock-vs-spectrum contrast, and it is a real caveat on research
question 2.

## 7. Acceptance and decision criteria (pre-registered)

Pre-registered means: fixed now, before any number exists. Do not adjust the
bar after seeing results.

### 7.1 Statistical method

Unpaired baseline, used for reporting marginal rates:

```
z = (p1 − p2) / sqrt( p_pool · (1 − p_pool) · (1/n1 + 1/n2) )
p_pool = (x1 + x2) / (n1 + n2)
```

Wherever both sides answered **identical items** — which is every comparison
in §4.3's first two families — use **paired McNemar** on the discordant-pair
counts instead, and report the paired result as the decision-relevant one.
Pairing roughly halves the detectable-effect threshold versus comparing
marginal rates (`04-cpt-sft/docs/spectrum-workflow.md`'s calibration note).
Report both when they disagree, and say which one the decision used.

Significance bar: **p < 0.05**, two-sided.

### 7.2 Decision rules

**Research question 1 — does Spectrum replicate?**

Six arm-vs-arm comparisons exist (3 stages × 2 holdouts). Spectrum
**replicates** iff a **majority (≥ 4 of 6)** of them are individually
significant wins for spectrum at p < 0.05, with sign consistency — a
significant *loss* in any cell counts against, and is reported loudly rather
than averaged away.

- 4+ significant wins, no significant losses → replicated. Spectrum is a real
  lever independent of dataset.
- 1–3 significant wins → **partial / not replicated**. Report as such. The
  most likely honest reading is that `04-cpt-sft`'s fresh-arm result was
  partly dataset-specific.
- 0 significant wins → not replicated. This is a publishable negative and
  materially changes how much the existing fresh-arm result should be
  trusted.
- Any significant loss → the asymmetry gets its own section, in the style of
  `2026-08-spectrum-vs-stock-comparison.md` §4, with a hypothesis clearly
  labeled as speculation rather than a mechanism this design tests.

A single-metric win is never sufficient. This mirrors CPT-v1's own lesson
(the MCQ result was a single-metric false negative) — guard here against the
mirror-image single-metric false positive.

**Research question 2 — is this dataset better?**

Direct paired McNemar of each holdout-(a) cell against its `04-cpt-sft`
fresh-arm counterpart, on the identical 855 items:

| This phase | vs | 04-cpt-sft fresh arm |
|---|---|---|
| A1 SFT stock | vs | 597/855 (69.8%) |
| A2 SFT spectrum | vs | 639/855 (74.7%) |
| A3 DPO-best stock | vs | 597/855 (69.8%) |
| A4 DPO-best spectrum | vs | 634/855 (74.2%) |
| A5 DPO-final stock | vs | 531/855 (62.1%) |
| A6 DPO-final spectrum | vs | 622/855 (72.7%) |

The dataset is **better** iff a **majority (≥ 4 of 6)** of these are
significant wins at p < 0.05 with no significant losses; **worse** iff the
mirror holds; **indistinguishable** otherwise. "Indistinguishable" is the
expected base rate and is a perfectly good result — a 7,627-file
single-task-type corpus matching a 9,608-example seven-category one on that
one's own holdout would itself be informative.

Two mandatory caveats on any positive answer here:

- Holdout (a) is 322/855 `conversion` rows — 38% of the graded set is
  literally this corpus's own task shape. A win concentrated entirely in the
  `conversion` slice is a *specialization* result, not a general one. **Report
  the per-slice breakdown alongside the headline**, using the harness's own
  category/gate_class rows.
- The training-set sizes differ (`04-cpt-sft` fresh: 8,100 SFT / 654 DPO;
  this phase: TBD). If this phase's pool is materially smaller or larger,
  that is a named confound on question 2 and must be stated in the headline,
  not buried.

### 7.3 Power

At n = 855 and a base rate near 70%, the existing fresh-arm comparisons
detected +4.9pp at p = 0.023 unpaired — so ~4–5pp is roughly the unpaired
floor at this n, and paired testing detects meaningfully less. Holdout (b)'s
floor cannot be stated until triage reports its real size; compute and
**publish it** in the comparison report rather than letting a null result on
a small holdout read as evidence of no effect. If holdout (b) lands below
~300 code-graded rows, say plainly that it is underpowered for effects of the
size seen on holdout (a), and let holdout (a) carry the decision.

### 7.4 Run-level acceptance gates

Before any cell is reported, each run must clear:

- `--verify-layers` preflight passes: trainable params **exactly 281.838M**
  on the spectrum arm.
- `adapter_config.json` rewritten before scoring, and the key assertion
  passes (**256 keys** = 16 per block × 16 blocks). Without this,
  `load_adapters` rebuilds LoRA on blocks 32–47 from the adapter's own
  `num_layers: 16` and `load_weights(strict=False)` silently drops the
  out-of-slice picks — the same failure class as the fuse bug, and it fails
  *quietly*.
- No all-zero adapter weights. Check before eval, not after a confusing
  result (comparison report §3.1).
- Training completed the full `iters: 8200` (or the shortfall is reported
  with its cause).
- Preflight lock respected: `pgrep -f "jac start"; pgrep -f mlx_lm` returns
  nothing before launch. A resident JMS server or stray mlx process OOMs this
  48GB box regardless of anything the script does. This repo's `jms/` server
  was mid-testing recently and may still have an instance up.

## 8. Live monitoring (explicit user requirement)

Loss curves and periodic eval-subset pass-rate curves are captured **as runs
progress**, not reconstructed afterward. The runners already write
`train.log` and periodic checkpoints (`save_every: 820`,
`steps_per_eval: 500`), and `run_sft_spectrum.sh`'s watchdog already invokes
`plot_metrics.jac` every `EVAL_EVERY` seconds.

Requirement for this phase: a lightweight watcher tails those logs and
regenerates PNGs (matplotlib, already in the project venv) whenever new data
lands, at the predictable path

```
model-experiments/06-nitin-ds-sft/<arm>_probe/results/<stage>/plots/
```

The orchestrating session — not a subagent — periodically publishes these as
an Artifact dashboard. Subagents only keep the PNGs current and in place.

## 9. Directory layout

```
model-experiments/06-nitin-ds-sft/
  CONTEXT_BRIEF.md
  docs/
    README.md  spec.md  workflow.md  dataset-structure.md
    reports/corpus-triage-report.md
    reports/<comparison reports as they land>
  dataset/
    candidate_pool.jsonl     # clean, deduped, decontaminated, pre-holdout-split
    nitin_holdout.jsonl      # frozen holdout, §4.2(b)
    sft_train.jsonl          # final SFT release
    dpo_train.jsonl          # final DPO release
    batches/                 # pending_batch.jsonl / responses_batch.jsonl handoff
    rejected/                # every drop, with reason — no silent caps
  stock_probe/               # copy-adapted from sft_fresh_probe, stock arm
  spectrum_probe/            # copy-adapted from sft_fresh_probe, spectrum arm
  external/                  # workspace for corpus-derived intermediates
```

`dataset/` follows this project's convention of being gitignored (large
generated artifacts, regenerable from the scripts, not source) — except the
triage and comparison reports, which are source.

## 10. Out of scope

- Re-running the SNR scan (§3 — the answer is already on disk and unchanged).
- Any CPT stage. `03-cpt-only/docs/cpt-2/analysis.md` recommends against it,
  and `04-cpt-sft` already measured Spectrum-on-CPT.
- GRPO / RL. That is `05-cpt-sft-grpo/`'s scope.
- Graph-native / OSP task generation. The corpus does not contain the
  material (§2.1), and synthesizing it would make this a different experiment
  with a different confound.
- Multi-seed repeats per arm. `04-cpt-sft`'s protocol calls for 2–3 seed
  repeats to estimate training-noise σ; this phase runs one seed per arm
  (four training runs, sequential, on one 48GB box). Consequence, stated
  honestly rather than omitted: deltas here are tested against sampling noise
  only, not against training-seed noise. A borderline single-cell result
  should therefore be read more cautiously than the same p-value in a
  seed-replicated design.
