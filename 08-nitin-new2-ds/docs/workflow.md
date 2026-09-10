# workflow.md — 08-nitin-new2-ds Runbook

Execution order for the two-arm Spectrum replication on the merged 7-source
corpus at `jac-data-gen@c95b7563`. Companion to [`spec.md`](spec.md) (design,
decision rules) and [`../CONTEXT_BRIEF.md`](../CONTEXT_BRIEF.md) (settled
facts).

```
[DONE] pin corpus @ c95b7563                                    [Stage -1]
[DONE] 7-source merge + denylist + quality + license filters    [Stage 0]
[DONE] dedup + decontam + holdout(b) reuse + leak-check          [Stage 0]
[DONE] 85/15 splits (sft_train/dpo_train -> sft/,dpo/)           [Stage 0b]
[NEXT] SFT training x2 arms (sequential only)                    [Stage 3]
  -> DPO training x2 arms (each on its own SFT adapter)          [Stage 4]
  -> eval: 2 arms x 2 holdouts x 3 stages                        [Stage 5]
  -> comparison report                                           [Stage 6]
```

Stages 1-2 (SFT/DPO-pair *authoring*) from 06/07's runbook don't apply here —
08's source material already comes as complete pairs from `jac-data-gen`
(instructions/prompts pre-authored upstream, or synthesized deterministically
from structured fields like `farm`'s `manifest`). There is no reverse-
instruction-authoring pass in this phase; Stage 0 is the merge, filter, and
release-split pipeline instead.

## Conventions used throughout

- **Run everything from the repo root**
  (`/Volumes/ExtremePro/JaseciLabs/jac_model_studio`).
- **Everything is Jac.** `.sh` wrappers call `jac run <driver>.jac <flags>`.
  Use the absolute venv binary, not bare `jac`:
  `/Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac`.
- **`with entry:__main__ { }`**, never a bare `with entry { }`.
- **No silent caps.** Every dropped item is counted, attributed by reason,
  written under `dataset/rejected/`, and summarized in the triage report.
- **Verify after any drive interruption.** This phase's own Stage 0 build was
  interrupted mid-run by an external-drive disconnect (`CONTEXT_BRIEF.md`
  §11); the partial output was re-verified (funnel arithmetic, duplicate-id
  check, schema spot-check) before being trusted rather than re-run blind.
  Do the same for any future interrupted stage.

## Stage 0 — Corpus merge, filter, dedup, decontam, holdout reuse (DONE)

```
.venv/bin/jac run model-experiments/08-nitin-new2-ds/scripts/pipeline.jac
.venv/bin/jac run model-experiments/08-nitin-new2-ds/scripts/copy_holdout.jac
```

`pipeline.jac` reads all 7 sources at the pin, applies (in order): the
eval-denylist drop, the composer quality-gradient test-count filter (≥18),
the js2jac license filter, within-corpus exact + near-dup dedup, decontam vs
the reference holdouts, and a global cross-source dedup pass. Output:
`dataset/candidate_pool.jsonl` (16,167 rows) and
`dataset/rejected/{sft,dpo}/rejected.jsonl` (9,404 / 224, every row tagged
with its drop reason). Full funnel: `docs/reports/corpus-triage-report.md`.

`copy_holdout.jac` copies `07-nitin-ds-new-sft/dataset/{nitin_holdout,
nitin_holdout_eval}.jsonl` byte-identically (asserted), then re-runs the
four-surface leak check (id / `jac` hash / `jac_rejected` hash / holdout-
prompt hash) against `candidate_pool.jsonl` on disk — not just against the
in-memory pool from the same run — and rewrites the pool (keeping a
`.preleak` backup) if anything collides. Result this run: **0 collisions**,
pool left untouched.

**Gate to leave Stage 0:** both holdout files present and byte-identical to
07's; leak-check reports 0 failures; funnel counts in
`/tmp/nitin_new2_triage/stats.json` sum correctly stage-to-stage.

## Stage 0b — Build the release + split files (DONE)

```
.venv/bin/jac run model-experiments/08-nitin-new2-ds/scripts/release.jac
model-experiments/08-nitin-new2-ds/scripts/prep_training_dirs.sh
```

`release.jac` is 08's `collect.jac` equivalent — reconciles 7 heterogeneous
source schemas into one trainer-facing wire format: SFT assistant turns
` ```jac `-fenced (matching 07's 100%-fenced convention; eval scripts strip
fences), DPO `chosen`/`rejected` unfenced raw code (matching 07's convention;
`dpo_fixed_train.jac` builds the chat turns itself). Splits on `task_kind`:
`pair_dpo` → DPO release, `completion`/`translation` → SFT release. Writes
`dataset/sft_train.jsonl` (14,792) and `dataset/dpo_train.jsonl` (1,375), plus
07-compatible `task_type`/`register`/`tier`/`axis` slicing columns. Hard
asserts on the trainer contract (required fields present, non-empty) so
`prep_training_dirs.sh` can't silently drop rows into a short split.

`prep_training_dirs.sh` — unchanged from 07: 85/15, seed 42, deterministic
sort-then-shuffle. Produces `dataset/sft/{train,valid}.jsonl` (12,573 /
2,219) and `dataset/dpo/{train,valid}.jsonl` (1,169 / 206).

**Gate cleared:** train∩valid = ∅ for both tracks, train∪valid == release for
both, 0 rows dropped by the splitter's required-field filter, leak-check
re-run post-split against both holdouts — still 0.

## Stage 3 — SFT training ×2 arms

### 3.0 The probe directories (as built)

Identical structure and layout to 07's — see `../CONTEXT_BRIEF.md` §9 for the
tree and §6 for the three inherited open items (holdout default in DPO
runners, the `total==855` collapse-gate filter, the stale "06 incumbent" echo
lines) that were not re-decided during scaffolding.

**Verified during scaffolding:**
- `jac check` clean on all 5 `.jac` files (`dpo_fixed_train.jac` ×2,
  `spectrum_lora_layers.jac`, `dpo_spectrum_train.jac`,
  `adapter_config_fix.jac`).
- `bash -n` clean on all 9 shell scripts.
- Grep for leftover `07-nitin-ds-new-sft` / `07_vs_06` references across
  every copied file: zero hits.
- `dpo_fixed_train.jac` byte-identical (md5) across both probes and matching
  07's.
- `spectrum_layers.json` unchanged, sha256 chain intact back to 04-cpt-sft.

**ONE shared `dataset/`, not a copy per arm** — unchanged rationale from
06/07: both arms train on identical data by design.

### 3.1 Preflight (every training launch, both arms)

```
pgrep -f "jac start"; pgrep -f mlx_lm
```

Both must return nothing before starting.

### 3.2 Stock arm SFT

```
CONFIRM_FULL_RUN=1 model-experiments/08-nitin-new2-ds/stock_probe/run_sft.sh
```

Outputs: `stock_probe/adapters/sft-on-nitin/`,
`stock_probe/results/sft/{train.log,metrics_functional.jsonl}`.

### 3.3 Spectrum arm SFT

```
CONFIRM_FULL_RUN=1 model-experiments/08-nitin-new2-ds/spectrum_probe/run_sft_spectrum.sh
```

`--verify-layers` must print `VERIFY: PASS` at 281.838M trainable params
before training proceeds. **Sequential only** — stock SFT must finish before
spectrum SFT starts.

### 3.4 Live monitoring (runs alongside 3.2/3.3)

```
.venv/bin/jac run model-experiments/08-nitin-new2-ds/scripts/plot_progress.jac \
    --train-log <arm>_probe/results/<stage>/train.log \
    --out       <arm>_probe/results/<stage>/plots/loss.png \
    --eval-curve <arm>_probe/results/<stage>/metrics_functional.jsonl \
    --eval-out   <arm>_probe/results/<stage>/plots/eval.png
```

Note: `plot_progress.jac` was not part of this phase's mechanical scaffold
pass (it lives in `scripts/`, which the dataset-pipeline agent populated with
the merge pipeline only). Copy it from
`07-nitin-ds-new-sft/scripts/plot_progress.jac` (path-substitute only, no
logic change needed — it's train-log/metrics-driven, not corpus-aware)
before Stage 3 starts monitoring.

## Stage 4 — DPO training ×2 arms

```
CONFIRM_FULL_RUN=1 model-experiments/08-nitin-new2-ds/stock_probe/run_dpo_nofuse.sh
CONFIRM_FULL_RUN=1 model-experiments/08-nitin-new2-ds/spectrum_probe/run_dpo_spectrum.sh
```

Order: stock SFT → spectrum SFT → stock DPO → spectrum DPO, sequential
throughout. Each arm's DPO seeds from its own SFT adapter via
`--resume-adapter-file`. `run_dpo_spectrum.sh`'s three gates (`--verify-
patches`, `--verify-layers`, per-snapshot `adapter_config.json` rewrite)
unchanged from 07.

**Known hazards, unchanged from 06/07:** DPO OOM at `DPO_MAXLEN=512` (start
there; the shrink ladder handles an OOM; report if it lands at 384); best-
checkpoint tracking not persisting across a crash-resume (diff `runs_pct`
across every log segment before trusting `.best_step`).

**Note on the idiomaticity DPO axis (`spec.md` §3.1):** `run_dpo_nofuse.sh`
and `run_dpo_spectrum.sh` are unmodified from 07 — they train on whatever
`dataset/dpo/{train,valid}.jsonl` contains, and 08's `chosen`/`rejected`
semantics differ from 07's (idiomatic-vs-floor, not correct-vs-buggy). No
script change is needed for this — the training loop is axis-agnostic — but
report interpretation must account for it (§3.1).

## Stage 5 — Eval: 2 arms × 2 holdouts × 3 stages

Same harness as every prior phase:
`model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac`.

### 5.1 Holdout (a) — the shared 855

```
model-experiments/08-nitin-new2-ds/stock_probe/eval_sft_sweep.sh
model-experiments/08-nitin-new2-ds/spectrum_probe/eval_sft_spectrum.sh
model-experiments/08-nitin-new2-ds/stock_probe/eval_dpo_nofuse.sh
model-experiments/08-nitin-new2-ds/spectrum_probe/eval_dpo_spectrum.sh
```

### 5.2 Holdout (b) — 07's holdout, reused

```
H=model-experiments/08-nitin-new2-ds/dataset/nitin_holdout_eval.jsonl
S=model-experiments/08-nitin-new2-ds/stock_probe
P=model-experiments/08-nitin-new2-ds/spectrum_probe

HOLDOUT=$H RDIR=$S/results/sft-holdoutB          $S/eval_sft_sweep.sh
HOLDOUT=$H RDIR=$P/results/sft-spectrum-holdoutB $P/eval_sft_spectrum.sh
HOLDOUT=$H RDIR=$S/results/dpo-nofuse-holdoutB   $S/eval_dpo_nofuse.sh
HOLDOUT=$H RDIR=$P/results/dpo-spectrum-holdoutB $P/eval_dpo_spectrum.sh
```

Remember: holdout (b) is 07's content, not 08's own multi-source
distribution (`spec.md` §4.2) — read the result as "does 08's training help
on 07's task shape," not "does it help on 08's own new material."

**Sanity check before trusting a holdout-B number:** if a full 855-row eval
finishes in seconds, it did not run (06 incident 4 — check row count in
`metrics_functional.jsonl`). Not expected here since holdout (b) was copied
with its `messages` field already populated (it's 07's already-authored
`nitin_holdout_eval.jsonl`), but verify rather than assume.

### 5.3 Non-negotiable per-eval checks

Unchanged from 06/07: `adapter_config_fix.jac` before any spectrum scoring
(256-key assertion); no all-zero adapter weights; base model scored once per
holdout as the floor.

### 5.4 Failure analysis (optional, after the headline numbers)

Same three-script pipeline as 07 (`gen_eval_detail.jac`,
`grade_eval_detail.jac`, `grade_reference.jac`) — copy from
`07-nitin-ds-new-sft/scripts/` (path-substitute only) before Stage 5.4;
these were also not part of this phase's dataset-build scaffold.

## Stage 6 — Comparison report

Write to `docs/reports/08-final-comparison.md`, following 07's structure:
bottom line up front → headline table per holdout → cross-dataset table (vs
07 and vs 06) → real problems hit → honest reading → live graphs → artifacts.

Contents, per `spec.md` §7:

1. The 12-cell matrix, both holdouts, all stages, both arms.
2. Cross-phase paired McNemar of each holdout-(a) cell against 07's
   corresponding cell → RQ2. Show 06's and 04-cpt-sft's columns alongside as
   the four-way lineage view.
3. Arm-vs-arm paired McNemar per column → RQ1.
4. Per-source-type breakdown of any RQ2 win — does it concentrate in
   `composer` (corroborating the quality filter) or spread uniformly
   (suggesting volume alone)?
5. RQ3's qualitative read — aggregate movement plus whatever the failure-data
   breakdown shows about osp/farm-sourced generalization, with the explicit
   caveat that no holdout is graph-native in-distribution this phase.
6. Named confounds in the headline: pool size (2.18× 07's) and content-type
   change simultaneously; single seed per arm; DPO axis change (idiomaticity,
   not correctness) making any DPO-stage comparison to 07 partially
   apples-to-oranges on *what* is being trained, even where the *numbers* are
   directly comparable.
7. Incidents — including the Stage 0 drive disconnect and its recovery
   verification, with the same forensic detail as 06's and 07's incident
   logs.

## Quick reference — failure modes already paid for

See `../CONTEXT_BRIEF.md` §11 for the full table, now including the Stage 0
drive-disconnect incident from this phase's own build.
