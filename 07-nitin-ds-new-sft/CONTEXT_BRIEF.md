# 07-nitin-ds-new-sft — master context brief

Read this FIRST, in full, before doing anything else. It exists so every agent
working this experiment shares the same facts instead of re-deriving them —
do not re-discover what's already answered here; do flag if something here
turns out to be wrong.

> **Status: DATASET GENERATION IN PROGRESS (2026-08-18).** Corpus pinned and
> triaged; holdout carved; batches prepared; instruction authoring under way.
> **No training has run yet, and no number in this file is a 07 result** —
> every result quoted below is a *prior* result (06 or 04-cpt-sft) carried
> forward as the baseline this phase will be measured against.
>
> Done: corpus pinned (§1), shape verified (§1.1), triage funnel complete
> (9,367 → 7,636 clean → 855 holdout + 6,781 training pool, 0 leaks; see
> `docs/reports/corpus-triage-report.md`), 26 batch shards prepared across the
> `sft` / `holdout_eval` / `dpo` tracks.
> Next: fill the batches, collect + gate into releases, split train/valid, then
> the four serialized training runs.

## 0. What this experiment is

Direct follow-up to `model-experiments/06-nitin-ds-sft/`, run on a **NEW
dataset from Nitin** — the successor to the corpus 06 used. Same battery, same
base checkpoint, same recipe, same eval instruments; only the training data
changes. Two research questions, both answered with the same paired
two-proportion z-test methodology used in 06 and in
`model-experiments/04-cpt-sft/docs/reports/2026-08-spectrum-vs-stock-comparison.md`:

1. **RQ1 — does Spectrum (SNR / Marchenko-Pastur layer selection) replicate on
   a THIRD dataset?** 04-cpt-sft's fresh arm found a significant Spectrum win at
   every stage; 06 did **not** replicate it (one significant win, one
   significant loss, four nulls). 06's final report closes with "worth a third
   replication before treating either direction as settled." This is that
   third replication.
2. **RQ2 — is Nitin's NEW dataset better than the one 06 used?** 06's own
   verdict on its corpus was "weak, inconsistent evidence it beats jacgen2 — 1
   of 4 cross-dataset comparisons significant, none against." This phase asks
   whether the successor corpus moves that needle, measured cell-for-cell
   against 06's numbers on the *identical* shared holdout.

## 1. Source data — the new corpus

**RESOLVED 2026-08-18.** The "new dataset" is the **same repo at a newer
commit**, not a different repo:

| Field | Value |
|---|---|
| Origin repo | `https://github.com/chess10kp/jac-data-gen` (same as 06) |
| **Commit pin** | **`11fa3f45a0a349337ae4c355708a7e4974b54a36`** — "Add idiomatic py2jac corpus (9,367 records)", 2026-08-17 |
| 06's pin, for contrast | `7c25aff3110f526eec59e0123ffe6c0c152cce91` — "data", 2026-08-10 |
| Local clone | `~/repos/jac-data-gen` (INTERNAL disk) |
| Shape on disk | **`data/py2jac_dataset_idiomatic.jsonl` — ONE JSONL, not loose files** |
| Record count | **9,367** (vs 06's 7,627 loose `.jac` files) |
| Record schema | `{id: int, entrypoint: str, source: str, jac: str}`; `source` is `"idiomatic"` for all 9,367 |
| Per-record content | one docstring + one top-level `def name(params) -> type { ... }` |

The pin, path, and `origin` prefix live in `scripts/pipeline.jac` (`CORPUS`,
`SRC`, `SRC_REL`, `COMMIT`) and `scripts/compile_check.jac` (`SRC`) — those are
the single source of truth the pipeline reads.

**Pipeline adaptation, and why it is small.** 06's loader globbed
`data/jac_outputs/*.jac`; 07's reads one JSONL. Every record is given the
**pseudo-filename `f"{id}.jac"`**, so every downstream stage — which keys off
`i["file"]`, sorts with `int(file[:-4])`, and joins against `compile_check`'s
output by that key — is byte-for-byte 06's logic over the same key space. Ids
are unique ints (9,367 unique over 9,367 rows, verified). Only the loader and
the `origin`/`source_file` strings changed; the funnel, the shingle machinery,
the thresholds, and the holdout carve are untouched. `origin` is now
`jac-data-gen:11fa3f45…:data/py2jac_dataset_idiomatic.jsonl#id=<id>`.

**I/O note, carried forward from 06 and still true.** The project's own
`/Volumes/ExtremePro/...` external drive measured ~4.3 MB/s (genuine
hardware/contention bottleneck, confirmed via raw `dd`, not an app bug). Do
bulk file operations — compile-checking thousands of files, shingling, dedup —
on an internal-disk clone. Only final JSONL / PNG / report artifacts get
written to the external project tree.

### 1.1 Corpus shape — VERIFIED, not assumed (2026-08-18)

06's corpus was flat, Python-transpiled utility functions with no
`node`/`edge`/`walker` anywhere — the same *shape* as this project's existing
`python_to_jac_function` task type, **not** graph-native OSP material. 06's docs
say loudly: do not describe that data as "idiomatic graph-native Jac."

The successor corpus was scanned in full (all 9,367 records), not sampled:

| Check | Result |
|---|---|
| Top-level `def` count per record | **1, for all 9,367 records** — no exceptions |
| Records starting with a docstring | 8,970 / 9,367 (95.8%) |
| **OSP archetype declarations** (`node X {`, `edge X {`, `walker X {`) | **0** |
| `walker` keyword anywhere | 0 |
| `with entry` blocks | 0 |
| `node` / `edge` / `root` / `here` word hits | 92 records — **all incidental**: docstring prose ("Returns if the line is a LABEL node") and identifiers in tree/graph *algorithms*, never archetypes |
| `obj` / `class` declarations | 17 records |

**Verdict: the structural shape is UNCHANGED from 06.** One docstring + one
top-level typed `def`, no OSP. So the name "idiomatic" in
`py2jac_dataset_idiomatic.jsonl` does **not** mean graph-native — it means
richer *Jac language* idiom inside the same flat-function shape:

| Idiom | Records |
|---|---|
| explicit `->` return type | 9,359 / 9,367 |
| union types (`\|`) | 2,566 |
| `match` statements | 143 |
| `isinstance` narrowing | 479 |
| `lambda` | 120 |

That is the real difference between 06's corpus and this one, and it is exactly
what RQ2 measures: **same task shape, better-written Jac**. Do not describe this
data as "graph-native" or "OSP" anywhere — it would be an overclaim against
evidence already gathered, the same overclaim 06 refused to make.

**Consequence for §2: the inherited framing decisions HOLD.** Because the shape
did not change, the single task type (`conversion` /
`python_to_jac_function`) and the correctness-axis DPO both still fit, and no
re-litigation was needed. Had the corpus turned out graph-native, §2 would have
had to change before generation; it did not.

## 2. Framing decisions — inherited from 06, RECONFIRMED for this corpus (§1.1)

These are 06's settled decisions (`06-nitin-ds-sft/CONTEXT_BRIEF.md` §2),
carried forward as the working defaults. Each was valid only if the new corpus
had the same shape as 06's — **§1.1 verified that it does (full-corpus scan, not
a sample), so all of them are now RECONFIRMED rather than merely inherited.**

**SFT — reverse-instruction authoring (reconfirmed).** One NL instruction
authored per selected file from its docstring + signature, paired with the
file's Jac code as the target completion. Matches this project's existing
`code_gen` / `python_to_jac_function` reverse-authoring pattern
(`model-experiments/01-sft-dpo/sft_dpo/jacgen2/`). Messages-format schema
mirroring `04-cpt-sft/dataset/fresh/releases/sft_train.jsonl`'s field set, with
`category="conversion"`, `task_type="python_to_jac_function"`,
`origin="<repo>:<commit>:<path>"`.
**RECONFIRMED 2026-08-18** (§1.1): the corpus is one-`def`-per-record with 0
OSP archetypes, so `python_to_jac_function` still fits and no second task type
is warranted.

**DPO — correctness axis, NOT the idiomatic-vs-Python-shaped axis
(reconfirmed).** `dpo-plan.md`'s flagship idiomatic axis (chosen = graph-native,
rejected = Python-shaped) did not fit 06's content because pure utility/math
functions have nothing to "graph-ify". Instead: **chosen** = the file's Jac
code as-is (assumed correct/functional, on the strength of the compile gate);
**rejected** = an LLM-authored variant carrying exactly one subtly-introduced
logic bug (off-by-one, flipped comparison, swapped branch, mutated default,
inverted accumulator, dropped edge-case guard). **Both sides must still
compile** (`jac run` exit 0) — the bug must be a behavioural divergence, not a
syntax break; a rejected side that fails to parse teaches "prefer code that
compiles" instead of "prefer code that is correct".
**RECONFIRMED 2026-08-18** (§1.1): 0 OSP archetypes in 9,367 records, so the
flagship idiomatic axis is still unavailable — these functions have no
graph-native rewrite to prefer. The correctness axis stands. Note the corpus IS
richer in Jac idiom (union types, `match`, `isinstance`), but idiom richness is a
property of the *chosen* side here, not a preference axis: both sides of a DPO
pair come from the same source function.

Both decisions were deliberate scope choices in 06, documented as such rather
than omitted. Keep that habit.

## 3. Base checkpoint — "fresh qwens" (unchanged, explicit standing instruction)

`models/qwen-q4` (registry label "Qwen · BASE") — the SAME base checkpoint used
by 06 both arms and by 04-cpt-sft's fresh arm. NOT `qwen-cpt-v1`, NOT any
existing SFT/DPO-tuned checkpoint. Holding the base fixed across 04 → 06 → 07
is the only thing that keeps the cross-phase dataset comparison clean: the
dataset (and, for one arm, the LoRA target-block selection) are the sole
variables.

**Spectrum layer selection: reuse verbatim, do NOT re-run the SNR scan.** The
selection `[0, 22, 23, 27, 30, 34, 36, 37, 38, 39, 41, 42, 43, 44, 45, 47]`
(11/16 overlap with trailing-16) is a property of the base model's *weights*,
not the training data. `spectrum_probe/spectrum/configs/spectrum_layers.json`
in this phase is **sha256-identical** to 06's, which is sha256-identical to
04-cpt-sft's. Re-running `snr_scan` / `layer_select` would burn hours for a
byte-identical answer.

## 4. Eval — BOTH holdouts, four result cells per stage (carried from 06 §4)

Do NOT skip either holdout. Per stage (SFT-final, DPO-best, DPO-final), for
EACH arm (stock, spectrum), eval against:

**(a) The existing shared holdout — reuse UNCHANGED, do not regenerate:**
`model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl` —
1,428 rows, **855 code-graded**. This is the instrument that makes every number
directly comparable to 06's and to 04-cpt-sft's fresh arm.

Note (06 §4.5 item 4, still true): `sft_fresh_probe/dataset/dpo/valid.jsonl`
(115 rows) is **not** a separate scored eval set — nothing reads it for
scoring; it is `mlx_lm_lora`'s internal validation split during DPO training.
All headline numbers at every stage are scored on the SAME 855 code-graded rows.

**(b) A new holdout carved from the NEW corpus** — `dataset/nitin_holdout.jsonl`,
frozen BEFORE any generation and excluded from the training pool by
id-exclusion at manifest build, not by assumption.

**Size: exactly 855 rows — a fixed constant, not a ratio.** 06 matched holdout
(b) to holdout (a)'s 855 at carve time ("Same 855-row size (matched deliberately
at holdout-carve time)", 06's final report §2) so both columns are the same size
and the cross-phase comparison stays like-for-like. **Carry 855 forward
unchanged regardless of this corpus's pool size** — do not rescale it just
because the new corpus has 9,367 records where 06 had 7,627.

`scripts/pipeline.jac` now hard-codes `HOLD_N = 855` and **raises** if the clean
pool falls below 4,000 rows, instead of 06's silent `len(pool)//5` fallback. A
silently-shrunk holdout is exactly the "no silent caps" violation this project
forbids: it would break cross-comparability with no signal. If that error ever
fires, it is a human decision — accept an underpowered holdout explicitly and
recompute the detectable-effect floor (`spec.md` §7.3), or loosen the triage
gates and re-run.

**Hard lesson from 06, do not repeat it.** 06's `nitin_holdout.jsonl` was frozen
from the candidate pool *before* the reverse-instruction authoring phase, so it
carried only raw `jac_code` + `docstring` and **no `messages` field** — which
`eval_functional.jac` needs to query the model at all. The first holdout-B
eval pass "completed" in under 30 seconds because it never actually ran. 06
fixed it after the fact by reverse-authoring instructions for all 855 holdout
rows into a separate `nitin_holdout_eval.jsonl`. **Plan for that up front here:**
either author instructions for the holdout in the same pass as the training
pool, or budget the extra pass explicitly. `scripts/gen_eval_detail.jac` already
points holdout "B" at `dataset/nitin_holdout_eval.jsonl`, not
`nitin_holdout.jsonl`.

**Comparison matrix to build:** 2 arms × 2 holdouts × 3 stages = up to 12
cells, plus cross-phase comparisons against 06's cells (§4.1) on the identical
855 items. Paired McNemar where both sides answer identical items; unpaired
two-proportion z-test for marginal rates; p < 0.05 two-sided.

### 4.1 The baselines this phase is measured against

All of the following are **prior results**, not this phase's.

**06-nitin-ds-sft, holdout A (the shared 855)** —
`06-nitin-ds-sft/docs/reports/2026-08-final-comparison.md` §1:

| Stage | Stock | Spectrum |
|---|---|---|
| Base (untrained) | 10.5% (90/855) | 10.5% (90/855) |
| SFT-final | 72.4% (619/855) | 74.4% (636/855) |
| DPO-best | 72.4% (619/855) | 73.8% (631/855) |
| DPO-final | 68.2% (583/855) | 74.4% (636/855) |

**06-nitin-ds-sft, holdout B (its own in-distribution 855)** — same report §2.
Reported for shape only; 07's holdout (b) is a *different* item set and is not
directly comparable to it:

| Stage | Stock | Spectrum |
|---|---|---|
| Base (untrained) | 0.0% (0/855) | 0.0% (0/855) |
| SFT-final | 99.3% (849/855) | 97.9% (837/855) |
| DPO-best | 98.7% (844/855) | 97.7% (835/855) |
| DPO-final | 97.0% (829/855) | 96.5% (825/855) |

**04-cpt-sft fresh arm (the original jacgen2 dataset), holdout A** — the
lineage baseline, all n = 855: SFT stock 597 / spectrum 639; DPO-best stock 597
/ spectrum 634; DPO-final stock 531 / spectrum 622.

**Training-pool sizes, for the size confound** — 04-cpt-sft fresh: 8,100 SFT
rows / 654 DPO pairs. 06: 5,474 SFT rows. **07: 6,781 SFT rows** (1.24× 06,
0.84× the fresh arm). That is a materially different size and therefore a
**named confound stated in the headline**, not a footnote — dataset size and
dataset content differ at once, so RQ2 is not a clean dataset-quality
isolation. 06 hit the same confound in the opposite direction and flagged it
as unresolved.

**06's verdicts, for context on what "moving the needle" would mean:** RQ1 —
Spectrum did not cleanly replicate (1 significant win, 1 significant loss, 4
nulls; the ≥4-of-6 bar cleared on neither holdout). RQ2 — weak, inconsistent
evidence 06's dataset beat jacgen2's (1 of 4 significant, none against; 06's
report still picks it if forced to choose, partly because it matched a 32%
larger dataset with less data).

### 4.2 Eval sweep sizing (carried from 06 §4.6)

Two distinct passes, not one:

- **Interim/sweep checkpoints** (at each `save_every` / segment boundary, the
  data behind the live curves): eval against a **15% slice** of the holdout
  (855 × 0.15 ≈ 128 rows) — seeded, frozen, carved once and reused at every
  sweep point so the curve is comparable step-to-step. Matches
  `eval_sft_sweep.sh`'s existing subset-sweep pattern.
- **Final checkpoints only** (SFT-final, DPO-best, DPO-final — the numbers that
  go in the report): full evaluation on **BOTH** holdouts at full size. Never
  substitute the sweep subset for a headline number.

## 5. Everything is Jac — explicit user instruction for this phase

**All scripts in this experiment are `.jac`, not `.py`.** This is the one
structural difference from 06, which mixed `.jac` pipeline scripts with `.py`
drivers. Jac is a Python superset that compiles to Python bytecode, so
`mlx`, `mlx_lm`, `mlx_lm_lora`, `matplotlib`, `transformers` and the rest
import and behave exactly as before — the port is syntax, not semantics.

- Every driver and pipeline script here is `<name>.jac`.
- `.sh` wrappers stay `.sh`, but every driver invocation is now
  `jac run <driver>.jac <flags>` instead of `python <driver>.py <flags>`.
  `jac run` sets `sys.argv = [filename] + flags`, so each driver's own
  argparse/argv handling is unchanged.
- `[TODO: confirm the `jac` resolved on PATH is the SAME interpreter/venv that
  has mlx / mlx_lm / mlx_lm_lora installed.]` `workflow.md`'s conventions warn
  that a bare `jac` can resolve to a different, stale venv. Every run/eval
  script carries this TODO inline.
- **Still Python, and deliberately left alone:** small inline `python3 -c`
  arithmetic and `python - <<'PY'` heredocs *inside* the `.sh` wrappers (the
  SFT-baseline lookup, the STEP_PCT / collapse-gate arithmetic, the adapter
  key assertion, the train/valid split in `prep_training_dirs.sh`). They are
  shell-embedded snippets, not script files. `[TODO: decide whether these
  heredocs should also be extracted into `.jac` helpers — doing so adds files
  06 does not have, so it was left as a call for the human.]`
- **New-file rule:** any new script written for this phase is `.jac` from the
  start. Use `with entry:__main__ { }`, never a bare `with entry { }` — plain
  `with entry` also executes on *import*, a failure mode this repo has already
  paid for (task18 report §4).

## 6. Training — copy-adapted from 06, which copy-adapted from 04-cpt-sft

Nothing in the recipe was rewritten. `stock_probe/` and `spectrum_probe/` are
06's trees with paths retargeted to `07-nitin-ds-new-sft` and the drivers
ported to Jac. Hyperparameters, `iters: 8200`, `batch_size: 1`,
`learning_rate: 2.0e-5` (cosine, warmup 820), `num_layers: 16`, LoRA rank 16 /
scale 2.0 / dropout 0.05, `max_seq_length: 3072`, `seed: 42`,
`mask_prompt: true`, `grad_checkpoint: true`, every watchdog / OOM ladder /
stall detector / dry-run gate / `--verify-layers` gate / `pgrep` preflight are
unchanged. That is what makes this a replication rather than a confounded new
experiment.

Three corrections inherited from 06 (§5.3 of its spec), still authoritative:

1. **`spectrum/configs/dpo_spectrum.yaml` does not exist.** DPO-spectrum
   hyperparameters are env-var defaults inside `run_dpo_spectrum.sh`
   (`DPO_ITERS=250`, `DPO_LR=1e-6`, `DPO_BETA=0.1`, `DPO_MAXLEN=512`) plus the
   shared `configs/dpo_lora.yaml`. Copying the runner carries them.
2. **The functional eval harness has exactly one copy in the repo:**
   `model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac`.
   It is env-var driven — **point at it, do not copy it.** Every eval script
   here already does.
3. **The stock DPO runner is `run_dpo_nofuse.sh`, never `run_dpo.sh`.**
   `run_dpo.sh` uses the `mlx_lm.fuse` path whose own results file reads 12%
   (107/855) — fusing an SFT LoRA delta into int4 weights re-quantizes and
   silently discards the delta. `run_dpo.sh` is deliberately absent from this
   tree.

`merge_frozen_keys.py`'s in-place-`mx.load`-truncation bug applies only to
`sft_cptv2_probe/`; a fresh base has no frozen prior-adapter keys, so neither
arm here touches that code path.

**Preflight hazard, unchanged:** the runners refuse to start if
`pgrep -f "jac start"` or `pgrep -f "mlx_lm"` finds anything running — a
resident model on this 48GB box collides with training memory. Check
`pgrep -f "jac start"; pgrep -f mlx_lm` and kill strays before any launch. The
`jms/` server in this repo is a frequent offender.

## 7. Decontamination — reuse the existing shingle machinery

`scripts/pipeline.jac` already carries it, ported from
`01-sft-dpo/sft_dpo/jacgen2/{decontam_v2,dedup2}.jac`: 14-token shingles over
comment-stripped, whitespace-tokenized code, contaminated at ≥ 0.5 overlap,
plus a relaxed identifier-only 10-shingle pass and a function-name-collision
audit at ident-Jaccard ≥ 0.5. Three mandatory passes: within-corpus dedup
(exact hash then near-dup), training pool vs the existing holdouts, and new
holdout vs everything.

Reference set as shipped (identical to 06's): `04-cpt-sft`'s
`sft_fresh_probe/dataset/{sft,dpo}/valid.jsonl` and `01-sft-dpo`'s
`dataset/eval_holdout/{conversion,graph_conversion}.jsonl`.
`[TODO: decide whether 06's own holdout
(`06-nitin-ds-sft/dataset/nitin_holdout.jsonl`) should be added as a fifth
reference set. The new corpus is a successor to 06's, so overlap is plausible —
but it is only a *required* decontam target if 07 also reports numbers on 06's
holdout. Left off by default so the triage logic stays mechanically identical.]`
The TODO is also inline in `scripts/pipeline.jac`.

**No silent caps.** Every drop is counted and attributed by reason in
`docs/reports/corpus-triage-report.md`. A pool that shrinks a lot is a finding
to report, not a number to round up.

## 8. Live monitoring (standing user requirement)

Loss curves and periodic eval-subset pass-rate curves are captured **as runs
progress**, not reconstructed afterward. `scripts/plot_progress.jac` regenerates
PNGs from `train.log` + `metrics_functional.jsonl` on demand (one shot — the
orchestrating session calls it on a loop; it does not loop itself), writing to

```
model-experiments/07-nitin-ds-new-sft/<arm>_probe/results/<stage>/plots/
```

The orchestrating session — not a subagent — publishes these as an Artifact
dashboard. Subagents keep the PNGs current and in that exact location.

## 9. Directory layout

```
model-experiments/07-nitin-ds-new-sft/
  CONTEXT_BRIEF.md          <- this file
  docs/
    README.md  spec.md  workflow.md  dataset-structure.md
    reports/                <- triage + comparison reports as they land
    reports/failure_data/   <- per-row generation dumps for failure analysis
  dataset/                  <- EMPTY until Stage 0 (gitignored, regenerable)
    candidate_pool.jsonl    <- clean, deduped, decontaminated pool
    nitin_holdout.jsonl     <- frozen holdout, carved pre-generation
    nitin_holdout_eval.jsonl<- holdout WITH authored instructions (see §4)
    sft_train.jsonl  dpo_train.jsonl
    sft/{train,valid}.jsonl  dpo/{train,valid}.jsonl   <- split dirs mlx reads
    raw_output/{sft,dpo,holdout_eval}/  <- batch handoff files
    rejected/{sft,dpo}/     <- every drop, with a reason
  external/                 <- workspace for corpus-derived intermediates
  scripts/                  <- all .jac (+ prep_training_dirs.sh)
  stock_probe/              <- stock arm, trailing-16 LoRA
  spectrum_probe/           <- spectrum arm, SNR-picked blocks
```

Both arms share ONE `dataset/` — they train on identical data by design, so two
copies could only drift.

**Adapter names as scaffolded** (06's `-nitin` suffix retained):

| Arm | SFT adapter | DPO adapter | DPO-best |
|---|---|---|---|
| stock | `adapters/sft-on-nitin` | `adapters/dpo-on-sft-nitin-nofuse` | `…-nofuse-best` |
| spectrum | `adapters/sft-on-nitin-spectrum` | `adapters/dpo-on-sft-nitin-spectrum` | `…-spectrum-best` |

## 10. Sequencing (respect the dependency order)

```
pin the corpus (§1 TODOs) -> compile-check + triage + dedup + decontam + holdout carve
  -> SFT-pair authoring (reverse-instruction, Opus) + holdout instruction authoring
  -> DPO-pair authoring (buggy-variant, Fable)
  -> prep_training_dirs.sh (85/15, seed 42)
  -> SFT training x2 arms   (sequential only — one training process on this box)
  -> DPO training x2 arms   (each resumes from its OWN arm's SFT adapter)
  -> eval x2 arms x 2 holdouts x 3 stages
  -> comparison report (12 cells + cross-phase vs 06 and vs 04-cpt-sft fresh)
```

No LLM API key exists in this environment. Every "LLM generation" step goes
through the batch-handoff architecture: `scripts/prepare_batches.jac` writes
`pending_batch_<track>_<NN>.jsonl`, a dispatched Claude Code Agent fills
`responses_batch_<track>_<NN>.jsonl`, and `scripts/collect.jac` gates and
persists. Reverse-instruction authoring is bulk → **Opus**; buggy-variant
authoring is precision → **Fable**. See
`04-cpt-sft/docs/reports/2026-07-task18-full-run-report.md` §2 for why.

## 11. Failure modes already paid for — do not rediscover these

| Symptom | Cause |
|---|---|
| DPO collapses to ~2–12% pass | `mlx_lm.fuse` on an int4 model re-quantizes and discards the SFT LoRA delta |
| Spectrum adapter loads but scores like a partial model | `adapter_config.json` not rewritten; `load_weights(strict=False)` silently drops out-of-slice blocks |
| Holdout-B eval "finishes" in 30 seconds | holdout rows have no `messages` field — nothing was ever generated (06 §4 incident 4) |
| Best-checkpoint tracker resets after a crash-resume | the watchdog's "best" comparison does not persist across a restart; diff `runs_pct` across both halves of the log (06 incident 2) |
| Bus error / `tee: Input/output error` mid-run | the external drive dropping off the bus; verify every artifact after remount before resuming (06 incidents 1 and 3) |
| DPO OOM at 37GB+ free RAM | macOS GPU wired-memory ceiling, not system RAM; policy + reference both resident |
| Behavioral gate always passes | `gate_class="behavioral"` with an empty `expected_output` |
| Script runs as a side effect of being imported | `with entry { }` instead of `with entry:__main__ { }` |
| Cross-directory Jac import dies at runtime but passes `jac check` | relative import with no known parent package — inline the shared logic |
