# workflow.md — 07-nitin-ds-new-sft Runbook

Execution order for the two-arm Spectrum replication on Nitin's **new** corpus
(`[TODO: confirm repo @ commit]`). Companion to [`spec.md`](spec.md) (design,
decision rules) and [`../CONTEXT_BRIEF.md`](../CONTEXT_BRIEF.md) (settled
facts). This file is the "what do I actually type" document.

Dependency order is real, not advisory:

```
pin the corpus (CONTEXT_BRIEF §1 TODOs)              [Stage -1]
  -> corpus triage + dedup + decontam + holdout carve [Stage 0]
  -> SFT-pair authoring (reverse-instruction, Opus)   [Stage 1]
     ... incl. the HOLDOUT rows (Stage 1b) -- do not skip
  -> DPO-pair authoring (buggy-variant, Fable)        [Stage 2]
  -> SFT training x2 arms (sequential only)           [Stage 3]
  -> DPO training x2 arms (each on its own SFT adapter)[Stage 4]
  -> eval: 2 arms x 2 holdouts x 3 stages             [Stage 5]
  -> comparison report                                [Stage 6]
```

## Conventions used throughout

- **Run everything from the repo root**
  (`/Volumes/ExtremePro/JaseciLabs/jac_model_studio`). Every path below is
  repo-root-relative.
- **Everything is Jac.** All scripts in this phase are `.jac`; the `.sh`
  wrappers call `jac run <driver>.jac <flags>`. `jac run` sets
  `sys.argv = [filename] + flags`, so each driver's own argparse/argv handling
  is unchanged. Any NEW script is `.jac` from the start.
- **Use the absolute venv binary**, not bare `jac` — a bare `jac` on `PATH` can
  resolve to a different, stale venv (task18 report §4):
  `/Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac`.
  `[TODO: the run/eval wrappers currently call bare `jac run` for the drivers.
  Confirm the resolved `jac` is the same environment that has mlx / mlx_lm /
  mlx_lm_lora installed, or pin the absolute path in each wrapper. Every
  wrapper carries this TODO inline.]`
- **`with entry:__main__ { }`**, never a bare `with entry { }` — plain
  `with entry` executes on *import* (task18 report §4). Every ported driver
  follows this, and it matters: `dpo_spectrum_train.jac` imports
  `dpo_fixed_train.jac` as a module.
- **Bulk file work happens on the internal clone.** The external drive measured
  ~4.3 MB/s; compile-checking thousands of files there is not viable. Only final
  JSONL / PNG / report artifacts get written into
  `model-experiments/07-nitin-ds-new-sft/`. `compile_check.jac` stages through
  `/tmp/nitin_new_triage/`.
- **No silent caps.** Every dropped item is counted, attributed by reason,
  written under `dataset/rejected/`, and summarized in the triage report.

## Stage -1 — Pin the corpus

Nothing downstream can run until these are filled in. They live as inline
`[TODO]`s in the scripts, which are the single source of truth:

| File | Constant | What to set |
|---|---|---|
| `scripts/pipeline.jac` | `CORPUS` | repo slug used in every `origin` string |
| `scripts/pipeline.jac` | `SRC` | absolute path to the corpus files on the INTERNAL clone |
| `scripts/pipeline.jac` | `COMMIT` | the commit SHA — currently the literal `"TODO-CONFIRM-COMMIT-SHA"` |
| `scripts/compile_check.jac` | `SRC` | same corpus path |

Also sample a dozen files and record in `../CONTEXT_BRIEF.md` §1 / `spec.md`
§2.1 whether this corpus is the same flat-function shape as 06's or contains
graph-native material. Two inherited decisions (single SFT task type,
correctness-only DPO axis) hang off that answer.

## Stage 0 — Corpus triage, dedup, decontam, holdout carve

```
.venv/bin/jac run model-experiments/07-nitin-ds-new-sft/scripts/compile_check.jac
.venv/bin/jac run model-experiments/07-nitin-ds-new-sft/scripts/pipeline.jac
```

`compile_check.jac` runs `jac run` over every corpus file (12 worker threads,
60 s timeout each) and appends to `/tmp/nitin_new_triage/compile_results.jsonl`
— it is resumable, re-running skips files already recorded. `pipeline.jac` then
does the whole funnel in one pass:

| Stage | What it does |
|---|---:|
| S0 | load corpus at the commit pin |
| S1 | compile gate — `jac run` exit != 0 dropped |
| S2 | quality filter — no_def / multi_def / no_docstring / trivial_body (< 12 body tokens or < 3 code lines) |
| S3 | exact dedup on normalized token hash |
| S4 | near-dup dedup, 14-token shingles, ≥ 0.5 overlap |
| S5 | decontamination vs the four reference holdouts (strict 14-token + relaxed identifier-only 10-token) |
| S5b | function-name collision audit, drop at ident-Jaccard ≥ 0.5 |
| S6 | holdout carve (seed 42) + leak-check holdout vs candidate pool |

Outputs:

| Path | Contents |
|---|---|
| `dataset/candidate_pool.jsonl` | clean, deduped, decontaminated pool — pre-holdout-split |
| `dataset/nitin_holdout.jsonl` | frozen holdout, carved per `spec.md` §4.2(b) |
| `/tmp/nitin_new_triage/stats.json` | full funnel + every drop with its reason |
| `docs/reports/corpus-triage-report.md` | write this up from `stats.json` |

**Gate to leave Stage 0:** both JSONLs exist, their id sets are provably
disjoint, the leak-check reports 0 failures, and the triage report states final
counts with every drop attributed. Report the true holdout size — do not force
855 if the pool cannot support it (`spec.md` §4.2(b)).

## Stage 1 — SFT-pair authoring (reverse-instruction, Opus)

Turn each pooled file into one `messages`-format SFT example: an authored NL
instruction, and the file's Jac as the assistant turn.

```
JAC_BATCH_TRACK=sft .venv/bin/jac run \
    model-experiments/07-nitin-ds-new-sft/scripts/prepare_batches.jac
# ... dispatch agents to fill each responses_batch_sft_NN.jsonl ...
JAC_BATCH_TRACK=sft .venv/bin/jac run \
    model-experiments/07-nitin-ds-new-sft/scripts/collect.jac
```

- `prepare_batches.jac` reads `dataset/candidate_pool.jsonl` and writes
  `dataset/raw_output/sft/pending_batch_sft_NN.jsonl`, one record per file with
  its docstring, signature, and full Jac body. It **asserts** the pool is
  disjoint from `nitin_holdout.jsonl` and refuses to run otherwise.
- `[TODO: re-derive `SFT_SHARDS` from the new pool size. Rule: 300–400 items per
  shard, the band one dispatched Opus agent carries in a single sitting. The
  scaffolded value 16 is 06's number (5,474 rows / 16 shards) carried as a
  placeholder, not a measured number for this corpus.]`
- **Model: `opus`.** Reverse-instruction authoring is bulk, mechanical work off
  a known-good target. One dispatch fills a whole shard, not one prompt.
- The agent writes instruction text only. It never touches the Jac — the
  completion is the corpus file verbatim, and `collect.jac` gate 4 enforces
  byte-identity.

**Emitted record schema** mirrors
`04-cpt-sft/dataset/fresh/releases/sft_train.jsonl`'s field set, with, for this
phase: `category="conversion"`, `task_type="python_to_jac_function"`,
`origin="<repo>:<commit>:<path>"`, `gate_class="compile_only"` (honest — this
corpus ships no pinned expected outputs, so no behavioural assertion is
available; do **not** label these `behavioral`), `generator="opus-api"`,
`dataset_version="nitin-new-ds-v1.0.0"`, `run_tag="nitin-new"`.

**Collect-phase SFT gates** (all in `collect.jac`, every rejection logged):
`ok == true` and instruction non-empty; instruction is prose not code (no
fence, no `{`+`;` pair, no long verbatim line lifted from the answer);
instruction is not a bare restatement of the function name and clears a length
floor; the assistant turn is byte-identical to the pool's `jac_code`; compile
evidence transfers from triage (set `JAC_RECHECK_COMPILE=1` to re-run `jac run`
on every row anyway — hours, and it re-proves an already-proven artifact).

**Gate to leave Stage 1:** `dataset/sft_train.jsonl` exists, its id set is
disjoint from the holdout, and the accept/reject accounting is written.

### Stage 1b — Author instructions for the HOLDOUT rows too

**Do not skip this.** 06 froze its holdout from the raw candidate pool, so the
rows had `jac_code` + `docstring` but no `messages` — and `eval_functional.jac`
had nothing to prompt the model with. The first holdout-B eval "completed" in
under 30 seconds having never generated anything, and had to be redone after
reverse-authoring instructions for all 855 rows.

Run the same reverse-instruction pass over `dataset/nitin_holdout.jsonl`
(batches under `dataset/raw_output/holdout_eval/`), emitting
`dataset/nitin_holdout_eval.jsonl` in the same `messages` + `gate_class` shape
the harness reads. `scripts/gen_eval_detail.jac` already points holdout `"B"` at
that filename.

**Gate:** `nitin_holdout_eval.jsonl` exists, has the same row count as
`nitin_holdout.jsonl`, and every row has a non-empty `messages` field.

## Stage 2 — DPO-pair authoring (buggy-variant, Fable)

Correctness axis only (`spec.md` §3.1). `chosen` = the file's Jac as-is;
`rejected` = the same function with one subtly-introduced logic bug.

```
JAC_BATCH_TRACK=dpo .venv/bin/jac run \
    model-experiments/07-nitin-ds-new-sft/scripts/prepare_batches.jac
# ... dispatch agents to fill each responses_batch_dpo_NN.jsonl ...
JAC_BATCH_TRACK=dpo .venv/bin/jac run \
    model-experiments/07-nitin-ds-new-sft/scripts/collect.jac
```

- Samples a seeded subset (`DPO_SAMPLE_N`, `DPO_SAMPLE_SEED=42`) from rows
  clearing an eligibility floor (`DPO_MIN_TOKENS`, `DPO_MIN_CODE_LINES`) — a
  subtle behavioural bug has nowhere to hide in a two-line function.
  `[TODO: recheck the floor against the new corpus's own median body size and
  record how many rows qualify. 06's floor was 30 tokens / 5 code lines.]`
- **Model: `fable`.** Buggy-variant authoring is precision work — a bug that
  doesn't change behaviour, or one that breaks the parse, poisons the pair
  either way.
- Bug types requested, one per pair: off-by-one, flipped comparison operator,
  swapped branch bodies, mutated default argument, inverted accumulator,
  dropped edge-case guard.

**Dual gate at collect, both mandatory:** `chosen` compiles (`jac run` exit 0),
and `rejected` **also** compiles. A rejected side that fails to parse teaches
"prefer code that compiles" instead of "prefer code that is correct" — the pair
is dropped and logged. Plus: `chosen != rejected` byte-wise AND after
comment/whitespace normalization, and `def <func_name>` still present in the
rejected side.

Because this corpus has no pinned expected outputs, "the bug actually changes
behaviour" cannot be mechanically proven. State that in the release notes rather
than implying a behavioural check happened.

**Output shape:** `prompt` / `chosen` / `rejected`, matching
`04-cpt-sft/sft_fresh_probe/dataset/dpo/{train,valid}.jsonl`.

**Gate to leave Stage 2:** `dataset/dpo_train.jsonl` exists, every pair
dual-gated, drop accounting written.

## Stage 3 — SFT training ×2 arms

### 3.0 The probe directories (as built)

Scaffolded and path-verified; copied from 06 and retargeted **only** on data
paths, adapter paths, and results paths, with the drivers ported to Jac.
Hyperparameters, iters, seed, LoRA geometry, and every watchdog / OOM ladder /
stall detector / dry-run gate / `--verify-layers` gate / `pgrep` collision
preflight are unchanged — that is what makes this a replication.

```
model-experiments/07-nitin-ds-new-sft/
  dataset/
    sft/{train,valid}.jsonl     <- ONE shared copy, both arms
    dpo/{train,valid}.jsonl     <- ONE shared copy, both arms
  scripts/
    compile_check.jac  pipeline.jac        <- Stage 0
    prepare_batches.jac  collect.jac       <- Stages 1-2
    prep_training_dirs.sh                  <- release JSONL -> train/valid split (3.0b)
    plot_progress.jac                      <- live curves (3.4)
    gen_eval_detail.jac  grade_eval_detail.jac  grade_reference.jac  <- failure analysis
  stock_probe/
    configs/sft.yaml            <- 06's sft.yaml; ONLY data + adapter_path changed
    configs/dpo_lora.yaml       <- unchanged
    dpo_fixed_train.jac         <- run_dpo_nofuse.sh's DRIVER
    run_sft.sh                  <- retargeted
    run_dpo_nofuse.sh           <- retargeted  (NOT run_dpo.sh -- see spec.md §5.3)
    eval_sft_sweep.sh  eval_dpo_nofuse.sh
    adapters/  results/
  spectrum_probe/
    configs/dpo_lora.yaml       <- unchanged
    dpo_fixed_train.jac         <- byte-identical to stock_probe's copy;
                                   dpo_spectrum_train.jac imports it from
                                   PROBE_DIR (its own parent), so this copy is
                                   REQUIRED, not incidental
    spectrum/
      spectrum_lora_layers.jac  <- SFT driver (drop-in for mlx_lm.lora)
      dpo_spectrum_train.jac    <- DPO driver
      adapter_config_fix.jac    <- adapter_config.json rewriter
      configs/sft_spectrum.yaml <- ONLY data + adapter_path changed
      configs/spectrum_layers.json  <- BYTE-IDENTICAL to 06's (sha256-verified)
    run_sft_spectrum.sh  run_dpo_spectrum.sh
    eval_sft_spectrum.sh  eval_dpo_spectrum.sh
    adapters/  results/
```

**ONE shared `dataset/`, not a copy per arm.** Both arms train on identical data
by design — that is the entire point of the replication — so two copies could
only drift. `data:` in both arms' SFT yamls and `--data` in both DPO runners
point at `model-experiments/07-nitin-ds-new-sft/dataset/{sft,dpo}`.

**Adapter names as built** (06's `-nitin` suffix retained):

| Arm | SFT adapter | DPO adapter | DPO-best adapter |
|---|---|---|---|
| stock | `adapters/sft-on-nitin` | `adapters/dpo-on-sft-nitin-nofuse` | `…-nofuse-best` |
| spectrum | `adapters/sft-on-nitin-spectrum` | `adapters/dpo-on-sft-nitin-spectrum` | `…-spectrum-best` |

**Verified during scaffolding:**

- `bash -n` clean on all 9 shell scripts.
- Every `.jac` file passed `validate_jac`.
- `spectrum_layers.json` sha256 matches 06's byte-for-byte:
  `[0, 22, 23, 27, 30, 34, 36, 37, 38, 39, 41, 42, 43, 44, 45, 47]`.
- The only surviving `04-cpt-sft` references in the wrappers are the two
  intended ones: the functional harness
  `04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac` (point at it, do not
  copy it) and the holdout-(a) default.
- `[TODO: on the first real launch, echo `$PWD` to confirm the
  `cd "$(cd "$(dirname "$0")/../../.." && pwd)"` repo-root hop still resolves —
  `07-nitin-ds-new-sft/<arm>_probe/` is the same depth as 06's, so it should,
  but verify rather than assume.]`

**`HOLDOUT` and `RDIR` are env-overridable** (needed for Stage 5.2's second
holdout):

- Every script defaults `HOLDOUT` to holdout (a),
  `04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl`. Unset it and you get the
  replication path.
- The four **eval** scripts also default `RDIR` overridably, with a `mkdir -p`
  so a redirected results dir gets created.
- The two **DPO eval** scripts have a `TRAIN_RDIR` (defaulting to the arm's own
  training results dir). `.dpo_progress_steps` / `.best_step` are read from
  `TRAIN_RDIR`, never from a redirected `RDIR` — reading them from a redirected
  `RDIR` silently falls back to 250/250 and then skips the DPO-best cell
  entirely via the `BEST_STEP != FINAL_STEP` test.

**Open item flagged for the orchestrator, unchanged from 06:** the two DPO
*runners* also default `HOLDOUT` to holdout (a), so the in-loop subset eval that
selects the DPO-**best** snapshot selects on holdout (a). That is what 06 and
04-cpt-sft's fresh arm both did, so the cross-phase DPO-best comparison stays
apples-to-apples — but holdout (b)'s DPO-best cell is scored on a snapshot
chosen by holdout (a). Deliberate, conservative, and it must be said in the
report. Related: the SFT-baseline lookup inside both DPO runners filters on
`total == 855`, so pointing a DPO *training* run at the new holdout would
silently drop the collapse gate to its 30% absolute floor. Both runners carry an
inline comment saying so.

### 3.0b Build the train/valid split dirs

`collect.jac` emits **single release files** — `dataset/sft_train.jsonl` and
`dataset/dpo_train.jsonl`. `mlx_lm` / `mlx_lm_lora` resolve `data:` to a
**directory** containing `train.jsonl` + `valid.jsonl`. Bridge them with:

```
model-experiments/07-nitin-ds-new-sft/scripts/prep_training_dirs.sh
```

- **Ratio: 85/15, seed 42** — matching 04-cpt-sft's fresh arm and 06 exactly.
  Overridable via `TRAIN_FRAC` / `SPLIT_SEED`, but don't: the ratio is part of
  the recipe.
- Deterministic (sorts, then seeded-shuffles) — same release file gives the same
  split every time.
- Drops and reports rows that are not valid JSON or that lack `messages` (SFT) /
  `prompt`+`chosen`+`rejected` (DPO). No silent caps.
- Refuses to overwrite an existing non-empty split unless `FORCE=1`.
- Skips cleanly when a release file doesn't exist yet, so it is safe to run while
  Stages 1/2 are still filling batches.

These `valid.jsonl` files are `mlx_lm_lora`'s **training-time** validation
splits, not scored eval sets. Every reported number comes from the eval scripts
against holdout (a) or (b).

**Run this before Stage 3.2** — both SFT runners preflight-check for
`dataset/sft/{train,valid}.jsonl` and refuse to start without them.

### 3.1 Preflight (every training launch, both arms)

```
pgrep -f "jac start"; pgrep -f mlx_lm
```

Both must return nothing. A resident JMS server or stray `mlx_lm` process
collides with training memory and OOMs this 48GB box regardless of what the
script does — the runners refuse to start if either is found, which is the gate
working, not a bug to bypass. This repo's `jms/` server is a frequent offender;
kill it first.

### 3.2 Stock arm SFT

```
CONFIRM_FULL_RUN=1 model-experiments/07-nitin-ds-new-sft/stock_probe/run_sft.sh
```

Without `CONFIRM_FULL_RUN=1` the script performs its self-test/dry-run and exits
before the multi-hour run — that is the intended safety gate. Outputs:
`stock_probe/adapters/sft-on-nitin/`,
`stock_probe/results/sft/{train.log,metrics_functional.jsonl}`.

### 3.3 Spectrum arm SFT

```
CONFIRM_FULL_RUN=1 model-experiments/07-nitin-ds-new-sft/spectrum_probe/run_sft_spectrum.sh
```

The `--verify-layers` preflight (`jac run spectrum_lora_layers.jac
--verify-layers ...`) loads `models/qwen-q4` twice and must print `VERIFY: PASS`
with **281.838M** trainable parameters. Training is gated on it. A different
count means the selection changed *capacity* rather than *placement*, which
invalidates the comparison — do not `SKIP_VERIFY=1` past it. Outputs:
`spectrum_probe/adapters/sft-on-nitin-spectrum/`,
`spectrum_probe/results/sft-spectrum/`.

**Sequential only.** One training process at a time on this box. Stock SFT must
finish before spectrum SFT starts.

### 3.4 Live monitoring (runs alongside 3.2/3.3)

`run_sft_spectrum.sh`'s watchdog already calls
`model-experiments/01-sft-dpo/sft_dpo/jacgen/plot_metrics.jac` every
`EVAL_EVERY` seconds. In addition, call

```
.venv/bin/jac run model-experiments/07-nitin-ds-new-sft/scripts/plot_progress.jac \
    --train-log <arm>_probe/results/<stage>/train.log \
    --out       <arm>_probe/results/<stage>/plots/loss.png \
    --eval-curve <arm>_probe/results/<stage>/metrics_functional.jsonl \
    --eval-out   <arm>_probe/results/<stage>/plots/eval.png
```

on a loop from the orchestrating session — `plot_progress.jac` is one-shot by
design and does not loop itself. The orchestrating session publishes the PNGs as
an Artifact dashboard; subagents only keep them current and in that exact
location.

## Stage 4 — DPO training ×2 arms

Each arm's DPO seeds from **its own** SFT adapter via `--resume-adapter-file`.
Never fuse first (`spec.md` §5.3 item 3).

```
CONFIRM_FULL_RUN=1 model-experiments/07-nitin-ds-new-sft/stock_probe/run_dpo_nofuse.sh
CONFIRM_FULL_RUN=1 model-experiments/07-nitin-ds-new-sft/spectrum_probe/run_dpo_spectrum.sh
```

Order: stock SFT → spectrum SFT → stock DPO → spectrum DPO. Sequential
throughout.

`run_dpo_spectrum.sh` carries three gates of its own — keep all three:
`--verify-patches` (seconds, no model load: both monkey-patches live, all three
call sites rebound), `--verify-layers` (minutes: the real DPO conversion path
converts the picks, trainable == 281.838M, and every SFT adapter key finds a
home — `mlx_lm_lora/train.py` loads with `strict=False` and will not tell you
otherwise), and the per-snapshot `adapter_config.json` rewrite before any
in-loop scoring.

**Known hazard — DPO OOM.** DPO holds both policy and reference model in memory.
The 04-cpt-sft cptv2 arm hit macOS's GPU wired-memory ceiling at
`DPO_MAXLEN=512` and had to drop to 384. The fresh arm and 06 both ran clean at
512, so **start at 512** — it is the value the baselines were produced at. If it
OOMs, the runner's shrink ladder handles it; if you end up at 384, that is a
recipe deviation and must be reported in the comparison, not footnoted.

**Known hazard — best-checkpoint tracking across a resume.** In 06 the external
drive dropped off the bus at DPO step 200/250. The watchdog resumed correctly
from 200, but its "best snapshot" comparison did **not** persist: it named step
220 (54.0%) as best, unaware the pre-crash half of the run had already scored
step 60 at 58.0%. Caught only by diffing `runs_pct` across both halves of the
log, and fixed by manually overwriting the `-best` adapter dir and `.best_step`.
**After any resume, diff `runs_pct` across every segment of `train.log` before
trusting `.best_step`.**

Each arm produces both a final adapter and a `-best` adapter (best snapshot by
the in-loop subset gate), which is what makes the DPO-best and DPO-final rows
distinct.

## Stage 5 — Eval: 2 arms × 2 holdouts × 3 stages

Twelve cells. All scoring goes through the single functional harness,
`model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac`,
driven by env vars — the same harness that produced every number in
`04-cpt-sft/RESULTS.md` and in 06's report. Using anything else forfeits
cross-phase comparability, which is the whole point of holdout (a).

### 5.1 Holdout (a) — the shared 855

`HOLDOUT="model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl"`
— 1,428 rows, 855 code-graded. **Reuse unchanged. Do not regenerate.**

```
model-experiments/07-nitin-ds-new-sft/stock_probe/eval_sft_sweep.sh
model-experiments/07-nitin-ds-new-sft/spectrum_probe/eval_sft_spectrum.sh
model-experiments/07-nitin-ds-new-sft/stock_probe/eval_dpo_nofuse.sh
model-experiments/07-nitin-ds-new-sft/spectrum_probe/eval_dpo_spectrum.sh
```

Each writes `base.txt` / `final.txt` / `final_best.txt` / `final_last.txt` plus
`metrics_functional.jsonl` into its arm's results dir. The `final*.txt`
`OVERALL runs: NN% (x/855)` line is the cell value; the per-slice lines above it
are the `spec.md` §7.2 per-category breakdown.

### 5.2 Holdout (b) — the new-corpus holdout

Same four scripts, with `HOLDOUT` and `RDIR` overridden so the (a) numbers are
not overwritten. **Point at `nitin_holdout_eval.jsonl`, not
`nitin_holdout.jsonl`** — the harness needs the `messages` field (Stage 1b).

```
H=model-experiments/07-nitin-ds-new-sft/dataset/nitin_holdout_eval.jsonl
S=model-experiments/07-nitin-ds-new-sft/stock_probe
P=model-experiments/07-nitin-ds-new-sft/spectrum_probe

HOLDOUT=$H RDIR=$S/results/sft-holdoutB          $S/eval_sft_sweep.sh
HOLDOUT=$H RDIR=$P/results/sft-spectrum-holdoutB $P/eval_sft_spectrum.sh
HOLDOUT=$H RDIR=$S/results/dpo-nofuse-holdoutB   $S/eval_dpo_nofuse.sh
HOLDOUT=$H RDIR=$P/results/dpo-spectrum-holdoutB $P/eval_dpo_spectrum.sh
```

`TRAIN_RDIR` is deliberately **not** overridden — the two DPO eval scripts read
`.dpo_progress_steps` / `.best_step` from it, and those live in the training
results dir regardless of which holdout is being scored.

**Sanity check before trusting a holdout-B number:** if a full 855-row eval
"finishes" in seconds, it did not run. That is the 06 incident — rows without
`messages` generate nothing. Check the row count in `metrics_functional.jsonl`.

### 5.3 Non-negotiable per-eval checks

- `adapter_config_fix.jac` runs **before** any scoring on the spectrum arm, and
  the key assertion passes: **256 keys** (16 per block × 16 blocks). Skipping it
  makes `load_adapters` rebuild LoRA on blocks 32–47 from the adapter's own
  `num_layers: 16`, and `load_weights(strict=False)` silently drops blocks
  0/22/23/27/30 — the arm gets scored as a partially-loaded model, with no error.
- Confirm no all-zero adapter weights before eval, not after a confusing number.
- Score the base model once (`JAC_EVAL_ADAPTER=""`) per holdout as the floor.

### 5.4 Failure analysis (optional, after the headline numbers)

```
ADAPTER=<path> OUT_PREFIX=docs/reports/failure_data/<tag> HOLDOUTS=A,B \
  .venv/bin/jac run model-experiments/07-nitin-ds-new-sft/scripts/gen_eval_detail.jac
.venv/bin/jac run model-experiments/07-nitin-ds-new-sft/scripts/grade_eval_detail.jac
HOLDOUT=<valid.jsonl> OUT=<jsonl> \
  .venv/bin/jac run model-experiments/07-nitin-ds-new-sft/scripts/grade_reference.jac
```

`gen_eval_detail.jac` mirrors the harness's generation path exactly and persists
one record per holdout row; `grade_eval_detail.jac` grades that dump;
`grade_reference.jac` measures the eval set's own ceiling by running each row's
*reference* answer through the same gate — rows whose reference does not survive
are unpassable by any model, and their contribution is a property of the eval
set, not the model.

## Stage 6 — Comparison report

Write to `docs/reports/2026-08-newnitin-vs-nitin-spectrum-comparison.md`,
following `06-nitin-ds-sft/docs/reports/2026-08-final-comparison.md`'s structure:
bottom line up front → headline table per holdout → cross-dataset table →
real problems hit → honest reading → live graphs → artifacts.

Contents, per `spec.md` §7:

1. **The 12-cell matrix**, both holdouts, all three stages, both arms.
2. **Cross-phase paired McNemar** of each holdout-(a) cell against its 06
   counterpart (619 / 636 / 619 / 631 / 583 / 636, all of 855) → RQ2. Decision rule: ≥ 4 of 6 significant wins, no significant losses.
   Show the 04-cpt-sft fresh-arm column (597 / 639 / 597 / 634 / 531 / 622)
   alongside as the three-way lineage view.
3. **Arm-vs-arm paired McNemar** per column (6 comparisons) → RQ1, same ≥ 4-of-6 rule. Say explicitly how it lands relative to 06's
   non-replication.
4. **Per-slice breakdown on holdout (a)** — 322 of the 855 graded rows are
   `conversion`. A win concentrated there is a specialization result and must be
   labeled that way.
5. **Power statement for holdout (b)**, computed from its real size, plus a
   ceiling note if it saturates the way 06's did (96–99%).
6. **Named confounds in the headline, not buried**: training-set size vs 06 and
   vs 04-cpt-sft; single seed per arm, so deltas are tested against sampling
   noise only; DPO `chosen` correctness assumed from a compile gate, not proven
   behaviourally; docstring/code inconsistency label noise in the corpus; any
   `DPO_MAXLEN` deviation from 512.
7. **Incidents**, with the same forensic detail as the prior reports — what was
   observed, what was ruled out and how, what the real cause was, what the fix
   was.

## Quick reference — failure modes already paid for

| Symptom | Cause | Source |
|---|---|---|
| DPO collapses to ~2–12% pass | `mlx_lm.fuse` on an int4 model re-quantizes and discards the SFT LoRA delta | 04-cpt-sft comparison report §3, `run_dpo_nofuse.sh` header |
| Spectrum adapter loads but scores like a partial model | `adapter_config.json` not rewritten; `load_weights(strict=False)` drops out-of-slice blocks silently | `eval_sft_spectrum.sh` header |
| Holdout-B eval "finishes" in 30 seconds | holdout rows have no `messages` field — nothing was generated | 06 final report §4 incident 4 |
| Best checkpoint is wrong after a crash-resume | the watchdog's best-snapshot comparison does not persist across a restart | 06 final report §4 incident 2 |
| `tee: Input/output error`, bus error mid-run | the external drive dropping off the bus | 06 final report §4 incidents 1 and 3 |
| All LoRA weights read back exactly zero | `mx.load()` + in-place overwrite with no `mx.eval()` between — truncates the mmap being lazily read | 04-cpt-sft comparison report §3.1 (cptv2 arm only; not on this phase's path) |
| DPO OOM at 37GB+ free RAM | macOS GPU wired-memory ceiling, not system RAM; policy + reference both resident | 04-cpt-sft comparison report §3.2 |
| Behavioral gate always passes | `gate_class="behavioral"` with an empty `expected_output` — a truthiness check on `""` | task18 report §4 |
| Cross-directory Jac import fails at runtime but passes `jac check` | relative import with no known parent package — inline the shared logic | task18 report §4 |
| Script runs as a side effect of being imported | `with entry { }` instead of `with entry:__main__ { }` | task18 report §4 |
