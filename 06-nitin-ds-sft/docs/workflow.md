# workflow.md — 06-nitin-ds-sft Runbook

Execution order for the two-arm Spectrum replication on Nitin's corpus
(`chess10kp/jac-data-gen` @ `7c25aff3110f526eec59e0123ffe6c0c152cce91`).
Companion to [`spec.md`](spec.md) (design, decision rules) and
[`../CONTEXT_BRIEF.md`](../CONTEXT_BRIEF.md) (settled facts). This file is
the "what do I actually type" document.

Dependency order is real, not advisory — §9 of the brief:

```
corpus triage + dedup + decontam + holdout carve      [Stage 0, concurrent agent]
  -> SFT-pair authoring (reverse-instruction, Opus)   [Stage 1]
  -> DPO-pair authoring (buggy-variant, Fable)        [Stage 2]
  -> SFT training x2 arms (sequential only)           [Stage 3]
  -> DPO training x2 arms (each on its own SFT adapter)[Stage 4]
  -> eval: 2 arms x 2 holdouts x 3 stages             [Stage 5]
  -> comparison report                                [Stage 6]
```

## Conventions used throughout

- **Run everything from the repo root**
  (`/Volumes/ExtremePro/JaseciLabs/jac_model_studio`). Every path below is
  repo-root-relative, matching the `jacgen`/`jacgen2` convention.
- **Use the absolute venv binary**, not bare `jac` — a bare `jac` on `PATH`
  resolves to a different, stale venv (task18 report §4):
  `/Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac`.
- **Bulk file work happens on the internal clone** (`~/repos/jac-data-gen`).
  The external drive measured ~4.3 MB/s; compile-checking 7,627 files there
  is not viable. Only final JSONL/PNG/report artifacts get written to
  `model-experiments/06-nitin-ds-sft/`.
- **No silent caps.** Every dropped item is counted and attributed by reason,
  written under `dataset/rejected/`, and summarized in the triage report.
- **`with entry:__main__ { }`**, not `with entry { }`, in any new Jac script —
  plain `with entry` executes on *import* (task18 report §4).

## Stage 0 — Corpus triage, dedup, decontam, holdout carve

> **Running concurrently in a separate agent. Do not block on these files
> existing; do not start Stage 1 before they do.**

Expected outputs:

| Path | Contents |
|---|---|
| `model-experiments/06-nitin-ds-sft/dataset/candidate_pool.jsonl` | clean, deduped, decontaminated pool — pre-holdout-split |
| `model-experiments/06-nitin-ds-sft/dataset/nitin_holdout.jsonl` | frozen holdout, carved per `spec.md` §4.2(b) |
| `model-experiments/06-nitin-ds-sft/docs/reports/corpus-triage-report.md` | yield accounting, drop reasons, final pool size |

What that agent is responsible for, restated so the interface is unambiguous:

1. Compile-gate all 7,627 files (`jac run` exit 0) on the internal clone.
   Files that do not compile are dropped, counted, and reported.
2. Within-corpus dedup: exact hash, then 14-token-shingle near-dup at
   ≥ 0.5 overlap.
3. Decontaminate the pool against **all** of:
   - `model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl`
   - `model-experiments/04-cpt-sft/sft_fresh_probe/dataset/dpo/valid.jsonl`
   - `model-experiments/01-sft-dpo/dataset/eval_holdout/conversion.jsonl`
   - `model-experiments/01-sft-dpo/dataset/eval_holdout/graph_conversion.jsonl`
4. Carve `nitin_holdout.jsonl` **before** any generation, and decontaminate
   it in both directions: against the remaining training pool, and against
   the (a) holdouts (a cross-holdout near-dup makes the two holdout columns
   non-independent — `spec.md` §6).
5. Report the true achievable holdout size. Do not force a target.

Reuse — do not rewrite — the shingle machinery in
`model-experiments/01-sft-dpo/sft_dpo/jacgen2/decontam_v2.jac`
(`shingles`, `is_contaminated`, `build_reference_shingles`; N = 14,
threshold 0.5). Porting it into a standalone script is fine; reimplementing
shingling from scratch is not.

**Gate to leave Stage 0:** `candidate_pool.jsonl` and `nitin_holdout.jsonl`
both exist, their id sets are provably disjoint, and the triage report states
final counts with every drop attributed.

## Stage 1 — SFT-pair authoring (reverse-instruction, Opus)

Turn each pooled file into one `messages`-format SFT example: an authored NL
instruction, and the file's Jac as the assistant turn.

**Scripts to write** under
`model-experiments/06-nitin-ds-sft/scripts/`:

| Script | Phase | Does |
|---|---|---|
| `prepare_sft_batch.jac` | prepare | reads `dataset/candidate_pool.jsonl`, writes `dataset/batches/sft/pending_batch.jsonl` — one record per file with its docstring, signature, and full Jac body |
| `collect_sft_batch.jac` | collect | joins `pending_batch.jsonl` + `responses_batch.jsonl`, gates, emits `dataset/sft_train.jsonl`, routes failures to `dataset/rejected/sft/` with a reason |

**The handoff.** There is no `ANTHROPIC_API_KEY` in this environment. Every
"LLM call" is a dispatched Claude Code Agent that reads the pending file and
fills a matching `responses_batch.jsonl` — the same batch-handoff
architecture `jacgen2` used for its entire production run
(`04-cpt-sft/docs/reports/2026-07-task18-full-run-report.md` §2). One
dispatch fills a whole batch, not one prompt; that is why the real cost is far
lower than a per-prompt estimate suggests.

- **Model: `opus`.** Reverse-instruction authoring is bulk, mechanical work
  off a known-good target — the project's Opus tier.
- Dispatch in chunks sized so one agent's batch is tractable. Record the
  batch index in the filename (`pending_batch.0001.jsonl` etc.) so a failed
  batch is re-runnable without redoing the rest.
- The agent writes instruction text only. It never touches the Jac code —
  the completion is the corpus file verbatim.

**Emitted record schema** — mirror
`04-cpt-sft/dataset/fresh/releases/sft_train.jsonl`'s field set, which is:

```
id, category, task_type, complexity, compiler_pass, test_pass,
manually_reviewed, generator, generator_model_id, gate_class, variant_idx,
generation_date, source_prompt_version, context_bundle_version,
validator_version, dataset_version, run_tag, seed_id, seed_tier, messages,
origin
```

with, for this phase:

- `category = "conversion"`, `task_type = "python_to_jac_function"`
- `origin = "jac-data-gen:7c25aff3110f526eec59e0123ffe6c0c152cce91:<path>"`
- `gate_class = "compile_only"` — honest, and the reason is worth stating in
  the field rather than inflating it: this corpus ships **no pinned expected
  outputs**, so no behavioral assertion is available. Do not label these
  `behavioral`. (`04-cpt-sft`'s own `seed_pool.jsonl` bug — 813 seeds tagged
  `behavioral` with an empty `expected_output`, making every gate a silent
  no-op — is exactly this failure mode; task18 report §4.)
- `generator = "opus-api"`, `generator_model_id` = the dispatched agent's
  model id, recorded once per batch.

**Collect-phase gate:** the assistant turn must compile (`jac run` exit 0);
the instruction must be non-empty, must not quote the answer code verbatim,
and must not be a bare restatement of the function name. Log every rejection.

**Gate to leave Stage 1:** `dataset/sft_train.jsonl` exists, its id set is
disjoint from `nitin_holdout.jsonl`, and the accept/reject accounting is
written.

## Stage 2 — DPO-pair authoring (buggy-variant, Fable)

Correctness axis only (`spec.md` §3.1). `chosen` = the file's Jac as-is;
`rejected` = the same function with one subtly-introduced logic bug.

| Script | Phase | Does |
|---|---|---|
| `prepare_dpo_batch.jac` | prepare | samples from `dataset/candidate_pool.jsonl` (excluding holdout ids), writes `dataset/batches/dpo/pending_batch.jsonl` |
| `collect_dpo_batch.jac` | collect | dual-gates, emits `dataset/dpo_train.jsonl` + `dataset/dpo_valid.jsonl`, routes failures to `dataset/rejected/dpo/` |

- **Model: `fable`.** Buggy-variant authoring is precision work — a bug that
  doesn't actually change behavior, or one that breaks the parse, produces a
  poisoned pair either way. This matches the project's Fable tier
  (`04-cpt-sft/docs/spec.md` §4.1: `gen_debug`, `gen_dpo`).
- Bug types to request, one per pair: off-by-one, flipped comparison
  operator, swapped branch bodies, mutated default argument, inverted
  accumulator, dropped edge-case guard.

**Dual gate at collect, both checks mandatory:**

1. `chosen` compiles (`jac run` exit 0).
2. `rejected` **also** compiles (`jac run` exit 0). A rejected side that
   fails to parse teaches "prefer code that compiles", not "prefer code that
   is correct" — reject the pair, log it.

Because this corpus has no pinned expected outputs, "the bug actually changes
behavior" cannot be mechanically proven the way `gen_debug` proved it. State
that limitation in the release notes rather than implying a behavioral check
happened. Where the collect script can cheaply construct an input and diff
stdout between the two variants, do so and record it — but do not claim
coverage it doesn't have.

**Output shape:** `prompt` / `chosen` / `rejected`, matching
`04-cpt-sft/sft_fresh_probe/dataset/dpo/{train,valid}.jsonl`. `mlx_lm_lora`
is pointed at the *directory* and reads `train.jsonl` + `valid.jsonl` from
it, so the final layout must be a directory with both files.

**Gate to leave Stage 2:** both DPO files exist in the expected directory
shape, every pair dual-gated, drop accounting written.

## Stage 3 — SFT training ×2 arms

### 3.0 Set up the probe directories

> **DONE.** Both probe dirs are scaffolded and path-verified. What is below is
> the as-built record, not a to-do. Nothing has been trained.

Copied from the fresh probe and retargeted **only** on data paths, adapter
paths, and results paths. Hyperparameters, iters, seed, LoRA geometry, and
every watchdog / OOM ladder / stall detector / dry-run gate / `--verify-layers`
gate / `pgrep` collision preflight are byte-identical — that is what makes this
a replication.

```
model-experiments/06-nitin-ds-sft/
  dataset/
    sft/{train,valid}.jsonl     <- ONE shared copy, both arms (see "shared dataset" below)
    dpo/{train,valid}.jsonl     <- ONE shared copy, both arms
  scripts/
    prep_training_dirs.sh       <- release JSONL -> train/valid split (see 3.0b)
  stock_probe/
    configs/sft.yaml            <- fresh sft.yaml; ONLY data + adapter_path changed
    configs/dpo_lora.yaml       <- byte-identical
    dpo_fixed_train.py          <- byte-identical (run_dpo_nofuse.sh's DRIVER)
    run_sft.sh                  <- fresh run_sft.sh, retargeted
    run_dpo_nofuse.sh           <- fresh run_dpo_nofuse.sh, retargeted  (NOT run_dpo.sh)
    eval_sft_sweep.sh           <- fresh eval_sft_sweep.sh, retargeted
    eval_dpo_nofuse.sh          <- fresh eval_dpo_nofuse.sh, retargeted
    adapters/  results/
  spectrum_probe/
    configs/dpo_lora.yaml       <- byte-identical
    dpo_fixed_train.py          <- byte-identical; dpo_spectrum_train.py imports it
                                   from PROBE_DIR (its own parent), so this copy
                                   is REQUIRED, not incidental
    spectrum/
      spectrum_lora_layers.py   <- byte-identical
      dpo_spectrum_train.py     <- byte-identical
      adapter_config_fix.py     <- byte-identical
      configs/sft_spectrum.yaml <- ONLY data + adapter_path changed
      configs/spectrum_layers.json  <- BYTE-IDENTICAL (sha256-verified against the
                                       fresh probe). SNR scan NOT re-run.
    run_sft_spectrum.sh         <- fresh run_sft_spectrum.sh, retargeted
    run_dpo_spectrum.sh         <- fresh run_dpo_spectrum.sh, retargeted
    eval_sft_spectrum.sh        <- fresh eval_sft_spectrum.sh, retargeted
    eval_dpo_spectrum.sh        <- fresh eval_dpo_spectrum.sh, retargeted
    adapters/  results/
```

**Deviation from this file's original sketch: ONE shared `dataset/`, not a copy
per arm.** The sketch above originally put `dataset/sft` and `dataset/dpo`
inside each probe dir. Both arms train on *identical* data by design — that is
the entire point of the replication — so two copies can only drift, and the
project's data is single-copy by convention. `data:` in both arms' SFT yamls
and `--data` in both DPO runners point at
`model-experiments/06-nitin-ds-sft/dataset/{sft,dpo}`.

**Adapter names as built** (`04-cpt-sft`'s `-fresh` suffix becomes `-nitin`):

| Arm | SFT adapter | DPO adapter | DPO-best adapter |
|---|---|---|---|
| stock | `adapters/sft-on-nitin` | `adapters/dpo-on-sft-nitin-nofuse` | `…-nofuse-best` |
| spectrum | `adapters/sft-on-nitin-spectrum` | `adapters/dpo-on-sft-nitin-spectrum` | `…-spectrum-best` |

**Verified during scaffolding:**

- `cd "$(cd "$(dirname "$0")/../../.." && pwd)"` resolves to the repo root from
  `stock_probe/`, `spectrum_probe/`, and `scripts/`. Checked empirically, not
  assumed.
- The only surviving `04-cpt-sft` references in the copied scripts are the two
  intended ones: the functional harness
  `04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac` (spec.md §5.3 item 2 —
  point at it, do not copy it) and the holdout-(a) default (below).
- `bash -n` clean on all 9 shell scripts; `py_compile` clean on all 5 Python files.
- `spectrum_layers.json` sha256 matches the fresh probe byte-for-byte:
  `[0, 22, 23, 27, 30, 34, 36, 37, 38, 39, 41, 42, 43, 44, 45, 47]`.

**`HOLDOUT` and `RDIR` are now env-overridable** (the one behavioural addition,
needed for Stage 5.2's second holdout — see §5.2 for the exact invocations):

- Every script defaults `HOLDOUT` to holdout (a),
  `04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl`. Unset it and you get the
  replication path.
- The four **eval** scripts also default `RDIR` overridably, plus a `mkdir -p`
  so a redirected results dir gets created.
- The two **DPO eval** scripts gained a `TRAIN_RDIR` (defaulting to the arm's
  own training results dir). `.dpo_progress_steps` / `.best_step` are read from
  `TRAIN_RDIR`, never from a redirected `RDIR` — reading them from a redirected
  `RDIR` silently falls back to 250/250 and then skips the DPO-best cell
  entirely via the `BEST_STEP != FINAL_STEP` test.

**Open item flagged for the orchestrator, not decided here:** the two DPO
*runners* also default `HOLDOUT` to holdout (a), which means the in-loop subset
eval that selects the DPO-**best** snapshot selects on holdout (a). That is
byte-identical to what `04-cpt-sft`'s fresh arm did, so the cross-phase
DPO-best comparison stays apples-to-apples — but it does mean holdout (b)'s
DPO-best cell is scored on a snapshot chosen by holdout (a). Deliberate, and
the conservative direction; say so in the report. Related: the SFT-baseline
lookup inside both DPO runners filters on `total == 855`, so pointing a DPO
*training* run at the Nitin holdout would silently drop the collapse gate to
its 30% absolute floor. Both runners carry an inline comment saying so.

### 3.0b Build the train/valid split dirs

`scripts/collect.jac` emits **single release files** — `dataset/sft_train.jsonl`
and `dataset/dpo_train.jsonl`. `mlx_lm` / `mlx_lm_lora` resolve `data:` to a
**directory** containing `train.jsonl` + `valid.jsonl`. Bridge them with:

```
model-experiments/06-nitin-ds-sft/scripts/prep_training_dirs.sh
```

- **Ratio: 85/15, seed 42** — matching `04-cpt-sft`'s fresh arm exactly
  (8100/1428 SFT and 654/115 DPO both compute to 85.0/15.0). Overridable via
  `TRAIN_FRAC` / `SPLIT_SEED`, but don't: the ratio is part of the recipe.
- Deterministic (sorts, then seeded-shuffles) — same release file gives the
  same split every time. Verified by re-running and comparing sha256.
- Drops and reports rows that are not valid JSON or that lack `messages` (SFT) /
  `prompt`+`chosen`+`rejected` (DPO). No silent caps.
- Refuses to overwrite an existing non-empty split unless `FORCE=1`.
- Skips cleanly when a release file doesn't exist yet, so it is safe to run
  while Stage 1/2 are still filling batches.

These `valid.jsonl` files are `mlx_lm_lora`'s **training-time** validation
splits, not scored eval sets (spec.md §4.2, CONTEXT_BRIEF.md §4.5 item 4). Every
reported number comes from the eval scripts against holdout (a) or (b).

**Run this before Stage 3.2** — `run_sft.sh` and `run_sft_spectrum.sh` both
preflight-check for `dataset/sft/{train,valid}.jsonl` and refuse to start
without them.

**Three specific corrections carried from `spec.md` §5.3 — read before
copying:**

1. `spectrum/configs/dpo_spectrum.yaml` **does not exist** in the fresh
   probe. DPO-spectrum hyperparameters live as env-var defaults inside
   `run_dpo_spectrum.sh` (`DPO_ITERS=250`, `DPO_LR=1e-6`, `DPO_BETA=0.1`,
   `DPO_MAXLEN=512`) plus `configs/dpo_lora.yaml`. Copying the runner carries
   them.
2. The functional eval harness has exactly one copy in the repo:
   `model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac`.
   `sft_fresh_probe/jacgen/` holds only `make_dashboard.jac`. It is
   env-var-driven — **do not copy it, point at it**.
3. The stock DPO runner is `run_dpo_nofuse.sh`. `run_dpo.sh` uses the
   `mlx_lm.fuse` path whose own results file reads 12% (107/855) — the
   collapsed run. Copying `run_dpo.sh` would reproduce a known-broken recipe.

Paths to edit in each copied `.sh` / `.yaml`:

- `sft.yaml` / `sft_spectrum.yaml`: `data:` → the arm's `dataset/sft`,
  `adapter_path:` → the arm's adapter dir. Leave `model: "models/qwen-q4"`,
  `iters: 8200`, `num_layers: 16`, `batch_size: 1`,
  `learning_rate: 2.0e-5`, the cosine schedule, `max_seq_length: 3072`,
  `save_every: 820`, `steps_per_eval: 500`, `seed: 42`,
  `mask_prompt: true`, `grad_checkpoint: true` **exactly as they are**.
- Every `.sh`: `SPEC_DIR`, `CFG`, `LAYERS`, `ADAPTER`, `RDIR`, `SFT_ADAPTER`,
  `DPO_ADAPTER`, `BEST_ADAPTER`, `HOLDOUT`, and the `for f in ...` preflight
  file list. Leave `BASE_MODEL="models/qwen-q4"` alone.
- The `cd "$(cd "$(dirname "$0")/../../.." && pwd)"` repo-root hop assumes a
  two-deep script location. `06-nitin-ds-sft/<arm>_probe/` is the same depth
  as `04-cpt-sft/sft_fresh_probe/`, so it stays correct — **verify it
  anyway** by echoing `$PWD` on the first run.

### 3.1 Preflight (every training launch, both arms)

```
pgrep -f "jac start"; pgrep -f mlx_lm
```

Both must return nothing. A resident JMS server or stray `mlx_lm` process
collides with training memory and OOMs this 48GB box regardless of what the
script does — the runners refuse to start if either is found, which is the
gate working, not a bug to bypass. This repo's `jms/` server was mid-testing
recently and may still have an instance up; kill it first.

### 3.2 Stock arm SFT

```
CONFIRM_FULL_RUN=1 model-experiments/06-nitin-ds-sft/stock_probe/run_sft.sh
```

Without `CONFIRM_FULL_RUN=1` the script performs its self-test/dry-run and
exits before the multi-hour run — that is the intended safety gate. Outputs:
`stock_probe/adapters/sft-on-nitin/`,
`stock_probe/results/sft/{train.log,metrics_functional.jsonl}`.

### 3.3 Spectrum arm SFT

```
CONFIRM_FULL_RUN=1 model-experiments/06-nitin-ds-sft/spectrum_probe/run_sft_spectrum.sh
```

The `--verify-layers` preflight loads `models/qwen-q4` twice and must print
`VERIFY: PASS` with **281.838M** trainable parameters. Training is gated on
it. A different count means the selection changed *capacity* rather than
*placement*, which invalidates the comparison — do not `SKIP_VERIFY=1` past
it. Outputs: `spectrum_probe/adapters/sft-on-nitin-spectrum/`,
`spectrum_probe/results/sft-spectrum/`.

**Sequential only.** One training process at a time on this box. Stock SFT
must finish before spectrum SFT starts.

### 3.4 Live monitoring (runs alongside 3.2/3.3)

`run_sft_spectrum.sh`'s watchdog already calls
`model-experiments/01-sft-dpo/sft_dpo/jacgen/plot_metrics.jac` every
`EVAL_EVERY` seconds with `JAC_TRAIN_LOG` / `JAC_PLOT_DIR` set. Add a
lightweight watcher (`scripts/watch_plots.sh`) that tails each arm's
`train.log` plus `metrics_functional.jsonl` and regenerates PNGs into

```
model-experiments/06-nitin-ds-sft/<arm>_probe/results/<stage>/plots/
```

whenever new data lands. The orchestrating session publishes these as an
Artifact dashboard; subagents only keep them current and in that exact
location.

## Stage 4 — DPO training ×2 arms

Each arm's DPO seeds from **its own** SFT adapter via
`--resume-adapter-file`. Never fuse first (`spec.md` §5.3 item 3).

```
CONFIRM_FULL_RUN=1 model-experiments/06-nitin-ds-sft/stock_probe/run_dpo_nofuse.sh
CONFIRM_FULL_RUN=1 model-experiments/06-nitin-ds-sft/spectrum_probe/run_dpo_spectrum.sh
```

Order: stock SFT → spectrum SFT → stock DPO → spectrum DPO. Sequential
throughout.

`run_dpo_spectrum.sh` carries three gates of its own — keep all three:
`--verify-patches` (seconds, no model load: both monkey-patches live, all
three call sites rebound), `--verify-layers` (minutes: the real DPO
conversion path converts the picks, trainable == 281.838M, and every SFT
adapter key finds a home — `mlx_lm_lora/train.py` loads with `strict=False`
and will not tell you otherwise), and the per-snapshot
`adapter_config.json` rewrite before any in-loop scoring.

**Known hazard — DPO OOM.** DPO holds both policy and reference model in
memory. The `04-cpt-sft` cptv2 arm hit macOS's GPU wired-memory ceiling at
`DPO_MAXLEN=512` and had to drop to 384 (comparison report §3.2). The fresh
arm ran clean at 512, and this phase is fresh-based, so **start at 512** — it
is the value the baseline numbers were produced at. If it OOMs, the runner's
own shrink ladder handles it; if you end up at 384, that is a recipe
deviation from the baseline and must be reported in the comparison, not
footnoted.

Each arm produces both a final adapter and a `-best` adapter (best snapshot
by the in-loop subset gate), which is what makes the DPO-best and DPO-final
rows distinct.

## Stage 5 — Eval: 2 arms × 2 holdouts × 3 stages

Twelve cells. All scoring goes through the single functional harness,
`model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac`,
driven by env vars — the same harness that produced every number in
`04-cpt-sft/RESULTS.md`. Using anything else forfeits cross-phase
comparability, which is the whole point of holdout (a).

### 5.1 Holdout (a) — the shared 855

`HOLDOUT="model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl"`
— 1,428 rows, 855 code-graded. **Reuse unchanged. Do not regenerate.**

```
                 model-experiments/06-nitin-ds-sft/stock_probe/eval_sft_sweep.sh
                 model-experiments/06-nitin-ds-sft/spectrum_probe/eval_sft_spectrum.sh
                 model-experiments/06-nitin-ds-sft/stock_probe/eval_dpo_nofuse.sh
                 model-experiments/06-nitin-ds-sft/spectrum_probe/eval_dpo_spectrum.sh
```

Each writes `base.txt` / `final.txt` / `final_best.txt` / `final_last.txt`
plus `metrics_functional.jsonl` into its arm's results dir. The `final*.txt`
`OVERALL runs: NN% (x/855)` line is the cell value; the per-slice lines above
it are the `spec.md` §7.2 per-category breakdown.

### 5.2 Holdout (b) — the Nitin holdout

Same four scripts, with `HOLDOUT` and `RDIR` overridden (both are env-var
defaults as of §3.0) so the (a) numbers are not overwritten. The harness is
holdout-agnostic; the holdout file just needs the same record shape
(`messages` + `gate_class` + the fields the harness reads).

```
H=model-experiments/06-nitin-ds-sft/dataset/nitin_holdout.jsonl
S=model-experiments/06-nitin-ds-sft/stock_probe
P=model-experiments/06-nitin-ds-sft/spectrum_probe

HOLDOUT=$H RDIR=$S/results/sft-nitinholdout          $S/eval_sft_sweep.sh
HOLDOUT=$H RDIR=$P/results/sft-spectrum-nitinholdout $P/eval_sft_spectrum.sh
HOLDOUT=$H RDIR=$S/results/dpo-nofuse-nitinholdout   $S/eval_dpo_nofuse.sh
HOLDOUT=$H RDIR=$P/results/dpo-spectrum-nitinholdout $P/eval_dpo_spectrum.sh
```

`TRAIN_RDIR` is deliberately **not** overridden — the two DPO eval scripts read
`.dpo_progress_steps` / `.best_step` from it, and those live in the training
results dir regardless of which holdout is being scored.

### 5.3 Non-negotiable per-eval checks

- `adapter_config_fix.py` runs **before** any scoring on the spectrum arm,
  and the key assertion passes: **256 keys** (16 per block × 16 blocks).
  Skipping it makes `load_adapters` rebuild LoRA on blocks 32–47 from the
  adapter's own `num_layers: 16`, and `load_weights(strict=False)` silently
  drops blocks 0/22/23/27/30 — the arm gets scored as a partially-loaded
  model, with no error.
- Confirm no all-zero adapter weights before eval, not after a confusing
  number.
- Score the base model once (`JAC_EVAL_ADAPTER=""`) per holdout as the floor.

## Stage 6 — Comparison report

Write to `docs/reports/2026-08-nitin-vs-jacgen2-spectrum-comparison.md`,
following `04-cpt-sft/docs/reports/2026-08-spectrum-vs-stock-comparison.md`'s
structure: headline table → per-stage detail → incidents → honest reading →
artifacts.

Contents, per `spec.md` §7:

1. **The 12-cell matrix**, both holdouts, all three stages, both arms.
2. **Arm-vs-arm paired McNemar** per column (6 comparisons) → research
   question 1. Decision rule: ≥ 4 of 6 significant wins, no significant
   losses.
3. **Cross-phase paired McNemar** of each holdout-(a) cell against its
   `04-cpt-sft` fresh-arm counterpart (597/639/597/634/531/622, all of 855)
   → research question 2. Same ≥ 4-of-6 rule.
4. **Per-slice breakdown on holdout (a)** — 322 of the 855 graded rows are
   `conversion`, this corpus's own task shape. A win concentrated there is a
   specialization result, not a general one, and must be labeled that way.
5. **Power statement for holdout (b)**, computed from its real size. If it
   lands under ~300 code-graded rows, say plainly that it is underpowered and
   let holdout (a) carry the decision.
6. **Named confounds**, stated in the headline rather than buried: training
   set sizes differ from the baseline; single seed per arm, so deltas are
   tested against sampling noise only and not training-seed noise; DPO
   `chosen` correctness is assumed from a compile gate, not proven
   behaviorally; if `DPO_MAXLEN` deviated from 512, that too.
7. **Incidents**, with the same forensic detail as the prior report — what
   was observed, what was ruled out and how, what the real cause was, what
   the fix was.

## Quick reference — failure modes already paid for

| Symptom | Cause | Source |
|---|---|---|
| DPO collapses to ~2–12% pass | `mlx_lm.fuse` on an int4 model re-quantizes and discards the SFT LoRA delta | comparison report §3, `run_dpo_nofuse.sh` header |
| Spectrum adapter loads but scores like a partial model | `adapter_config.json` not rewritten; `load_weights(strict=False)` drops out-of-slice blocks silently | `eval_sft_spectrum.sh` header |
| All LoRA weights read back exactly zero | `mx.load()` + in-place overwrite with no `mx.eval()` between — truncates the mmap being lazily read | comparison report §3.1 (cptv2 arm only; not on this phase's path) |
| DPO OOM at 37GB+ free RAM | macOS GPU wired-memory ceiling, not system RAM; policy + reference both resident | comparison report §3.2 |
| Behavioral gate always passes | `gate_class="behavioral"` with an empty `expected_output` — a truthiness check on `""` | task18 report §4 |
| Cross-directory Jac import fails at runtime but passes `jac check` | relative import with no known parent package — inline the shared logic | task18 report §4 |
| Script runs as a side effect of being imported | `with entry { }` instead of `with entry:__main__ { }` | task18 report §4 |
