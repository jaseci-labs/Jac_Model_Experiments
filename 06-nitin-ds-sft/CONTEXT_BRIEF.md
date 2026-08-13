# 06-nitin-ds-sft — master context brief

Read this FIRST, in full, before doing anything else. It exists so every agent
working this experiment shares the same facts instead of re-deriving them —
do not re-discover what's already answered here; do flag if something here
turns out to be wrong.

## 0. What this experiment is

Two research questions, both answered with the same paired two-proportion
z-test methodology already used in `model-experiments/04-cpt-sft/docs/reports/2026-08-spectrum-vs-stock-comparison.md`:

1. **Does Spectrum (SNR/Marchenko-Pastur layer selection) replicate its win
   over stock trailing-16 LoRA targeting on a NEW dataset?**
2. **Is this new dataset "truly better" than the existing jacgen2-derived
   dataset used throughout `04-cpt-sft`?**

## 1. Source data

Cloned from `https://github.com/chess10kp/jac-data-gen`, commit pin: run
`git -C ~/repos/jac-data-gen log -1 --format=%H` and cite it everywhere.
Cloned to `~/repos/jac-data-gen` on FAST internal disk — the project's own
`/Volumes/ExtremePro/...` external drive measured at ~4.3 MB/s this session
(genuine hardware/contention bottleneck, confirmed via raw `dd`, unrelated to
any app bug). Do bulk file operations (compile-checking 7627 files, etc.) on
the internal copy; only write final JSONL/report artifacts to the external
project tree.

**Shape:** 7627 files under `data/jac_outputs/*.jac`. Each file = ONE
Python-transpiled function: a docstring followed by
`def name(params) -> type { ... }`. No metadata, no split, single commit.

**IMPORTANT — honest framing, already verified by sampling:** these are FLAT
FUNCTION conversions (dicts/lists/loops), NOT graph-native OSP Jac (no
node/edge/walker anywhere in the samples checked). This corpus is the same
SHAPE as this project's existing `python_to_jac_function` task type, not
graph-native material. Do NOT frame this as "idiomatic graph-native Jac" —
that would overclaim. Framing decisions below are already made accordingly.

## 2. Framing decisions (already made — don't re-litigate)

**SFT:** reverse-author one NL instruction per selected file (from its
docstring + signature — matches this project's existing `code_gen` /
`python_to_jac_function` reverse-authoring pattern, e.g. see
`model-experiments/01-sft-dpo/sft_dpo/jacgen2/` task prompts), paired with the
file's Jac code as the target completion. Messages-format schema, matching
`model-experiments/04-cpt-sft/dataset/fresh/releases/sft_train.jsonl`'s shape
(`id, category, task_type, ..., messages, origin, ...` — reuse the same
field set where it applies; `category="conversion"`,
`task_type="python_to_jac_function"`, `origin="jac-data-gen:<commit>:<path>"`).

**DPO — use the CORRECTNESS axis, NOT the idiomatic-vs-Python-shaped axis.**
`docs/dpo-plan.md`'s flagship idiomatic axis (chosen=graph-native,
rejected=Python-shaped) doesn't fit this content — these files have no
natural graph-native rewrite (e.g. `expand_comma(value: str)`,
`Distance_modulus_to_distance(dm, absorption)` are pure utility/math
functions with nothing to "graph-ify"). Instead: **chosen** = the file's Jac
code as-is (assumed correct/functional); **rejected** = an LLM-authored
variant with one subtly-introduced bug (off-by-one, wrong operator, swapped
branch, mutated default, etc. — matching `dpo-plan.md` §2.3's
correct-vs-subtly-wrong axis pattern, which this project already builds from
`gen_debug`'s `buggy_variants.jsonl` elsewhere). Both variants must still
compile (`jac run` exit 0) — the bug must be a LOGIC/behavioral divergence
detectable by differing output on some input, not a syntax break. This is a
deliberate scope decision to keep the pipeline honest and tractable, not an
oversight — document it as such if writing docs.

## 3. Base checkpoint — "fresh qwens" (explicit user instruction)

`models/qwen-q4` (registry label "Qwen · BASE") — the SAME base checkpoint
used in `04-cpt-sft`'s "fresh arm" (`sft_fresh_probe/spectrum/configs/sft_spectrum.yaml`
confirms `model: "models/qwen-q4"`). NOT `qwen-cpt-v1`, NOT any existing
SFT/DPO-tuned checkpoint. This keeps the comparison to the existing fresh-arm
numbers clean — only the dataset and (for one arm) the LoRA target-block
selection differ.

## 4. Eval — BOTH holdouts, four result cells per stage (final, corrected design)

Do NOT skip either. Per-stage (SFT, DPO-best, DPO-final), for EACH arm
(stock, spectrum), eval against:

**(a) Existing "mine" holdout** — reuse UNCHANGED, do not regenerate:
- `model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl`
  (1428 rows, 855 code-graded — the SFT-stage functional eval set)
- `model-experiments/04-cpt-sft/sft_fresh_probe/dataset/dpo/valid.jsonl`
  (115 rows — the DPO-stage eval set)
- Eval scripts to copy-adapt (NOT rewrite from scratch):
  `sft_fresh_probe/eval_sft_spectrum.sh`, `eval_dpo_spectrum.sh`, `eval_dpo.sh`,
  `sft_fresh_probe/jacgen/eval_functional.jac` (or the `sft_cptv2_probe`
  copy if the fresh one differs — diff them before picking).
- This is what makes every number DIRECTLY comparable to the existing
  `2026-08-spectrum-vs-stock-comparison.md` fresh-arm table:
  SFT stock 69.8% (597/855), SFT spectrum 74.7% (639/855),
  DPO-best stock 69.8% (597/855), DPO-best spectrum 74.2% (634/855),
  DPO-final stock 62.1% (531/855), DPO-final spectrum 72.7% (622/855).
  These are the numbers "is this dataset better" gets compared against.

**(b) New holdout carved from Nitin's own corpus** — frozen BEFORE any SFT
generation happens, held out of the training pool entirely. Size: pick
something the corpus comfortably supports after cleaning (aim similar order
of magnitude to (a), e.g. several hundred rows) — the corpus-triage step
below determines the real achievable number; do not force a specific count
if the clean pool doesn't support it, report the true number. This tests
in-distribution performance on the new data's own distribution — NOT
comparable to (a)'s prior numbers, but comparable stock-vs-spectrum WITHIN
itself.

**Comparison matrix to build in the final report:** 2 arms × 2 holdouts ×
3 stages (SFT / DPO-best / DPO-final) = up to 12 cells, using the same
two-proportion z-test (paired McNemar where both arms answer identical items,
per `spectrum-workflow.md`'s own calibration note) as every prior comparison
in this project. Cross-reference cell (a) against the existing fresh-arm
numbers above to answer research question 2.

## 4.5 CORRECTIONS to this brief (found by the docs-scaffolding agent, verified)

Four things below were wrong or imprecise as originally written. Full detail
in `docs/spec.md` §5.3/§4.2 and `docs/workflow.md` Stage 3.0 — treat those as
authoritative over the original text in §4-5 below where they conflict:

1. **`spectrum/configs/dpo_spectrum.yaml` does NOT exist.** DPO-spectrum
   hyperparameters are env-var defaults inside `run_dpo_spectrum.sh`
   (`DPO_ITERS=250`, `DPO_LR=1e-6`, `DPO_BETA=0.1`, `DPO_MAXLEN=512`) plus the
   shared `configs/dpo_lora.yaml`.
2. **`sft_fresh_probe/jacgen/eval_functional.jac` does NOT exist.** The only
   copy of that harness in the repo is
   `04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac` — the fresh arm's
   own eval scripts already point at it. Use that one path, no diffing needed.
3. **CRITICAL — the stock DPO runner is `run_dpo_nofuse.sh`, NOT `run_dpo.sh`.**
   `run_dpo.sh`'s own output collapses to 12% (107/855) via a known
   `mlx_lm.fuse` int4-requantization bug. The REAL stock-arm DPO baselines
   this brief cites (597/855 best, 531/855 final) come from
   `results/dpo-nofuse/final_best.txt` / `final_last.txt`, produced by
   `run_dpo_nofuse.sh` + `eval_dpo_nofuse.sh`. **Copy `run_dpo_nofuse.sh` for
   the 06-nitin stock arm, never `run_dpo.sh`.**
4. **`sft_fresh_probe/dataset/dpo/valid.jsonl` (115 rows) is NOT a separate
   scored eval set** — nothing reads it directly for scoring; it's
   `mlx_lm_lora`'s internal validation split during DPO training. ALL
   headline numbers at every stage (SFT, DPO-best, DPO-final) are scored on
   the SAME 855 code-graded rows of `sft_fresh_probe/dataset/sft/valid.jsonl`
   via `eval_dpo_nofuse.sh` / `eval_dpo_spectrum.sh`. There is only ONE eval
   file to reuse for §4(a), not two.

`merge_frozen_keys.py` (the in-place-mx.load-truncation bug from
`2026-07-cpt-vs-fresh-comparison.md` §3.1) only applies to `sft_cptv2_probe/` —
a fresh base has no frozen prior-adapter keys to merge, so neither 06-nitin
arm touches that code path. Not a concern here.

## 4.6 Eval sweep sizing (explicit user instruction, resolves §7's open subset size)

Two distinct eval passes, not one:

- **Interim/sweep checkpoints** (during training, at each `save_every`/segment
  boundary — this is what feeds the live loss+eval-curve graphs from §7):
  eval against a **15% slice of the Nitin holdout** (855 × 0.15 ≈ 128 rows,
  seeded, frozen — carve it once, reuse the same 128 rows at every sweep
  point so the curve is comparable step-to-step). Matches this project's own
  existing convention (`eval_sft_sweep.sh`'s subset-sweep pattern, and the
  Spectrum report §2.2/2.3's `subset=100` checkpoint tables) — full-holdout
  eval at every checkpoint would be far too slow to be "live."
- **Final checkpoints only** (SFT-final, DPO-best, DPO-final — the numbers
  that actually go in the comparison report): full evaluation on **BOTH**
  holdouts at full size — the 855-row Nitin holdout AND the existing
  855-code-graded `04-cpt-sft` holdout (§4a). Do not substitute the 15%
  sweep subset for either of these — headline numbers are always full-size,
  both holdouts, no exceptions.

## 5. Training — copy-adapt the existing script suite, don't rewrite

Everything needed already exists and is battle-tested at
`model-experiments/04-cpt-sft/sft_fresh_probe/`:

- `run_sft.sh` / `run_dpo.sh` — stock arm
- `run_sft_spectrum.sh` / `run_dpo_spectrum.sh` — spectrum arm (wraps
  `mlx_lm.lora` via `spectrum/spectrum_lora_layers.py`, has a
  `--verify-layers` preflight gating on exact trainable-param count —
  keep this gate, it's cheap insurance)
- `eval_sft_spectrum.sh` / `eval_dpo_spectrum.sh` / `eval_dpo.sh` — eval
  runners
- `configs/sft.yaml`, `configs/dpo_lora.yaml` — stock hyperparameters
- `spectrum/configs/sft_spectrum.yaml` (`spectrum/configs/dpo_spectrum.yaml`
  if it exists, check) — spectrum hyperparameters
- **`spectrum/configs/spectrum_layers.json`** — the SNR-selected 16-block
  list. **REUSE VERBATIM, do not re-run `snr_scan.py`/`layer_select.py`** —
  the SNR scan is a property of the base model's WEIGHTS (`qwen-q4`), not the
  training dataset. Since the base checkpoint is identical, the selection
  `[0, 22, 23, 27, 30, 34, 36, 37, 38, 39, 41, 42, 43, 44, 45, 47]` (11/16
  overlap with trailing-16) is still correct here. Re-running it would waste
  hours for an identical result.

Copy the whole `sft_fresh_probe/` layout into
`model-experiments/06-nitin-ds-sft/{stock,spectrum}_probe/` (or a flatter
layout your docs agent decides, as long as it's documented), retargeting only:
data paths (point at the new dataset), adapter output paths, results output
paths. Everything else (hyperparameters, iters=8200, batch_size=1,
learning_rate, lora rank/scale/dropout, seed=42) stays IDENTICAL to the
existing fresh-arm recipe — that's what makes this a clean replication, not a
confounded one.

**Preflight hazard, already known:** these scripts refuse to start if
`pgrep -f "jac start"` or `pgrep -f "mlx_lm"` finds anything running (a
resident model on this 48GB box collides with training memory). Before
kicking off ANY training run, confirm no JMS server / stray mlx process is
alive: `pgrep -f "jac start"; pgrep -f mlx_lm`. Kill them first if found —
this repo's `jms/` server was mid-testing earlier this session and may still
have an instance up.

## 6. Decontamination — reuse existing shingle machinery, don't reinvent

`model-experiments/01-sft-dpo/sft_dpo/jacgen2/decontam_v2.jac` (newer) and
`.../jacgen/decontam.jac` (older, same lineage) already do shingle-based
near-dup detection for this project. Reuse the algorithm (port it into a
standalone script if it's too coupled to the old pipeline's file layout —
don't rebuild shingling logic from scratch). Two things must be checked:

1. **Dedup within the corpus itself** (large Python-mined corpora cluster
   heavily around common utility patterns).
2. **Decontaminate the training pool against BOTH existing holdouts named in
   §4(a)** — `sft_fresh_probe/dataset/sft/valid.jsonl` and
   `sft_fresh_probe/dataset/dpo/valid.jsonl` — plus the older
   `model-experiments/01-sft-dpo/dataset/eval_holdout/{conversion,graph_conversion}.jsonl`
   for good measure. ANY match above threshold gets dropped from training —
   log exactly how many and why, this project's convention is no silent
   caps.

The NEW holdout carved per §4(b) must ALSO be decontaminated against the
training pool (obviously — it has to be held out, not just labeled as such)
and against (a)'s holdouts (avoid a Nitin-holdout item accidentally being a
near-dup of an existing (a)-holdout item, which would make the two "holdout"
results non-independent).

## 7. Live monitoring — explicit user requirement

Loss curves and periodic eval-subset pass-rate curves must be captured AS
each run progresses, not only computed after the fact. `run_sft_spectrum.sh`
et al. already write `train.log` and periodic checkpoints
(`save_every: 820`, `steps_per_eval: 500` in the existing configs) — a
separate lightweight watcher script should tail these and regenerate a PNG
(matplotlib, already available in the project venv) every time new data
lands, saved to `model-experiments/06-nitin-ds-sft/<arm>_probe/results/<stage>/plots/`.
The orchestrating session (not a subagent) will periodically publish these as
an Artifact dashboard — subagents just need to keep the PNGs current and in a
predictable location.

## 8. Directory layout

```
model-experiments/06-nitin-ds-sft/
  CONTEXT_BRIEF.md          <- this file
  docs/
    README.md  spec.md  workflow.md  dataset-structure.md
    reports/corpus-triage-report.md
    reports/<comparison reports as they land>
  dataset/
    candidate_pool.jsonl     <- clean, deduped, decontaminated pool (pre-holdout-split)
    nitin_holdout.jsonl      <- frozen holdout carved per §4(b)
    sft_train.jsonl  dpo_train.jsonl   <- final training releases
  stock_probe/    <- copy-adapted from sft_fresh_probe, stock arm
  spectrum_probe/ <- copy-adapted from sft_fresh_probe, spectrum arm
```

## 9. Sequencing (respect the dependency order)

corpus triage + dedup + decontam + holdout carve
  → SFT-pair authoring (reverse-instruction) + DPO-pair authoring (buggy-variant)
  → SFT training ×2 arms (can run sequentially only — single 48GB box, one
    training process at a time, confirmed by §5's preflight lock)
  → DPO training ×2 arms (each depends on its own arm's SFT adapter)
  → eval ×2 arms × 2 holdouts × (SFT, DPO-best, DPO-final)
  → comparison report (2×2×3 matrix + cross-reference to existing fresh-arm
    numbers)

No LLM API key exists in this environment — all "LLM generation" steps
(reverse-instruction authoring, buggy-variant authoring) go through the same
batch-handoff architecture already used by `jacgen2`: write a
`pending_batch.jsonl`, dispatch a real Claude Code Agent (`model: "opus"` or
`"fable"` matching this project's per-category convention — bulk/volume work
uses Opus, precision/error-prone work uses Fable; reverse-instruction
authoring is bulk → Opus, buggy-variant authoring needs precision → Fable) to
fill `responses_batch.jsonl`, then a collect phase gates+persists. See
`model-experiments/04-cpt-sft/docs/reports/2026-07-task18-full-run-report.md`
§2 for why/how.
