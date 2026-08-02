# 05-cpt-sft-grpo — Design Spec

Status: **approved, pre-implementation (docs-only phase).**
Date: 2026-08-02.
Source of record: `docs/superpowers/specs/2026-08-02-05-cpt-sft-grpo-design.md` (approved
via `superpowers:brainstorming`, 6 sequential clarifying questions: harness choice,
folder naming, GRPO conditions, rung scope, eval metric, corpus sources, compute
budget). This file is that spec expanded; where the two disagree, the committed spec
wins.

This is the umbrella. It does **not** restate the corpus design or the grader mechanics
(`dataset/spec.md`) or the mining run order (`dataset/workflow.md`) — it covers what is
shared: purpose, the prior findings this phase is built on, the training matrix,
lineage preservation, eval design, directory layout, and what is explicitly out of
scope.

---

## 1. Purpose

Close the loop on Attempt03's original locked architecture
(`03-cpt-only/docs/cpt-1/design.md`):

    base → +CPT → +CPT+SFT/DPO → +CPT+SFT/DPO+GRPO,  eval at all 4 checkpoints

`04-cpt-sft` executed the first three checkpoints for both arms (fresh base vs CPT-v2
base). The fourth — GRPO on top of the SFT/DPO policy — has never been run in any phase.
05 runs it, for both arms, plus two no-SFT cold controls.

Two things ride on that run, and they are not the same question:

1. **The structural-vs-behavioral question** (04's open thread). CPT-v2's fingerprint
   survives SFT in the weights but not in the pass rate. RL is the first stage where a
   differently-shaped policy has a different amount of room to move, so it is the first
   place that fingerprint can show up behaviorally — or be shown inert.
2. **The harness question** (02's open thread). `02-rl-grpo`'s "GRPO ≡ SFT everywhere"
   was confirmed three times on one corpus with one grader. 05 changes both at once,
   deliberately, because those are precisely the two suspected artifacts.

---

## 2. Why now — the exact prior findings

Numbers below are quoted from the artifacts, not paraphrased. Every one of them
constrains a decision later in this spec.

### 2.1 `04-cpt-sft` Phase 1–2: the behavioral convergence

Identical SFT/DPO recipe, byte-identical dataset (MD5-verified) and 855-row
code-graded holdout, one variable (CPT-v2 adapter present or not):

| Stage | fresh (no CPT) | CPT-v2 | Δ | Significance |
|---|---|---|---|---|
| base (no SFT) | 10.5% (90/855) | 47.3% (404/855) | **+37.0pp** | z≈16.75 — overwhelming |
| +SFT (final, 8200 iters) | 69.8% (597/855) | 72.6% (621/855) | +2.8pp | z≈1.28, p≈0.20 — **not significant** |

Neither arm's SFT run had a better stopping point than its own final checkpoint: fresh
peaked 72% at step 4100 and *tied* at 8200; cptv2 peaked 79% at step 7380 and *tied* at
8200 (subset=100, code_gen-only curve). No sweet-spot correction was available to either
arm.

### 2.2 `04-cpt-sft` Phase 3: the `mlx_lm.fuse` root cause — the single most load-bearing prior finding for this spec

The original DPO stage collapsed both arms to a 12.0–13.5% floor: immediate (already at
step 20), total, uniform across all 13 checkpoints from step 20 to 250, identical across
two arms, and immune to β and learning-rate changes. That shape does not look like a hard
optimization problem; it looks like the policy was broken before DPO touched it. It was:

| generation source | output on a held-out Jac-component prompt |
|---|---|
| `models/qwen-q4` + SFT adapter, unfused | correct Jac (`def:pub IngredientRow(...) { … }`) |
| `mlx_lm.fuse`'d SFT checkpoint, no adapter | plain React/JSX — indistinguishable from raw base |
| `models/qwen-q4`, no adapter, no fuse | plain React/JSX — **near-identical to the fused checkpoint** |

`mlx_lm.fuse` dequantizes the int4 base, adds the LoRA delta, and re-quantizes; the
re-quantization rounds the fine-grained SFT delta away almost entirely (~15% of packed
4-bit elements changed bit-pattern — not a literal no-op, but functionally close to one).
Every DPO run before that discovery had been training on top of an effectively un-SFT'd
base.

Corrected results after removing the fuse step and seeding via `--resume-adapter-file`
against raw `models/qwen-q4`:

| Arm | SFT baseline | DPO best | DPO final (step 250) |
|---|---|---|---|
| fresh | 69.8% (597/855) | 69.8% (597/855, step 20) | 62.1% (531/855) |
| cptv2 | 72.6% (621/855) | 71.7% (613/855, step 40) | 64.9% (555/855) |

DPO at that recipe (β=0.1, lr=1e-6, sigmoid loss, 654 pairs) caps out at SFT parity and
regresses ~7.7pp in both arms if run to the full budget. Higher β was tested (β=0.3,
60 iters, fresh arm) and ruled out as a lever.

**Consequence for 05:** the GRPO stage must never fuse. Every line seeds from an adapter
file against the raw quantized base. This is not a preference; it is the difference
between measuring GRPO and measuring re-quantization noise. See §5.

### 2.3 `04-cpt-sft` Phase 4: the q_proj structural fingerprint — 05's actual motivation

An adapter-only (un-merged) LoRA SVD probe on `self_attn.q_proj`, layers 32–47,
applying mlx-lm's real effective update `scale · A @ B` (`04-cpt-sft/lora_svd_qproj.py`;
an earlier orphaned analysis omitting `lora_parameters.scale` produced a materially
different spectrum and is superseded). Stable rank = `sum(s)² / sum(s²)`, out of a
rank-16 budget:

| layer | stable rank: CPT-v2 (no SFT) | Base+SFT (fresh) | CPT+SFT (cptv2) | rank-1 magnitude: CPT-v2 | fresh | cptv2 |
|---|---|---|---|---|---|---|
| 32 | 3.06 | 4.91 | 2.96 | 1.14 | 0.66 | 1.37 |
| 38 | 3.35 | 5.18 | 3.25 | 0.98 | 0.53 | 1.11 |
| 44 | 4.11 | 6.04 | 4.29 | 0.87 | 0.47 | 0.93 |
| 47 | 2.70 | 4.42 | 2.83 | 1.10 | 0.53 | 1.13 |

A synthetic untrained-init reference (n=50 mean of random rank-16 draws at mlx-lm's own
A-init scale) sits at essentially full stable rank ~16 with a flat spectrum (0.43–0.52
across all 16 ranks). Two patterns hold across all 8 sampled layers:

1. The fresh-base SFT adapter's stable rank is consistently **higher** (~4.4–6.0) than
   either CPT-v2-involving adapter (~2.7–4.3) — its update is less concentrated in a
   single direction.
2. The CPT-v2-base SFT adapter tracks the **standalone CPT-v2 adapter**, not the
   fresh-base SFT adapter, on both stable rank and rank-1 magnitude (layer 47: 2.70 vs
   2.83, within noise; the fresh adapter sits at 4.42 with roughly half the rank-1
   magnitude).

Reading: SFT reaches the same pass rate from two differently-shaped q_proj updates, one
of which visibly inherits CPT-v2's concentration/magnitude signature. The convergence in
§2.1 is behavioral, not mechanistic. **05 tests whether that mechanistic difference has
any behavioral consequence at all once RL trains on top.** Not checked in 04, and out of
scope here: whether the pattern holds for the other projection types (o/k/v_proj, MLP
gates, MoE projections).

### 2.4 `02-rl-grpo`: what "GRPO ≡ SFT everywhere" actually rests on

Authoritative corrected results: `02-rl-grpo/RL_FINDINGS.md` (the verdicts inside
`02-rl-grpo/docs/rl/` predate the extractor-bug fix and are superseded there).

- SFT lifts greedy pass 38.9% → 61.1% (rung-20 peak); SFT+GRPO at rung-all is flat
  against SFT alone; the raw-base GRPO control equals base **exactly**.
- The tuned-GRPO arm (500 iterations, 10× learning rate) is identical to the untuned
  one — "GRPO just needed more training" is closed.
- The σ=0 explanation is **refuted**: the dense similarity term produced real
  within-group variance (σ=0.09–0.21) and GRPO still moved nothing. The null is not a
  cold-start artifact.
- Deployable answer on that corpus: SFT + best-of-k with the Jac compiler as verifier
  (~78–82%).

What that evidence does **not** cover, and 05 changes:

| Suspected artifact | `02-rl-grpo` | `05-cpt-sft-grpo` |
|---|---|---|
| Grading | byte-exact stdout only; a structurally correct body scoring 0 is invisible | Type-B AST-equivalence (α-normalized), stdout kept as a second pass route |
| Corpus | `this_is_jac` only — 116 drivers on disk, 84 with a `this_is_jac/` origin across 12 source files, 32 authored/synthetic (`source: "unknown"`); one file (`littlex/social_graph.jac`) supplies 32 tasks | 17-repo pinned corpus (`this_is_jac` + 16), per `dataset/spec.md` |
| Holdout | on-disk splits are 18 holdout / 62 trainpool / 4 valid (built when the corpus was 84 tasks; `tasks.jsonl` has since grown to 116 and the splits are stale) | rebuilt across the wider source set, target and composition per `dataset/spec.md` |
| Starting policy | rung-SFT LoRAs trained on the rung's own hole-fill tasks | full 7-category SFT policies at ~70–73% functional pass (04's arms) |

---

## 3. Scope decision: harness

Extends `model-experiments/02-rl-grpo/rl/` — **reused, extended in place, not rebuilt
and not duplicated into 05.** Three real upgrades, all of which Attempt03's original RL
redesign called for and the RL thread closed before building:

1. **Multi-source corpus.** Task mining from the other 16 repos in CPT-v1's
   already-vetted, decontaminated 17-repo code corpus
   (`03-cpt-only/dataset/cpt/manifest.json`, SHAs pinned there), on top of the existing
   `this_is_jac` pool. Same driver / HOLE-marker mining convention as
   `build_tasks.jac`. Full design: `dataset/spec.md` §1–§3.
2. **Type-B AST-equivalence grading**, replacing exact-stdout, in **both** the GRPO
   reward and the final eval — partial credit for structurally-equivalent-but-textually-
   different completions, and a pass route for tasks that cannot be graded on stdout at
   all. Full design: `dataset/spec.md` §4.
3. **Holdout rebuild.** Reuse `build_rl_splits.jac`'s family-interleave + file-disjoint
   logic, generalized across the wider multi-repo source set, so the old
   single-file-dominance concentration problem does not simply reappear at repo
   granularity. Full design: `dataset/spec.md` §5, `dataset/workflow.md` §4.

Files that change, all under `02-rl-grpo/rl/`: `build_tasks.jac` (multi-repo mining),
`build_rl_splits.jac` (repo × family interleave), `reward_logic.jac` (AST tier + AST
similarity term), `eval_rl.jac` (AST-equivalence headline metric), `run_grpo.sh`
(lineage flags, §5.3). Writing them is **out of scope for this phase** (§8).

---

## 4. Scope decision: the training matrix

4 GRPO training lines × 6 rungs = **24 training runs.** No compute/time ceiling (user
confirmed) — run to completion like prior full ladders.

### 4.1 The four lines

| line | base model | resumed adapter | LoRA history | tests |
|---|---|---|---|---|
| `fresh-SFT-warm` | `models/qwen-q4` | `04-cpt-sft/sft_fresh_probe/adapters/sft-on-fresh/adapters.safetensors` (final ckpt, 8200 iters) | SFT → GRPO | CPT-vs-fresh at warm start (H1) |
| `fresh-cold` | `models/qwen-q4` | *none* | GRPO only | cold-start GRPO, no CPT, no SFT (H3) |
| `cptv2-SFT-warm` | `models/qwen-q4` | `04-cpt-sft/sft_cptv2_probe/adapters/sft-on-cptv2/adapters.safetensors` (final ckpt, 8200 iters) | CPT-v2 → SFT → GRPO, one continuous lineage | CPT-vs-fresh at warm start (H1) |
| `cptv2-cold` | `models/qwen-q4` | `03-cpt-only/adapters/cpt-v2/adapters.safetensors` | CPT-v2 → GRPO | does CPT alone (skipping SFT) change GRPO's floor? (H1b) |

### 4.2 The rungs

`1, 3, 5, 10, 20, all` — the same ladder convention as `02-rl-grpo`
(`01-design.md` §3), sliced by `pick_rung.jac` as the front N of a stable, interleaved
`trainpool.jsonl`, so **rung N's task set is a strict superset of rung N−1's** and the
only variable between rungs is task count. The holdout is reserved once, before any rung
runs, and is untouched by rung selection.

Rung 1 keeps its `02-rl-grpo` meaning as a plumbing/memorize check: with one training
task, an unmoved holdout is expected, and the diagnostic of interest is whether the
model's score on *that task* moves at all. It is not evidence about generalization.

### 4.3 Lineage preservation — mechanics

This is the section that 04 §3.2 paid for; treat it as a hard requirement, not guidance.

**Rule: never `mlx_lm.fuse`.** Do not bake any adapter into base weights at any point in
this phase — not before GRPO, not before eval, not for convenience. Fusing into an int4
base rounds the delta away (§2.2).

**Mechanism: `--resume-adapter-file`.** Each line trains against the raw, quantized,
untouched `models/qwen-q4` and *seeds its LoRA A/B matrices* from the named adapter file.
This is the same pattern that fixed 04's DPO
(`04-cpt-sft/sft_fresh_probe/run_dpo_nofuse.sh:141-143`,
`sft_cptv2_probe/run_dpo_v4_nofuse.sh`) and the same pattern CPT-v2's own 12-leg run used
to chain legs (`03-cpt-only/cpt_train/run_cpt_leg.py:74-76` →
`model.load_weights(resume_adapter_file, strict=False)`).

**Why the cptv2 lines are one lineage, not two adapters.** Read
`04-cpt-sft/sft_cptv2_probe/configs/sft.yaml`: its `model` is `models/qwen-q4` and its
`resume_adapter_file` is `03-cpt-only/adapters/cpt-v2/adapters.safetensors`. The
SFT run therefore *continued writing into CPT-v2's own A/B matrices* — there is no
separate "CPT adapter" and "SFT adapter" to stack. `sft-on-cptv2/adapters.safetensors`
already **is** CPT-v2 + SFT. GRPO must continue that same lineage by resuming that one
file; composing two adapters would be a different (and untested) model.

**Shape compatibility — the trap to check before launching anything.** The adapters
being resumed were written by 04's SFT config:

| parameter | value (both arms, `configs/sft.yaml`) |
|---|---|
| `fine_tune_type` | `lora` |
| `num_layers` | **16** |
| `lora_parameters.rank` | 16 |
| `lora_parameters.scale` | 2.0 |
| `lora_parameters.dropout` | 0.05 |
| `max_seq_length` | 3072 |

`02-rl-grpo/rl/run_grpo.sh`'s default is `GRPO_LAYERS=8`, with `MAX_SEQ=1280` and
`MAX_COMPLETION=256`, tuned to fit a 30B-A3B q4 in 48GB at ~38GB peak (its own comment
records that `group6/comp512` OOMs Metal at ~iteration 2). **A GRPO run at 8 layers
cannot faithfully continue a 16-layer adapter.** Required: run GRPO at
`--num-layers 16` with rank 16 / scale 2.0 to match, and re-fit the memory budget by
moving `GROUP_SIZE` / `MAX_COMPLETION` / `MAX_SEQ` instead — verified by a dry run
before the matrix launches (`workflow.md` Phase 4). The two cold lines have the same
constraint for a different reason: `fresh-cold` has no adapter to match, but it must use
the *same* LoRA geometry as the warm lines or the four lines are not comparable;
`cptv2-cold` must match CPT-v2's own adapter geometry
(`03-cpt-only/adapters/cpt-v2/adapter_config.json` — read it at implementation time,
do not assume it equals 04's SFT config).

### 4.4 GRPO hyperparameters

Held identical across all 24 runs; the only variables are (line, rung). Starting point =
`run_grpo.sh`'s current defaults, with the layer count corrected per §4.3 and the
remainder re-fit for memory:

| knob | `run_grpo.sh` default | 05 |
|---|---|---|
| `--iters` | 200 | 200 (pending the Phase-4 dry run; the tuned-GRPO arm at 500 iters/10× lr changed nothing in 02, so a longer default is not indicated) |
| `--learning-rate` | 1e-6 | 1e-6 |
| `--beta` (KL) | 0.04 | 0.04 |
| `--group-size` | 4 | TBD, memory-fit at 16 layers |
| `--max-completion-length` | 256 | TBD, memory-fit at 16 layers |
| `--max-seq-length` | 1280 | TBD — must cover the longest mined template + completion; multi-repo templates are larger than `this_is_jac`'s |
| `--temperature` | 1.0 | 1.0 |
| `--num-layers` | 8 | **16** (§4.3) |
| reference model | unset (frozen base; one weight set in RAM) | unchanged — this is what makes LoRA-GRPO fit 48GB |

Any knob marked TBD is measured in the Phase-4 dry run and then **frozen for all 24
runs**; a knob that varies mid-matrix invalidates cross-rung comparison.

---

## 5. Eval design

**Metric:** Type-B AST-equivalence pass rate on the new RL holdout, defined in
`dataset/spec.md` §4. One headline number per measurement point, plus the standing
diagnostics (`02-rl-grpo/docs/rl/01-design.md` §5 convention): graded reward score,
near-pass rate, AST similarity distribution, idiom density (descriptive only, never a
reward term).

**Measurement points — 24 + 4:**

| # | point | what it is |
|---|---|---|
| 24 | each line × each rung, at that run's final checkpoint | the matrix |
| 2 | `models/qwen-q4` bare; `models/qwen-q4` + `cpt-v2` adapter | base references, no training — the spec's required floor |
| 2 | `models/qwen-q4` + `sft-on-fresh`; `models/qwen-q4` + `sft-on-cptv2` | **rung-0 warm anchors** (see judgment call below) |

The two warm anchors are an addition to the committed spec, which names only the two
base references. They are required by the hypotheses: H2 asks whether GRPO moves a line
*above its own starting point*, and for the warm lines that starting point is the SFT
adapter, not the bare base. Cost is two eval passes, no training. Without them, a warm
line's GRPO delta is unmeasurable and only the cross-line comparison survives.

**Both eval reads, per `02-rl-grpo` convention:**

- **gen** — the fixed holdout, comparable across every rung and line. The headline.
- **mem** — re-eval on the rung's own training tasks. Overfit check, not a
  generalization measure; at rung 1 it is the plumbing check.

**No dual-eval against 04's 855-row functional holdout.** Explicitly excluded by the
committed spec: this experiment measures what the new reward and corpus actually
optimize. Reporting a second, differently-shaped metric alongside invites selecting
whichever moved.

**Statistics.** Every arm answers the same holdout items, so deltas are computed
**paired** over items (McNemar / paired bootstrap), which roughly halves the detectable
effect versus comparing marginal rates — the same rule `04-cpt-sft/docs/workflow.md` §6
pre-registers. Report Wilson intervals on every raw rate; holdouts in this project have
historically been small (n=11–32 in `02-rl-grpo`) and the intervals are wide. Decision
thresholds are stated in `strat.md`'s hypotheses and are pre-registered — do not
re-choose them after seeing the curves.

---

## 6. Directory layout

```
model-experiments/05-cpt-sft-grpo/
  docs/
    README.md          # index + prerequisite reading
    strat.md           # why + research questions + falsifiable hypotheses + carried scars
    spec.md            # this file — umbrella design of record
    workflow.md        # phased runbook: mining -> holdout -> 24 runs -> eval -> write-up
    dataset/
      spec.md          # multi-source corpus + Type-B AST-equivalence grading, in depth
      workflow.md      # mining pipeline runbook: 17 repos -> drivers -> decontam -> splits
  adapters/            # empty — 24 GRPO LoRA checkpoints land here, 4 lines x 6 rungs
  dataset/             # empty — new multi-source RL corpus + splits land here
  results/             # empty — per-line/rung eval output (jsonl + images)
  resultspub/          # empty — published graphs/summaries (matches 02-rl-grpo convention)
```

Naming convention for the 24 adapters (fixed now so results tooling can assume it):
`adapters/<line>-r<rung>/` — e.g. `adapters/cptv2-SFT-warm-r10/`, `adapters/fresh-cold-rall/`.
Per-run eval rows append to `results/ladder.jsonl` with keys `{line, rung, checkpoint,
ast_pass, stdout_pass, graded, near_pass, idiom, n}` (mirroring
`02-rl-grpo/results/rl_ladder.jsonl`'s row-per-cell convention).

Harness code stays in `02-rl-grpo/rl/` (§3). Nothing executable belongs under
`05-cpt-sft-grpo/`.

**Gitignore policy:** follow the sibling phases — generated dataset artifacts and LoRA
checkpoints are large and regenerable from the harness; docs, manifests, results
`.jsonl` and published summaries are source. Confirm against the repo's existing
`.gitignore` entries for `02-rl-grpo/adapters/` and `04-cpt-sft/dataset/` before the
first real run, rather than inventing a new rule here.

---

## 7. Risks and how they are handled

| Risk | Why it matters here | Mitigation |
|---|---|---|
| AST grader is the shared reward+eval surface | Exactly the shape of `02-rl-grpo`'s Era-2 extractor bug, which faked a flat ladder for three weeks by capping every cell's ceiling | Self-test before any training: every mined task's gold body must score 1.0 against itself under the grader, and a deliberately-mutated body must not (`workflow.md` Phase 3) |
| Layer/rank mismatch on resume | Silently trains a geometry the adapter was not written in; the four lines stop being comparable | §4.3 shape check + a dry run per line before the matrix |
| Memory at 16 layers | `run_grpo.sh`'s 38GB peak was measured at 8 layers; 16 may not fit at the same group size | Phase-4 dry run re-fits group size / completion length, then freezes them for all 24 runs |
| Multi-repo corpus is thin per repo | 4 of the 17 repos contributed ≤12 gated files in the CPT build; `inr-codelabs` contributed 0 | Report real per-repo yield; do not force a quota (`dataset/spec.md` §6) |
| Repo-level concentration replaces file-level concentration | The old bug was one file at 47% of the holdout; the new failure mode is one repo | Interleave and cap at **repo × family**, not family alone (`dataset/spec.md` §5) |
| Single run per cell | 04's own caveat: a single holdout measurement per arm cannot separate a small real effect from training stochasticity | Stated as a named limitation (§8); seed repeats are out of scope for this phase |

---

## 8. Out of scope / deferred

- **Writing the harness extensions.** Multi-source mining, the Type-B grader, and the
  holdout rebuild are designed here and implemented in a later phase, in
  `02-rl-grpo/rl/`.
- **Launching any training.** No run in this phase.
- **Dual-eval against 04's 855-row functional holdout.**
- **Whole-file regeneration as a task type** (`02-rl-grpo/docs/rl/01-design.md` §1
  Type B, the task track). 05 borrows Type-B's *grading* idea only; the whole-file
  ladder remains its own later track.
- **A DPO stage inside 05.** 04 established that DPO at its tested recipe caps at SFT
  parity; adding it here would add a variable the matrix cannot resolve.
- **Repeated-seed runs per cell.** 04's three-arm protocol used 2–3 seed repeats per arm
  to establish an empirical training-noise σ; 05's committed matrix is 24 single runs.
  Consequence, stated plainly rather than hidden: per-cell deltas are not separable from
  LoRA training stochasticity, and the *shape of the curve across six rungs* — not any
  single cell — is the readable result. If a headline claim ends up resting on one cell,
  that cell gets repeated before the claim ships.
- **Extending the SVD probe to other projection types** (o/k/v_proj, MLP gates, MoE) —
  flagged as a follow-up in 04 Phase 4, unrelated to this phase's behavioral question.
- **Any change to `04-cpt-sft`'s or `02-rl-grpo`'s recorded results.** 05 adds a stage;
  it does not re-open prior numbers.

---

## 9. Provenance

- Approved design: `docs/superpowers/specs/2026-08-02-05-cpt-sft-grpo-design.md`
  (2026-08-02).
- Prior results this builds on: `04-cpt-sft/RESULTS.md`,
  `04-cpt-sft/docs/reports/2026-07-cpt-vs-fresh-comparison.md` (Phases 1–4),
  `02-rl-grpo/RL_FINDINGS.md` (authoritative), `02-rl-grpo/docs/rl/{strat,01-design,
  workflow}.md`, `03-cpt-only/docs/cpt-1/design.md` (the original 4-stage architecture),
  `03-cpt-only/docs/cpt-2/{results,analysis}.md` (CPT-v2's own rejection).
- Configs read directly for this spec: `04-cpt-sft/sft_{fresh,cptv2}_probe/configs/sft.yaml`,
  `04-cpt-sft/sft_fresh_probe/run_dpo_nofuse.sh`, `02-rl-grpo/rl/run_grpo.sh`,
  `03-cpt-only/dataset/cpt/manifest.json`.
