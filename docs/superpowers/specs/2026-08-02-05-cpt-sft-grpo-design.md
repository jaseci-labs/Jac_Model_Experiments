# 05-cpt-sft-grpo — Design Spec

Status: approved, pre-implementation (docs-only phase).
Date: 2026-08-02.

## 1. Purpose

Close the loop on Attempt03's original locked architecture (`03-cpt-only/docs/cpt-1/design.md`):
`base → +CPT → +CPT+SFT/DPO → +CPT+SFT/DPO+GRPO`, eval at all 4 checkpoints. `04-cpt-sft`
covered base/SFT/DPO for both arms (fresh vs CPT-v2) and found CPT-v2's real base-stage
advantage (+37pp) gets statistically absorbed by SFT (+2.8pp, p≈0.20) — but survives
*structurally* in the SFT adapter's q_proj weight geometry (`04-cpt-sft/docs/reports/
2026-07-cpt-vs-fresh-comparison.md` Phase 4). `05-cpt-sft-grpo` runs the missing GRPO stage
for both arms and asks whether that structural fingerprint translates into a behavioral
difference once RL trains on top — something SFT alone couldn't show.

## 2. Research questions (both, per user)

1. **Does CPT change GRPO's ceiling, or just SFT's?** cptv2-arm GRPO vs fresh-arm GRPO,
   compared outside CI. A real, outside-CI gap reopens the CPT thread; convergence (like SFT)
   closes it for good.
2. **Is "GRPO ≡ SFT everywhere" (the `02-rl-grpo` finding, confirmed 3x across two corpora) a
   property of GRPO/model-scale, or an artifact of the old harness** (exact-stdout grading,
   `this_is_jac`-only corpus, ~84-task ceiling)? A move on the new harness (either arm) is the
   headline; CPT-vs-fresh is secondary in that case.

## 3. Scope decision: harness

Extends `model-experiments/02-rl-grpo/rl/` (reused, not rebuilt) with two real upgrades that
Attempt03's original RL redesign called for and the RL thread closed before building:

- **Multi-source corpus.** Adds task-mining from the other 16 repos in CPT-v1's already-vetted,
  decontaminated 17-org code corpus (`03-cpt-only/dataset/cpt/code/raw.jsonl` — jac,
  Agentic-AI, jac-shadcn, jac-load-test, jac-mcp-playground, jasketch, jac-client-playground,
  Algo, llvm-slice, littleX, jaseci-blogs, jaseci-studio, agentic-ai-tutorial,
  the-jac-workshop, inr-codelabs, tree-sitter-jac, jaseci-llmdocs), on top of the existing
  `this_is_jac` pool. Same driver/HOLE-marker mining convention as `build_tasks.jac`.
- **Type-B AST-equivalence grading**, replacing exact-stdout, in both the GRPO reward and the
  final eval — partial credit for structurally-equivalent-but-textually-different completions.
- **Holdout**: reuse `build_rl_splits.jac`'s family-interleave + file-disjoint logic,
  generalized across the wider multi-repo source set, to avoid the old harness's sg=57%
  concentration bug.

No dual-eval against 04's 855-row functional holdout — this experiment measures what the new
reward/corpus actually optimizes.

## 4. Scope decision: training matrix

4 GRPO training lines, lineage-preserving via `--resume-adapter-file` (never `mlx_lm.fuse` —
that silently drops the LoRA delta on re-quantization, root cause of 04's DPO-collapse bug,
`2026-07-cpt-vs-fresh-comparison.md` §3.2):

| line | resumes from | tests |
|---|---|---|
| fresh-SFT-warm | `04-cpt-sft/sft_fresh_probe/adapters/sft-on-fresh` final ckpt | CPT-vs-fresh @ warm start |
| fresh-cold (control) | `models/qwen-q4`, no adapter | cold-start GRPO, no CPT no SFT |
| cptv2-SFT-warm | `04-cpt-sft/sft_cptv2_probe/adapters/sft-on-cptv2` final ckpt | CPT-vs-fresh @ warm start |
| cptv2-cold (control) | `03-cpt-only/adapters/cpt-v2/adapters.safetensors`, no SFT | does CPT alone (skip SFT) change GRPO's floor? |

× 6 rungs (1, 3, 5, 10, 20, all — same ladder convention as `02-rl-grpo`) = **24 training
runs**. No compute/time ceiling (user confirmed) — run to completion like prior full ladders.

Note the cptv2 lines are a **single continuous LoRA lineage** (CPT-v2 → SFT → GRPO all the same
A/B matrices, never fused, never stacked as separate adapters) — confirmed by reading
`sft_cptv2_probe/configs/sft.yaml`'s own `resume_adapter_file`. GRPO must continue that same
lineage, not compose CPT-v2 and SFT as two adapters.

## 5. Eval

New RL holdout only, AST-equivalence pass rate, at every line's final checkpoint per rung (24
points) + base-model reference (both `models/qwen-q4` and the CPT-v2 base, no training).

## 6. Directory layout (this phase: docs + empty scaffold only)

`model-experiments/05-cpt-sft-grpo/`, mirroring the `04-cpt-sft` docs convention
(`README.md`/`spec.md`/`workflow.md` + a nested `dataset/{spec.md,workflow.md}` for the
corpus-mining pipeline specifically, matching `04-cpt-sft/docs/datagen/`) and `02-rl-grpo`'s
`strat.md` (why/research-questions/hypotheses) convention:

```
05-cpt-sft-grpo/
  docs/
    README.md
    strat.md            # why + the 2 research questions + falsifiable hypotheses
    spec.md              # umbrella design spec (this doc, expanded)
    workflow.md           # phased runbook: mining -> holdout -> training matrix -> eval
    dataset/
      spec.md             # multi-source corpus + Type-B AST-equivalence grading, in depth
      workflow.md          # mining pipeline runbook: 16 repos -> drivers -> decontam -> splits
  adapters/               # empty — 24 GRPO LoRA checkpoints land here, 4 lines x 6 rungs
  dataset/                # empty — new multi-source RL corpus + splits land here
  results/                # empty — per-line/rung eval output
  resultspub/             # empty — published graphs/summaries (matches 02-rl-grpo convention)
```

No code, no data, no training in this phase — docs + empty directories only, per explicit user
scope. Harness code stays in `02-rl-grpo/rl/` (extended in place, not duplicated).

## 7. Out of scope / deferred

- Actually writing the harness extensions (multi-source mining, Type-B grader, holdout rebuild).
- Launching any training.
- Dual-eval against 04's functional holdout.
- Whole-file regen task type (still deferred from the original ladder design, unrelated to this
  phase).

## 8. Provenance

Approved via `superpowers:brainstorming` in-session (2026-08-02), 6 sequential clarifying
questions (harness choice, folder naming, GRPO conditions, rung scope, eval metric, corpus
sources, compute budget) — see conversation for full Q&A. Builds on
`04-cpt-sft/RESULTS.md`, `04-cpt-sft/docs/reports/2026-07-cpt-vs-fresh-comparison.md`,
`02-rl-grpo/docs/rl/strat.md`, `03-cpt-only/docs/cpt-1/design.md`.
