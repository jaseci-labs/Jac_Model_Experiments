# 05-cpt-sft-grpo — docs index

Status: **docs-only phase.** No harness code, no dataset, no training exists yet.
`adapters/`, `dataset/`, `results/`, `resultspub/` are empty scaffolds (`.gitkeep` only).

Phase goal: run the **GRPO stage that Attempt03's original four-stage architecture
specified and no phase ever executed** — `base → +CPT → +CPT+SFT/DPO →
+CPT+SFT/DPO+GRPO`, eval at all four checkpoints
(`03-cpt-only/docs/cpt-1/design.md`). `04-cpt-sft` covered base / SFT / DPO for both
arms (fresh vs CPT-v2) and found CPT-v2's real base-stage advantage (+37.0pp) is
statistically absorbed by SFT (+2.8pp, z≈1.28, p≈0.20) — yet survives **structurally**
in the SFT adapter's q_proj LoRA geometry (elevated rank-1 singular value, stable rank
2.7–4.3 vs the fresh-base SFT adapter's 4.4–6.0, against ~16 for an untrained
reference). 05 asks whether that surviving fingerprint translates into a *behavioral*
difference once RL trains on top — the one thing SFT alone could not show — and, in the
same run, whether the `02-rl-grpo` finding "GRPO ≡ SFT everywhere" was a property of
GRPO at this scale or an artifact of that phase's harness (exact-stdout grading,
`this_is_jac`-only corpus).

## Prerequisite reading (in this order)

1. [`../../04-cpt-sft/RESULTS.md`](../../04-cpt-sft/RESULTS.md) — the consolidated
   two-arm result 05 builds directly on.
2. [`../../04-cpt-sft/docs/reports/2026-07-cpt-vs-fresh-comparison.md`](../../04-cpt-sft/docs/reports/2026-07-cpt-vs-fresh-comparison.md)
   — especially **§3** (the `mlx_lm.fuse` root cause, which dictates 05's
   lineage-preservation rule) and **§4** (the q_proj SVD probe, which is 05's whole
   motivation).
3. [`../../02-rl-grpo/docs/rl/strat.md`](../../02-rl-grpo/docs/rl/strat.md) +
   [`01-design.md`](../../02-rl-grpo/docs/rl/01-design.md) — the ladder convention,
   the carried scars, and the harness 05 extends.
4. [`../../02-rl-grpo/RL_FINDINGS.md`](../../02-rl-grpo/RL_FINDINGS.md) —
   **authoritative** corrected RL results (the pre-correction verdicts inside
   `docs/rl/` are superseded there).

## Layout

| File | Contents |
|---|---|
| [`strat.md`](strat.md) | The *why*: reset, anchors, the two research questions written as falsifiable hypotheses (H1/H1b/H2/H3, no verdicts — nothing has run), and the carried scars that must not be re-broken. |
| [`spec.md`](spec.md) | Umbrella design of record: purpose, the 04 findings in full numeric detail, the 4-line × 6-rung training matrix, lineage-preservation mechanics (`--resume-adapter-file`, never `mlx_lm.fuse`), eval design, directory layout, out-of-scope list. The most detailed doc here. |
| [`workflow.md`](workflow.md) | Phased runbook: corpus mining → holdout construction → 24-run training matrix → eval → write-up, with per-phase inputs/outputs and checklists. |
| [`dataset/spec.md`](dataset/spec.md) | The multi-source RL corpus in depth: the 16 repos beyond `this_is_jac`, the HOLE-marker driver convention, Type-B AST-equivalence grading (what it compares, how partial credit works, how it differs from exact-stdout and from the old dense reward), decontamination, licensing/provenance. |
| [`dataset/workflow.md`](dataset/workflow.md) | Mining-pipeline runbook: reuse the 17-repo pinned corpus → mine/author HOLE drivers per repo → family-interleave + file-disjoint splits → decontam → corpus stats + validation. |

## Reading order

1. `strat.md` — why 05 exists and what would falsify it.
2. `spec.md` — the design of record; every decision and its provenance.
3. `dataset/spec.md` — the corpus and the new grader (the bulk of the new design work).
4. `dataset/workflow.md` — how that corpus actually gets built.
5. `workflow.md` — how the 24 runs and the eval execute once the corpus exists.

## Where the code lives

**Not here.** Harness code stays in
[`../../02-rl-grpo/rl/`](../../02-rl-grpo/rl/) and is extended in place, not duplicated
into 05 — `build_tasks.jac` (HOLE-marker mining), `build_rl_splits.jac`
(family-interleave + file-disjoint holdout), `reward_logic.jac` (tiered reward + the
`unwrap_unit` scar), `pick_rung.jac` (ladder slicing), `eval_rl.jac` (holdout scoring),
`run_grpo.sh` (the `mlx_lm_lora` GRPO launcher). 05 owns the **data, adapters, results
and docs**; 02 owns the machinery.

## Related, outside this phase

- `03-cpt-only/adapters/cpt-v2/` — the CPT-v2 LoRA checkpoint (rejected on its own
  gates, `03-cpt-only/docs/cpt-2/results.md`) that both `cptv2-*` lines resume from.
- `03-cpt-only/dataset/cpt/manifest.json` — the pinned 17-repo code corpus (+ the
  docs-only 18th repo) that 05's task mining re-uses; SHAs recorded there.
- `04-cpt-sft/lora_svd_qproj.py` — the q_proj SVD probe whose output is the structural
  claim 05 tests behaviorally.
