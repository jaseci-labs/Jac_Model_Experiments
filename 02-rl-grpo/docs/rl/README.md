# 02-rl-grpo/docs/rl

> **⚠️ CORRECTION (2026-07): the conclusions below predate the extractor-bug fix.**
> The "RL is not the lever / SFT doesn't move greedy" verdict was measured on a
> broken eval that undercounted ~3.5–4×. The corrected numbers (root
> [`02-rl-grpo/RL_FINDINGS.md`](../../RL_FINDINGS.md), authoritative): **SFT lifts greedy
> 38.9%→61.1% (rung-20 peak); GRPO ≈ SFT (valid null); deployable = SFT +
> best-of-k with the Jac compiler as verifier (~78–82%).** These docs remain as
> the design/runbook record; read results from 02-rl-grpo/RL_FINDINGS.md only.

Slow, ground-up redesign of the Jac RL workflow. Read in order:

1. [00-overview.md](00-overview.md) — goal, north star, the two models, principles.
2. [strat.md](strat.md) — the *why*: anchors, research questions, hypotheses, scope discipline.
3. [01-design.md](01-design.md) — task formats, holdout, training ladder, reward, eval.
4. [workflow.md](workflow.md) — the *how*: end-to-end runbook for one ladder run.
5. [references.md](references.md) — literature on whether GRPO/RLVR moves a Qwen-class base (it mostly doesn't, at our scale).
6. [RL_WEEKEND_RESULTS.md](RL_WEEKEND_RESULTS.md) — archive: what the first GRPO run found and why it failed.

(The old `02-results.md` living table was invalidated by the extractor bug and removed; raw corrected data lives in [`raw/`](raw/) and `02-rl-grpo/results/`.)

**Pre-correction status (2026-06-28, superseded — see banner):** ladder run complete (32 cells, 2 holdouts); neither SFT nor GRPO appeared to move greedy holdout generalization at 30B/LoRA/48GB. See [strat.md](strat.md) CONCLUSION.

**Harness:** built and scaffolded (`02-rl-grpo/rl/`). 56 hole-fill tasks (`build_tasks.jac`),
dense v2 reward + `unwrap_unit` scar (`reward_logic.jac`), fixed stratified holdout
(`build_rl_splits.jac`), rung picker (`pick_rung.jac`), and the one-command ladder
driver (`run_ladder.jac`, dry by default). Self-check: `jac run 02-rl-grpo/rl/test_ladder.jac`.
Status: RUN COMPLETE (on the 66-task corpus). Full ladder + tuned-GRPO arm + sg-generalization run all done and recorded (docs + live Studio RL graph). Corpus since squeezed to **84 tasks** (`02-rl-grpo/rl/drivers/`, near the this_is_jac ceiling) — available for a higher-power re-run; the recorded results are from the 66-task split.
