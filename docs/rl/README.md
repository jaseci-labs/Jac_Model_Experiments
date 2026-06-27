# docs/rl

Slow, ground-up redesign of the Jac RL workflow. Read in order:

1. [00-overview.md](00-overview.md) — goal, north star, the two models, principles.
2. [strat.md](strat.md) — the *why*: anchors, research questions, hypotheses, scope discipline.
3. [01-design.md](01-design.md) — task formats, holdout, training ladder, reward, eval.
4. [workflow.md](workflow.md) — the *how*: end-to-end runbook for one ladder run.
5. [02-results.md](02-results.md) — living results table, filled per rung.
6. [references.md](references.md) — literature on whether GRPO/RLVR moves a Qwen-class base (it mostly doesn't, at our scale).
7. [RL_WEEKEND_RESULTS.md](RL_WEEKEND_RESULTS.md) — archive: what the first GRPO run found and why it failed.

**One-line status:** restarting from scratch on a measurable ladder; SFT is the proven lever, GRPO re-tested as a controlled challenger; dataset is `this_is_jac` only; Qwen3.6 removed.

**Harness:** built and scaffolded (`rl/`). 51 hole-fill tasks (`build_tasks.jac`),
dense v2 reward + `unwrap_unit` scar (`reward_logic.jac`), fixed stratified holdout
(`build_rl_splits.jac`), rung picker (`pick_rung.jac`), and the one-command ladder
driver (`run_ladder.jac`, dry by default). Self-check: `jac run rl/test_ladder.jac`.
Pending: more hole-fill tasks (~10–15 minable), then launch the ladder.
