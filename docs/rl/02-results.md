# RL Ladder — Results

*Auto-recorded 2026-06-28 01:06 by `rl/record_results.jac` from `results/rl_ladder.jsonl`. Re-run after each ladder stage. Live version also renders in the Studio RL section.*

**Method:** SFT-first ladder (train-N = 1,3,5,10,20,all) × 2 models × conditions {base, SFT(gold), SFT+GRPO, raw-GRPO control}. Holdout is file-disjoint + decontaminated; headline = exact-stdout pass@1 with Wilson 95% CI + sampled pass@k.

**Reward (GRPO):** tiered monotone — exact=1.0 > runs-wrong≤0.80 > compiles≤0.35 > none≤0.15, with a dense body-sim term in every tier (breaks the σ=0 trap).

## Ladder cells

| model | rung | cond | gen pass@1 | gen 95%CI | gen pass@k | mem recall | gen near | osim |
|---|---|---|---|---|---|---|---|---|
| qwen3coder | 1 | base | 26.67% | 11.0-52.0% | 26.67% | 0.0% | 26.67% | 0.2808 |
| qwen3coder | 1 | sft | 26.67% | 11.0-52.0% | 33.33% | 100.0% | 26.67% | 0.2667 |

_gen=holdout (generalization), mem=train recall (overfit gauge). Read pass@k + CI, not bare pass@1 (Yue 2504.13837). A rung 'plateaus' only when consecutive CIs overlap._

See [strat.md](strat.md) for hypotheses, [references.md](references.md) for the literature, and [RL_WEEKEND_RESULTS.md](RL_WEEKEND_RESULTS.md) for the prior GRPO archive.
