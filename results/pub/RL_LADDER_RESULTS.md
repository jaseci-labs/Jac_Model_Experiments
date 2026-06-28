# RL Ladder — Results

*Auto-recorded 2026-06-28 14:36 by `rl/record_results.jac` from `results/rl_ladder.jsonl`. Re-run after each ladder stage. Live version also renders in the Studio RL section.*

**Method:** SFT-first ladder (train-N = 1,3,5,10,20,all) × 2 models × conditions {base, SFT(gold), SFT+GRPO, raw-GRPO control}. Holdout is file-disjoint + decontaminated; headline = exact-stdout pass@1 with Wilson 95% CI + sampled pass@k.

**Reward (GRPO):** tiered monotone — exact=1.0 > runs-wrong≤0.80 > compiles≤0.35 > none≤0.15, with a dense body-sim term in every tier (breaks the σ=0 trap).

## Ladder cells — primary holdout (gb+lib, file-disjoint)

| model | rung | cond | gen pass@1 | gen 95%CI | gen pass@k | mem recall | gen near | osim |
|---|---|---|---|---|---|---|---|---|
| jac-qwen3coder | 1 | base | 26.67% | 11.0-52.0% | 33.33% | 100.0% | 26.67% | 0.2808 |
| jac-qwen3coder | 1 | sft | 26.67% | 11.0-52.0% | 46.67% | 100.0% | 26.67% | 0.2808 |
| jac-qwen3coder | 3 | base | 26.67% | 11.0-52.0% | 40.0% | 33.33% | 26.67% | 0.2808 |
| jac-qwen3coder | 3 | sft | 26.67% | 11.0-52.0% | 46.67% | 33.33% | 26.67% | 0.2808 |
| jac-qwen3coder | 5 | base | 26.67% | 11.0-52.0% | 46.67% | 20.0% | 26.67% | 0.2808 |
| jac-qwen3coder | 5 | sft | 26.67% | 11.0-52.0% | 40.0% | 20.0% | 26.67% | 0.2808 |
| jac-qwen3coder | 10 | base | 26.67% | 11.0-52.0% | 33.33% | 10.0% | 26.67% | 0.2808 |
| jac-qwen3coder | 10 | sft | 26.67% | 11.0-52.0% | 40.0% | 10.0% | 26.67% | 0.2808 |
| jac-qwen3coder | 20 | base | 26.67% | 11.0-52.0% | 33.33% | 5.0% | 26.67% | 0.2808 |
| jac-qwen3coder | 20 | sft | 26.67% | 11.0-52.0% | 33.33% | 5.0% | 26.67% | 0.2808 |
| jac-qwen3coder | all | base | 26.67% | 11.0-52.0% | 46.67% | 13.33% | 26.67% | 0.2808 |
| jac-qwen3coder | all | sft | 26.67% | 11.0-52.0% | 33.33% | 13.33% | 26.67% | 0.2808 |
| jac-qwen3coder | all | sft_grpo | 26.67% | 11.0-52.0% | 40.0% | 13.33% | 26.67% | 0.2808 |
| jac-qwen3coder | all | raw_grpo | 26.67% | 11.0-52.0% | 46.67% | 13.33% | 26.67% | 0.2808 |
| jac-qwen3coder | all | sft_grpo_tuned | 26.67% | 11.0-52.0% | 53.33% | 13.33% | 26.67% | 0.2808 |
| qwen3coder | 1 | base | 26.67% | 11.0-52.0% | 26.67% | 0.0% | 26.67% | 0.2808 |
| qwen3coder | 1 | sft | 26.67% | 11.0-52.0% | 40.0% | 100.0% | 26.67% | 0.2667 |
| qwen3coder | 3 | base | 26.67% | 11.0-52.0% | 40.0% | 0.0% | 26.67% | 0.2808 |
| qwen3coder | 3 | sft | 33.33% | 15.0-58.0% | 40.0% | 33.33% | 33.33% | 0.3333 |
| qwen3coder | 5 | base | 26.67% | 11.0-52.0% | 26.67% | 0.0% | 26.67% | 0.2808 |
| qwen3coder | 5 | sft | 20.0% | 7.0-45.0% | 40.0% | 20.0% | 20.0% | 0.2 |
| qwen3coder | 10 | base | 26.67% | 11.0-52.0% | 26.67% | 0.0% | 26.67% | 0.2808 |
| qwen3coder | 10 | sft | 33.33% | 15.0-58.0% | 40.0% | 10.0% | 33.33% | 0.3333 |
| qwen3coder | 20 | base | 26.67% | 11.0-52.0% | 26.67% | 0.0% | 26.67% | 0.2808 |
| qwen3coder | 20 | sft | 26.67% | 11.0-52.0% | 46.67% | 20.0% | 26.67% | 0.2667 |
| qwen3coder | all | base | 26.67% | 11.0-52.0% | 26.67% | 8.89% | 26.67% | 0.2808 |
| qwen3coder | all | sft | 26.67% | 11.0-52.0% | 46.67% | 15.56% | 26.67% | 0.2667 |
| qwen3coder | all | sft_grpo | 26.67% | 11.0-52.0% | 40.0% | 15.56% | 26.67% | 0.2667 |
| qwen3coder | all | raw_grpo | 26.67% | 11.0-52.0% | 26.67% | 8.89% | 26.67% | 0.2808 |
| qwen3coder | all | sft_grpo_tuned | 26.67% | 11.0-52.0% | 40.0% | 15.56% | 26.67% | 0.2667 |

_gen=holdout (generalization), mem=train recall (overfit gauge). Read pass@k + CI, not bare pass@1 (Yue 2504.13837). A rung 'plateaus' only when consecutive CIs overlap._

### by condition — pass@k reframe (gen / holdout)

| cond | n | mean pass@1 | mean pass@8 | max pass@8 (boundary) |
|---|---|---|---|---|
| base | 12 | 26.7% | 33.9% | 46.7% |
| sft | 12 | 27.2% | 41.1% | 46.7% |
| sft_grpo | 2 | 26.7% | 40.0% | 40.0% |
| sft_grpo_tuned | 2 | 26.7% | 46.7% | 53.3% |
| raw_grpo | 2 | 26.7% | 36.7% | 46.7% |

_Greedy (pass@1) flat across all conditions. pass@8 MEAN lifts under SFT (sampling efficiency), but the BOUNDARY (max pass@8) is unchanged from base — neither SFT nor GRPO expands what's reachable (Yue 2504.13837). GRPO ≈ SFT on every metric._

## Step 7 — sg-inclusive holdout (sg idiom held out file-disjoint)

Re-sourced social_graph into feature buckets (`JAC_RL_SG_BUCKETS=1`) so the sg OSP-walker idiom sits in the holdout. The held-out sg slice is the hard frontier (0/5 for base, both models); fresh qwen3coder SFT at rung-all is the only cell to crack it (sg 0->1/5, overall 14.3%->28.6%).

| model | rung | cond | gen pass@1 | gen 95%CI | gen pass@k | mem recall | gen near | osim |
|---|---|---|---|---|---|---|---|---|
| jac-qwen3coder | 5 | base | 21.43% | 8.0-48.0% | 28.57% | 0.0% | 21.43% | 0.2294 |
| jac-qwen3coder | 5 | sft | 21.43% | 8.0-48.0% | 35.71% | 0.0% | 21.43% | 0.2294 |
| jac-qwen3coder | all | base | 21.43% | 8.0-48.0% | 28.57% | 13.04% | 21.43% | 0.2294 |
| jac-qwen3coder | all | sft | 21.43% | 8.0-48.0% | 35.71% | 13.04% | 21.43% | 0.2294 |
| qwen3coder | 5 | base | 14.29% | 4.0-40.0% | 28.57% | 0.0% | 14.29% | 0.158 |
| qwen3coder | 5 | sft | 14.29% | 4.0-40.0% | 28.57% | 0.0% | 14.29% | 0.1429 |
| qwen3coder | all | base | 14.29% | 4.0-40.0% | 28.57% | 10.87% | 14.29% | 0.158 |
| qwen3coder | all | sft | 28.57% | 12.0-55.0% | 42.86% | 13.04% | 28.57% | 0.2857 |

_gen=holdout (generalization), mem=train recall (overfit gauge). Read pass@k + CI, not bare pass@1 (Yue 2504.13837). A rung 'plateaus' only when consecutive CIs overlap._

### by condition — pass@k reframe (gen / holdout)

| cond | n | mean pass@1 | mean pass@8 | max pass@8 (boundary) |
|---|---|---|---|---|
| base | 4 | 17.9% | 28.6% | 28.6% |
| sft | 4 | 21.4% | 35.7% | 42.9% |

_Greedy (pass@1) flat across all conditions. pass@8 MEAN lifts under SFT (sampling efficiency), but the BOUNDARY (max pass@8) is unchanged from base — neither SFT nor GRPO expands what's reachable (Yue 2504.13837). GRPO ≈ SFT on every metric._

See [strat.md](strat.md) for hypotheses, [references.md](references.md) for the literature, and [RL_WEEKEND_RESULTS.md](RL_WEEKEND_RESULTS.md) for the prior GRPO archive.
