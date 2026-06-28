# Published RL ladder results

Raw, published results from the SFT/GRPO ladder (the rest of `results/` is gitignored).

- `rl_ladder.jsonl` — primary holdout (gb+lib, file-disjoint). 30 cells: base / sft /
  sft_grpo / sft_grpo_tuned / raw_grpo × rungs 1,3,5,10,20,all × {qwen3coder, jac-qwen3coder}.
- `rl_ladder_sg.jsonl` — step-7 sg-inclusive holdout (sg idiom held out file-disjoint).
- `RL_LADDER_RESULTS.md` — rendered tables + conclusion (snapshot of docs/rl/02-results.md).

Each `.jsonl` row = one eval, tagged `r<N>/<model>/<cond>/<gen|mem>`. Regenerate the
tables with `jac run rl/show_ladder.jac` (or `rl/record_results.jac` for the doc).

**Headline:** neither SFT nor GRPO (even tuned) moves greedy holdout pass@1 at
30B/LoRA/48GB; only fresh-qwen3coder SFT on the harder sg holdout cracks it faintly
(sg 0→1/5). pass@k mean + train-recall move; GRPO ≈ SFT everywhere. See docs/rl/strat.md.
