# Jac RL — Results (living)

Filled in as each rung runs. Pass = exact stdout (Type A). Empty cells = not yet run. See [01-design.md](01-design.md) for the ladder definition.

## Hole-fill ladder

Holdout = fixed ~15-task pool (same every rung). `mem` = trained-tasks re-eval, `gen` = holdout.

| rung (train N) | model | base | SFT mem | SFT gen | SFT+GRPO gen | raw-GRPO gen |
|---|---|---|---|---|---|---|
| 1 | jac-qwen3coder | | (target 100%) | n/a | | |
| 1 | qwen3coder | | (target 100%) | n/a | | |
| 3 | jac-qwen3coder | | | | | |
| 3 | qwen3coder | | | | | |
| 5 | jac-qwen3coder | | | | | |
| 5 | qwen3coder | | | | | |
| 10 | jac-qwen3coder | | | | | |
| 10 | qwen3coder | | | | | |
| 20 | jac-qwen3coder | | | | | |
| 20 | qwen3coder | | | | | |
| all | jac-qwen3coder | | | | | |
| all | qwen3coder | | | | | |

> Rung 1 `gen` is `n/a` — the memorize rung evaluates the trained task itself (mem column).

## Diagnostics log
Per run, also record: graded score, near-pass (osim≥0.9), avg-osim. Append notes here as the curve develops.

## Sweet spot
_To be read off the curve once the ladder is run: the train-N where fixed-holdout pass plateaus._

## Whole-file track
_Started after the hole-fill ladder is mapped. Table TBD with the AST grader design._
