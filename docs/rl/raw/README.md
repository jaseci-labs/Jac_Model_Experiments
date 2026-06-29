# Raw RL experiment record (preserved)

The full experimental record — durable on GitHub (the live `results/*.jsonl` are gitignored).

- `rl_ladder.jsonl` — primary 66-task-corpus ladder (gb+lib holdout), 58 eval rows / 30 cells
- `rl_ladder_sg.jsonl` — step-7 sg-inclusive holdout run
- `rl_ladder_v84.jsonl` — 84-task higher-power re-run
- `tasks_manifest.json` — the 84-task eval set (id, source file, expected stdout) — reconstructable ground truth

Each ladder row = one eval, tagged `r<N>/<model>/<cond>/<gen|mem>`. Drivers (the hand-authored
source) are in `rl/drivers/` (committed). Rebuild tasks: `jac run rl/build_tasks.jac`.
Render: `jac run rl/show_ladder.jac` / `python3 rl/make_graphs.py`.
