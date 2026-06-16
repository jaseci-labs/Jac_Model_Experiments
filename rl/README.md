# rl — GRPO on Jac with a verifiable reward

Reinforcement-learning stage on top of the SFT+DPO phase (`sft_dpo/`). The
policy is rewarded for producing **compiler-correct, behaviorally-correct,
idiomatic Jac** — the reward is the `jac run` gate itself (no learned reward
model). Tasks are body-completion over the real open-source Jac corpus in
`this_is_jac/`. Runs locally on Apple-Silicon MLX via `mlx-lm-lora` GRPO.

## Pipeline

```
this_is_jac/*.jac
   │  (authored, by hand)
   ▼
rl/drivers/*.jac          deterministic jac-run-able files; the target unit's
   │                      body is wrapped in  # >>>HOLE id=.. instruction=..  /  # <<<HOLE
   ▼ jac run rl/build_tasks.jac
dataset/rl/tasks.jsonl    {prompt, answer} GRPO records   (+ templates/<id>.jac sidecars)
   │ jac run rl/build_rl_splits.jac
dataset/rl/{train,valid,holdout}.jsonl
   │ RL_BASE=<model> ./rl/run_grpo.sh <name>      reward = rl/reward.py:jac_behavioral
   ▼
adapters/<name>-grpo  +  results/<name>/grpo/
   │ JAC_EVAL_MODEL=<base> JAC_EVAL_ADAPTER=adapters/<name>-grpo jac run rl/eval_rl.jac
   ▼
results/<name>/grpo/eval.txt   (run% / behavior-pass% / idiom vs base)
```

## Reward

`rl/reward.py:jac_behavioral(prompts, completions, answer)` — for each sampled
completion: splice it into the task template (`__HOLE__`), `jac run` it (isolated
cwd so the persistent `root` graph never accumulates state across runs), score:

```
0.3*compiles + 0.3*runs + 0.3*output_match + 0.1*idiom_bonus
```

Non-running output earns no idiom credit (can't sneak partial credit). Registered
with `mlx_lm_lora` via `--reward-functions-file rl/reward.py --reward-functions
jac_behavioral`. Tests: `.venv/bin/python -m pytest rl/tests -q`.

## Run order

```bash
jac run rl/build_tasks.jac          # drivers -> dataset/rl/tasks.jsonl + templates/
jac run rl/build_rl_splits.jac      # -> train/valid/holdout.jsonl
RL_BASE=<mlx-model> ./rl/run_grpo.sh <name>
JAC_EVAL_MODEL=<mlx-model> JAC_EVAL_ADAPTER=adapters/<name>-grpo jac run rl/eval_rl.jac
```

## Model lineup (3 sequential runs)

| name | RL base (`RL_BASE`) | notes |
|---|---|---|
| `qwen-coder` | a Q4 of `models/qwen-jac-dpo-fused-q8` | RL on the current SFT+DPO best |
| `qwen3` | `models/qwen-q4` (Qwen3-Coder-30B-A3B) | same base, RL-only ablation |
| `qwen36` | `models/qwen36-q4` (Qwen/Qwen3.6-27B, dense) | dense → slowest; run last, `GROUP_SIZE=4` |

## Env / knobs (`run_grpo.sh`)

`GRPO_ITERS`(300) `GRPO_LR`(1e-6) `GRPO_BETA`(0.04) `GROUP_SIZE`(6)
`MAX_COMPLETION`(512) `MAX_SEQ`(2048) `GRPO_TEMP`(1.0) `GRPO_LAYERS`(8).

## Gotchas

- **jac persistence:** every `jac run` writes a `.jac/` graph in the cwd; a
  persistent `root` accumulates across runs. The reward + eval + build all run
  snippets with `cwd` set to a throwaway temp dir. Keep this if you touch them.
- **Determinism:** `jid()` returns random UUIDs and `_now()` is time-based, so
  drivers must print only deterministic projections (usernames, content, counts,
  likes) — never raw view objects that embed ids/timestamps.
- **Frozen reference:** GRPO leaves `--reference-model-path` unset, so the KL
  reference is the frozen base — only one weight set in RAM (fits 48 GB).
- **jac startup tax:** the reward spawns `jac` per completion (~1–2 s each ×
  group size). Acceptable for LoRA GRPO; optimize to an in-process runner if
  step time hurts.
- **Dense-27B:** `qwen36` is all-active → heaviest rollouts; lower `GROUP_SIZE`,
  run it last.
- **build_tasks.jac** runs fine but only parse-checks (`jac check -p`) — its
  dynamic dict access trips the strict type-checker, same as `eval_probe.jac`.
