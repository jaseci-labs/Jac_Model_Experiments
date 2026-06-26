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
   │ (fresh/cold bases only) RL_BASE=<cold> ./rl/run_rft.sh <name>   warm-start
   ▼                                                                  ↳ models/<name>-rft-q4
   │ RL_BASE=<model-or-warm> ./rl/run_grpo.sh <name>   reward = rl/reward_logic.jac (jac_behavioral)
   ▼
adapters/<name>-grpo  +  results/<name>/grpo/
   │ JAC_EVAL_MODEL=<base> JAC_EVAL_ADAPTER=adapters/<name>-grpo jac run rl/eval_rl.jac
   ▼
results/<name>/grpo/eval.txt   (run% / behavior-pass% / idiom vs base)
```

## Reward

Logic lives in **`rl/reward_logic.jac`** (`jac_behavioral`) — for each sampled
completion: splice it into the task template (`__HOLE__`), `jac run` it (isolated
cwd so the persistent `root` graph never accumulates state across runs), score:

```
0.3*compiles + 0.3*runs + 0.3*output_score + 0.1*idiom_bonus
```

- `output_score` — 1.0 on exact stdout match; a near-miss earns `0.5 ·
  difflib ratio` (softer gradient for behaviourally-close code, never rivals exact).
- `idiom_bonus` — weighted: graph-traversal/object-spatial ops (`visit`, `-->`,
  `spawn`, `report`, `here`, `root`, …) count more than plain declarations.
- Non-running output earns no idiom or output credit.
- Identical rollouts in a group are scored once (dedup cache) — fewer `jac` runs.

**Why a `.py` shim exists.** `mlx_lm_lora` loads the reward via
`importlib.spec_from_file_location`, which needs a `.py`. `rl/reward.py` is a
~5-line shim that imports the Jac module (whose `with entry` registers
`jac_behavioral` into the reward registry). All reward logic is Jac; the shim is
just the bridge the trainer's loader requires. Wired via
`--reward-functions-file rl/reward.py --reward-functions jac_behavioral`.

Test (pure Jac, self-checking, exits non-zero on failure):

```bash
jac run rl/build_tasks.jac      # ensure a task set exists
jac run rl/test_reward.jac      # perfect / garbage / wrong-output / batch cases
```

## Warm-start (RFT) — for cold bases only

Fresh, non-jac-trained bases emit mostly non-compiling Jac, so GRPO sees ~all-zero
reward and stalls. `rl/run_rft.sh` bootstraps them: sample from the base, keep
completions that PASS the *same* jac reward (`rft_sample.jac`), LoRA-SFT on them,
fuse → a warmed base GRPO can climb from. Skip it for the already-jac-trained
base. Knobs: `RFT_SAMPLES`(8) `RFT_TEMP`(1.0) `RFT_PASS`(0.9; drop to 0.6 if 0
pass) `RFT_ITERS`(150).

## Run order

```bash
jac run rl/build_tasks.jac          # drivers -> dataset/rl/tasks.jsonl + templates/
jac run rl/build_rl_splits.jac      # -> train/valid/holdout.jsonl

# fresh bases: warm-start first (produces models/<name>-rft-q4)
RL_BASE=models/qwen-q4 ./rl/run_rft.sh qwen3coder
RL_BASE=models/qwen3coder-rft-q4 ./rl/run_grpo.sh qwen3coder

# jac-trained base: GRPO direct (no warm-start)
RL_BASE=models/jac-qwen3coder-q4 ./rl/run_grpo.sh jac-qwen3coder

JAC_EVAL_MODEL=<base> JAC_EVAL_ADAPTER=adapters/<name>-grpo jac run rl/eval_rl.jac
```

## Model lineup (3 sequential runs)

| name | RL base (`RL_BASE`) | warm-start? |
|---|---|---|
| `qwen3coder` | `models/qwen-q4` (Qwen3-Coder-30B-A3B, fresh) | yes → `run_rft.sh` |
| `jac-qwen3coder` | Q4 of `models/qwen-jac-dpo-fused-q8` (SFT+DPO best) | no (already warm) |
| `qwen36` | `models/qwen36-q4` (Qwen/Qwen3.6-27B, dense, fresh) | yes; `GROUP_SIZE=4`, run last |

## Env / knobs (`run_grpo.sh`)

`GRPO_ITERS`(200) `GRPO_LR`(1e-6) `GRPO_BETA`(0.04) `GROUP_SIZE`(4)
`MAX_COMPLETION`(256) `MAX_SEQ`(1280) `GRPO_TEMP`(1.0) `GRPO_LAYERS`(8).
Defaults fit a 30B-A3B q4 on 48GB (peak ~38GB); group6/comp512 OOMs Metal.

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

## Deferred (evaluated, not done — on purpose)

- **In-process jac runner** (replace per-completion subprocess). Rejected: running
  snippets in-process shares jaclang's `root` graph + archetype namespace across
  completions (every driver redefines `node Profile`, …) → silent reward
  corruption. Subprocess isolation is correct; the startup tax is acceptable for
  LoRA GRPO. The dedup cache recovers most of the win safely.
- **Curriculum (easy→hard ordering in the split).** Doesn't work: `mlx_lm_lora`
  GRPO `np.random.permutation`s the train set every epoch, so file order is
  ignored. A real curriculum needs 2-phase training — deferred (low value at ~40
  tasks).
