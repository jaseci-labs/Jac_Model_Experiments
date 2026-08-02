# rl — GRPO on Jac with a verifiable reward

Reinforcement-learning stage on top of the SFT+DPO phase (`01-sft-dpo/sft_dpo/`). The
policy is rewarded for producing **compiler-correct, behaviorally-correct,
idiomatic Jac** — the reward is the `jac run` gate itself (no learned reward
model). Tasks are body-completion over the real open-source Jac corpus in
`this_is_jac/`. Runs locally on Apple-Silicon MLX via `mlx-lm-lora` GRPO.

## Pipeline

```
this_is_jac/*.jac
   │  (authored, by hand)
   ▼
02-rl-grpo/rl/drivers/*.jac          deterministic jac-run-able files; the target unit's
   │                      body is wrapped in  # >>>HOLE id=.. instruction=..  /  # <<<HOLE
   ▼ jac run 02-rl-grpo/rl/build_tasks.jac
02-rl-grpo/dataset/rl/tasks.jsonl    {prompt, answer} GRPO records   (+ templates/<id>.jac sidecars)
   │ jac run 02-rl-grpo/rl/build_rl_splits.jac      (fixed ~15 holdout, spread across families)
02-rl-grpo/dataset/rl/{holdout,trainpool,train,valid}.jsonl
   │ jac run 02-rl-grpo/rl/run_ladder.jac           THE LADDER: loops rungs × models × conditions
   ▼                                     (dry by default; JAC_LADDER_GO=1 to execute)
02-rl-grpo/results/rl_ladder.jsonl   one row per eval, tagged r<N>/<model>/<cond>/<gen|mem>
```

The ladder driver (`02-rl-grpo/rl/run_ladder.jac`) reuses every primitive below; the manual
single-run path still works if you want one cell by hand:

```
   │ RUNG=<N> jac run 02-rl-grpo/rl/pick_rung.jac           train.jsonl = first N of trainpool (superset-grow)
   │ jac run 02-rl-grpo/rl/build_sft_gold.jac               gold SFT set for those N tasks
   │ RL_BASE=<base> ./rl/run_rft.sh <name>       A) gold-SFT (LoRA + fuse) → models/<name>-rft-q4
   │ RL_BASE=<warm-or-base> ./rl/run_grpo.sh <n> B/C) GRPO   reward = 02-rl-grpo/rl/reward_logic.jac
   ▼
adapters/<name>-grpo  +  02-rl-grpo/results/<name>/grpo/
   │ JAC_EVAL_MODEL=<base> JAC_EVAL_ADAPTER=adapters/<name>-grpo jac run 02-rl-grpo/rl/eval_rl.jac
   ▼
02-rl-grpo/results/<name>/grpo/eval.txt   (run% / pass@1 / pass@k / near-pass / idiom vs base)
```

## Reward

Logic lives in **`02-rl-grpo/rl/reward_logic.jac`** (`jac_behavioral`) — for each sampled
completion: splice it into the task template (`__HOLE__`), `jac run` it (isolated
cwd so the persistent `root` graph never accumulates state across runs), score
(dense v2):

```
0.25*compiles + 0.25*runs + 0.25*output + 0.10*idiom + 0.15*body_sim
```

- `output` — 1.0 on exact stdout match; a near-miss earns `0.5 · difflib ratio`
  (softer gradient for behaviourally-close code, never rivals exact).
- `idiom` — weighted: graph-traversal/object-spatial ops (`visit`, `-->`,
  `spawn`, `report`, `here`, `root`, …) count more than plain declarations; gated
  behind `runs`.
- `body_sim` — `difflib ratio(body, gold refbody)`, scored for **every** completion
  including non-compiling ones. The ONLY term not gated behind `runs`, so a group of
  all-failing rollouts still has within-group variance → non-zero GRPO advantage.
  This is what kills the σ=0 zero-gradient trap (see `02-rl-grpo/docs/rl/strat.md` scar #2).
- Non-running output earns no idiom or output credit.
- Identical rollouts in a group are scored once (dedup cache) — fewer `jac` runs.

**Why a `.py` shim exists.** `mlx_lm_lora` loads the reward via
`importlib.spec_from_file_location`, which needs a `.py`. `02-rl-grpo/rl/reward.py` is a
~5-line shim that imports the Jac module (whose `with entry` registers
`jac_behavioral` into the reward registry). All reward logic is Jac; the shim is
just the bridge the trainer's loader requires. Wired via
`--reward-functions-file 02-rl-grpo/rl/reward.py --reward-functions jac_behavioral`.

Test (pure Jac, self-checking, exits non-zero on failure):

```bash
jac run 02-rl-grpo/rl/build_tasks.jac      # ensure a task set exists
jac run 02-rl-grpo/rl/test_reward.jac      # perfect / garbage / wrong-output / batch cases
```

## Warm-start (RFT) — for cold bases only

Fresh, non-jac-trained bases emit mostly non-compiling Jac, so GRPO sees ~all-zero
reward and stalls. `02-rl-grpo/rl/run_rft.sh` bootstraps them: sample from the base, keep
completions that PASS the *same* jac reward (`rft_sample.jac`), LoRA-SFT on them,
fuse → a warmed base GRPO can climb from. Skip it for the already-jac-trained
base. Knobs: `RFT_SAMPLES`(8) `RFT_TEMP`(1.0) `RFT_PASS`(0.9; drop to 0.6 if 0
pass) `RFT_ITERS`(150).

## The ladder (default path)

`02-rl-grpo/rl/run_ladder.jac` is the whole experiment in one command: for each rung
(train-N ∈ 1,3,5,10,20,all) and each model it runs the three design conditions —
**base**, **A) gold-SFT**, **B) SFT+GRPO**, **C) raw-base GRPO control** — and evals
each on the fixed holdout (gen) and the rung's own train tasks (mem). Rows append to
`02-rl-grpo/results/rl_ladder.jsonl` tagged `r<N>/<model>/<cond>/<gen|mem>`.

```bash
jac run 02-rl-grpo/rl/build_tasks.jac           # drivers -> 02-rl-grpo/dataset/rl/tasks.jsonl + templates/
jac run 02-rl-grpo/rl/build_rl_splits.jac       # fixed ~15 holdout + trainpool (JAC_RL_HOLDOUT_N=15)
jac run 02-rl-grpo/rl/test_ladder.jac           # self-check: disjoint splits, family spread, nested rungs

jac run 02-rl-grpo/rl/run_ladder.jac            # DRY: print the full command plan, run nothing heavy
JAC_LADDER_GO=1 jac run 02-rl-grpo/rl/run_ladder.jac   # execute (sequential; hours per cell)
jac run 02-rl-grpo/rl/show_ladder.jac           # Phase 3: pivot 02-rl-grpo/results/rl_ladder.jsonl -> curve table

# scope it while iterating:
JAC_LADDER_RUNGS=1,3 JAC_LADDER_MODELS=qwen3coder:models/qwen-q4:4 \
  JAC_LADDER_GO=1 jac run 02-rl-grpo/rl/run_ladder.jac
```

### Recommended execution (compute-smart, ~½ the GPU-hours)

Don't run the blind 36-cell grid — ~80% re-tests the weekend's known null.

```bash
# 0. plumbing smoke test — must hit ~100% mem on rung 1 or STOP
JAC_LADDER_CONDITIONS=sft JAC_LADDER_RUNGS=1 \
  JAC_LADDER_MODELS=qwen3coder:models/qwen-q4:4 JAC_LADDER_GO=1 jac run 02-rl-grpo/rl/run_ladder.jac
# 1. cheap SFT curve first (both models, all rungs) — find the elbow
JAC_LADDER_CONDITIONS=base,sft JAC_LADDER_GO=1 jac run 02-rl-grpo/rl/run_ladder.jac
jac run 02-rl-grpo/rl/show_ladder.jac          # read the curve + CIs; pick the elbow rung E
# 2. GRPO only at the elbow (+ a tuned arm), and ONE raw control
JAC_LADDER_CONDITIONS=sft_grpo JAC_LADDER_RUNGS=<E> \
  GRPO_LR=1e-5 GRPO_ITERS=500 JAC_LADDER_GO=1 jac run 02-rl-grpo/rl/run_ladder.jac
JAC_LADDER_CONDITIONS=raw_grpo JAC_LADDER_RUNGS=1 JAC_LADDER_GO=1 jac run 02-rl-grpo/rl/run_ladder.jac
```

### Known limitations (design review, 2026-06-27)

- **Corpus is sg-dominated** (social_graph.jac = 57% of tasks, one file). File-disjoint
  holdout therefore can't put the core OSP-walker idiom in BOTH train and holdout — on
  the 56-task corpus the holdout is sg-light. **Fix: mine to ~120–150 tasks** (more
  this_is_jac files) before trusting the generalization number. `build_rl_splits` WARNs.
- **Power floor:** n≈15–30 holdout, binary exact-stdout → ~CI of ±15pp. Read `show_ladder`'s
  Wilson CI + pass@k, never the bare pass@1; a rung "plateaus" only when CIs overlap.
- **Before launch (runtime, not code):** raise the Metal wired-memory limit —
  `sudo sysctl iogpu.wired_limit_mb=44000` (peaks hit 38.2GB on the ~36GB default cap).
- **Deferred:** `extract_jac`/`unwrap_unit` pick first/last unit; on a multi-unit completion
  they can mis-extract → condition-correlated false fails. Match by the hole's signature if
  you see anomalous base-vs-RL deltas.

## Manual single-run path

```bash
jac run 02-rl-grpo/rl/build_tasks.jac          # drivers -> 02-rl-grpo/dataset/rl/tasks.jsonl + templates/
jac run 02-rl-grpo/rl/build_rl_splits.jac      # -> holdout/trainpool/train/valid.jsonl
RUNG=5 jac run 02-rl-grpo/rl/pick_rung.jac     # optional: train.jsonl = first 5 tasks (else "all")

# fresh bases: warm-start first (produces models/<name>-rft-q4)
RL_BASE=models/qwen-q4 ./rl/run_rft.sh qwen3coder
RL_BASE=models/qwen3coder-rft-q4 ./rl/run_grpo.sh qwen3coder

# jac-trained base: GRPO direct (no warm-start)
RL_BASE=models/jac-qwen3coder-q4 ./rl/run_grpo.sh jac-qwen3coder

JAC_EVAL_MODEL=<base> JAC_EVAL_ADAPTER=adapters/<name>-grpo jac run 02-rl-grpo/rl/eval_rl.jac
```

## Model lineup (2 sequential runs)

| name | RL base (`RL_BASE`) | warm-start? |
|---|---|---|
| `qwen3coder` | `models/qwen-q4` (Qwen3-Coder-30B-A3B, fresh) | yes → `run_rft.sh` |
| `jac-qwen3coder` | Q4 of `models/qwen-jac-dpo-fused-q8` (SFT+DPO best) | no (already warm) |

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
