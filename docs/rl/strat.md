# Reinforcement Learning Strategy

*GRPO with a verifiable reward to push Jac code generation toward compiler-correct, behaviorally-correct, idiomatic output — the RL phase after SFT + DPO. Runs locally on 48 GB Apple-Silicon MLX.*

| | |
|---|---|
| Method | GRPO (group-relative policy optimization) — no value net, no learned reward model. |
| Reward | The `jac run` gate itself: `0.3·compiles + 0.3·runs + 0.3·output_match + 0.1·idiom`. |
| Task source | Authored body-completion drivers over the real open-source corpus `this_is_jac/`. |
| Models | `qwen3coder` (fresh), `jac-qwen3coder` (SFT+DPO best), `qwen36` (Qwen3.6-27B dense). |
| Compute | Local MLX, `mlx-lm-lora` 2.1.0 GRPO, frozen-base reference (one weight set in RAM). |
| Eval | Held-out `this_is_jac` tasks: run% / behavior-pass% / idiom-construct count, base vs RL. |
| Spec / plan | [`docs/superpowers/specs/2026-06-16-jac-rl-grpo-design.md`](../superpowers/specs/2026-06-16-jac-rl-grpo-design.md), harness in [`rl/`](../../rl/). |

---

## Why RL, why verifiable reward

SFT teaches the shape of Jac; DPO de-Python-ifies it. Neither optimizes the thing we can actually measure for free: **does the generated Jac compile and behave correctly.** Jac ships a compiler and runtime, so every candidate has a ground-truth grade with zero LLM in the loop. That makes this RL-with-verifiable-rewards (RLVR), the same regime that works for math/code — the reward is an oracle, not a preference model that can be gamed or drift.

The reward is deliberately **multi-term**, not pass/fail:

- `compiles` (0/1) — parses + type-resolves (`jac check -p` on failure path).
- `runs` (0/1) — `jac run` exits 0.
- `output_match` (0/1) — stdout equals the reference program's stdout.
- `idiom` (0..1) — density of graph-spatial constructs (`walker`, `node`, `visit`, `-->`, `report`, …), normalized.

Partial credit (compiles-but-wrong scores 0.6) gives a gradient even before a model produces fully-correct output — critical for a sparse-reward domain. `idiom` only applies to running completions, so non-running output can never earn it: no reward-hacking via idiom keywords on dead code.

---

## Task construction: authored drivers

`this_is_jac/` is finished applications, not a task/test set. An audit of all 77 files found **zero** that run standalone *and* deterministic (UI components, server walkers, wasm/native demos, timing benchmarks). So tasks are **authored**, not harvested.

Each task is a hand-written **driver** (`rl/drivers/<id>.jac`): a complete, deterministic `jac run`-able file that exercises one corpus unit, with that unit's body wrapped in sentinels:

```jac
# >>>HOLE id="graph_trending_tally" instruction="<natural-language spec>"
<the real body>
# <<<HOLE
```

`rl/build_tasks.jac` runs each driver as-authored to capture ground-truth stdout, then emits a GRPO record `{prompt, answer}` plus a sidecar template (body replaced by `__HOLE__`). At train time the policy regenerates the body; the reward splices its completion into the template and runs it.

Two hard rules make this work:

1. **Persistence isolation.** Every `jac run` writes a `.jac/` graph in the cwd; a persistent `root` accumulates state across runs. Build, reward, and eval all run snippets with `cwd` = a throwaway temp dir, so each run starts clean.
2. **Determinism.** `jid()` returns random UUIDs and `_now()` is time-based. Drivers print only deterministic projections (usernames, content, counts, likes) — never raw view objects that embed ids/timestamps.

The corpus splits into two task tiers (used as a curriculum, easy → hard):

- **Pure-lib** (`source_lexer`, `source_index`, pure helpers) — function-shaped, ~8–12 units. Warm-up.
- **Graph-spatial** (`littlex/social_graph` 65 units, `guestbook` 22) — walkers/nodes/edges/abilities spawned on an in-memory graph. The idiomatic gold and the real target.

---

## The three models + warm-start

| `NAME` | base model | jac-trained? | RL path |
|---|---|---|---|
| `qwen3coder` | `models/qwen-q4` (Qwen3-Coder-30B-A3B) | no | **warm-start → GRPO** |
| `jac-qwen3coder` | `models/qwen-jac-dpo-fused-q8` → q4 | yes (SFT+DPO) | **GRPO direct** |
| `qwen36` | `Qwen/Qwen3.6-27B` (dense) → q4 | no | **warm-start → GRPO** |

**The cold-start problem.** GRPO needs the base to *sometimes* produce compiling Jac, or every rollout scores ~0 → zero advantage → no gradient. The jac-trained base already produces Jac, so it goes straight to GRPO. The two fresh bases have never seen Jac and emit Python-shaped, mostly non-compiling code → sparse reward → GRPO stalls.

**Fix (combine, minimally):** warm-start the fresh bases before GRPO. Two options, cheapest first:

1. **RFT / rejection-sampling FT** — sample N completions per task with the *same* reward oracle, keep the passing ones, SFT on them. Reuses the reward, no new data, bootstraps a non-zero compile rate.
2. **SFT** on the existing conversion corpus (`sft_dpo/` pipeline, `run_probe.sh`) — heavier but the proven path; the project already has the data and the runner.

Then GRPO on the warmed base. The jac-trained model needs none of this — it is the control showing GRPO-on-top-of-SFT+DPO, while the fresh models are the ablation showing how far RL alone (after a light warm-start) can reach.

---

## Why GRPO, and what *not* to add

GRPO is the right spine: it drops PPO's value network (memory the 48 GB box does not have), works directly off scalar rewards, and the group-relative baseline suits a verifiable oracle. The reference model is left frozen (= the base), so only one weight set sits in RAM.

**Combine where it pays:**
- **Warm-start (SFT/RFT) → GRPO** for cold bases — see above.
- **Multi-term reward** — already in place; keeps the gradient dense and fights hacking.
- **Curriculum** — order `build_rl_splits` easy → hard so early steps see solvable tasks.

**Do not add:**
- **PPO** — reintroduces the value net; GRPO exists precisely to remove it here.
- **Learned reward model** — we have a perfect verifiable oracle; a learned RM is strictly worse and gameable.
- **Heavy KL machinery** — the frozen-reference KL term (`--beta`) is enough.

---

## GRPO configuration (local MLX)

`mlx-lm-lora` 2.1.0, invoked by [`rl/run_grpo.sh`](../../rl/run_grpo.sh):

```
python -m mlx_lm_lora.train --train-mode grpo --data dataset/rl \
  --reward-functions-file rl/reward.py --reward-functions jac_behavioral \
  --group-size 6 --max-completion-length 512 --beta 0.04 ...
```

Knobs (env): `GRPO_ITERS`(300) `GROUP_SIZE`(6) `GRPO_LR`(1e-6) `GRPO_BETA`(0.04) `MAX_COMPLETION`(512) `GRPO_LAYERS`(8). LoRA only; `--grad-checkpoint` to fit 30B activations. Dense `qwen36` is all-active → heaviest rollouts → `GROUP_SIZE=4`, run last.

**Cost shape.** Each step generates `group_size` completions per prompt and scores each with a `jac` subprocess (~1–2 s startup tax × group size). ~8–12 h per 30B model at 300 iters; the dense 27B is longer. If step time hurts, replace the per-completion subprocess with an in-process jaclang runner (optimization, not a blocker).

---

## Evaluation

Held-out `this_is_jac` tasks (disjoint from train, decontaminated), scored by [`rl/eval_rl.jac`](../../rl/eval_rl.jac): load the model once, generate per task, splice + isolated `jac run`, report **run% / behavior-pass% / avg idiom count**. Run base vs `+grpo` adapter per model; a win = behavior-pass% and/or idiom up with no compile-rate regression. Aggregate into `results/RL_RESULTS.md`.

---

## Harness state & remaining work

**Built + validated** (branch `rl-phase`): repo isolated into `sft_dpo/` + `rl/`; reward (`rl/reward.py`, 4 tests green); task pipeline (`build_tasks` / `build_rl_splits`); runner + eval. The full GRPO loop ran on a real 30B (`qwen-q4`, 2 iters): the `jac_behavioral` reward loaded, scored rollouts, frozen reference fit RAM, adapter produced. **No engineering risk remains.**

**Remaining (content + compute, not scaffolding):**

1. **Author ≥30 drivers.** Three seeds exist; RL on three tasks gives uniform reward (zero gradient). Bulk authoring is the gate before any run is worth the wall-clock. ~3–5 h batched (shared graph fixtures) or ~1.5–2 days solo.
2. **Warm-start the two fresh bases** (RFT or SFT) — required, else cold-start stall.
3. **Run the three GRPO jobs**, sequential, ~24–36 h total; eval each; write `RL_RESULTS.md`.

---

## Run order

```bash
source .venv/bin/activate

# 1. build the task set (after authoring rl/drivers/*.jac)
jac run rl/build_tasks.jac
jac run rl/build_rl_splits.jac

# 2. (fresh bases only) warm-start, then GRPO
RL_BASE=models/qwen-q4 ./rl/run_grpo.sh qwen3coder

mlx_lm.convert --hf-path models/qwen-jac-dpo-fused-q8 --mlx-path models/jac-qwen3coder-q4 -q --q-bits 4
RL_BASE=models/jac-qwen3coder-q4 ./rl/run_grpo.sh jac-qwen3coder

mlx_lm.convert --hf-path Qwen/Qwen3.6-27B --mlx-path models/qwen36-q4 -q --q-bits 4
RL_BASE=models/qwen36-q4 GROUP_SIZE=4 ./rl/run_grpo.sh qwen36

# 3. eval each (baseline then +grpo)
JAC_EVAL_MODEL=<base> jac run rl/eval_rl.jac
JAC_EVAL_MODEL=<base> JAC_EVAL_ADAPTER=adapters/<name>-grpo jac run rl/eval_rl.jac
```
