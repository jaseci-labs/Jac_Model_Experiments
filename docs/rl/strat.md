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
| See also | [workflow.md](workflow.md) (diagram) · [`rl/README.md`](../../rl/README.md) (how-to) · [spec](../superpowers/specs/2026-06-16-jac-rl-grpo-design.md) · harness [`rl/`](../../rl/). |

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

- **Pure-lib** (`source_index._lang_label`, `source_lexer` tokenisation, other pure helpers) — function-shaped, ~8–12 maskable bodies. Warm-up.
- **Graph-spatial** (`littlex/social_graph`: 32 abilities + 5 defs; `guestbook`: 10 abilities + 4 defs) — walker/node/ability logic spawned on an in-memory graph, ~50 maskable bodies in two files. The idiomatic gold and the real target.

Counts are *maskable bodies* (`can … with` abilities, `def` methods) — the units a driver can hole out — not archetype declarations (`edge {}`/`node` headers have no body). ~50 graph bodies + the pure-lib tier comfortably clears the ≥30-driver target.

---

## The three models + warm-start

| `NAME` | base model | jac-trained? | RL path |
|---|---|---|---|
| `qwen3coder` | `models/qwen-q4` (Qwen3-Coder-30B-A3B) | no | **warm-start → GRPO** |
| `jac-qwen3coder` | `models/qwen-jac-dpo-fused-q8` → q4 | yes (SFT+DPO) | **GRPO direct** |
| `qwen36` | `Qwen/Qwen3.6-27B` (dense) → q4 | no | **warm-start → GRPO** |

**The cold-start problem.** GRPO needs the base to *sometimes* produce compiling Jac, or every rollout scores ~0 → zero advantage → no gradient. The jac-trained base already produces Jac, so it goes straight to GRPO. The two fresh bases have never seen Jac and emit Python-shaped, mostly non-compiling code → sparse reward → GRPO stalls.

**Fix (combine, minimally):** warm-start the fresh bases before GRPO. Wired as
**`rl/run_rft.sh`** (RFT): `rft_sample.jac` samples N completions per task from the
base, keeps the ones that pass the *same* jac reward (`reward_logic.score_one`),
then LoRA-SFTs on them and fuses → `models/<name>-rft-q4`, a warmed base GRPO can
climb from. Reuses the reward + the task prompts, no new data. (Alternative: full
SFT on the conversion corpus via `sft_dpo/run_probe.sh` — heavier, not needed.)

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
  --group-size 4 --max-completion-length 256 --beta 0.04 ...
```

Knobs (env): `GRPO_ITERS`(200) `GROUP_SIZE`(4) `GRPO_LR`(1e-6) `GRPO_BETA`(0.04) `MAX_COMPLETION`(256) `MAX_SEQ`(1280) `GRPO_LAYERS`(8). LoRA only; `--grad-checkpoint` to fit 30B activations. Defaults fit a 30B-A3B q4 on 48GB (measured peak ~38GB); `group6/comp512` OOMs Metal at ~iter 2. Dense `qwen36` is all-active → run last, keep `GROUP_SIZE=4` (or 3).

**Cost shape.** Per step = `group_size` completions per prompt, each scored by a `jac` subprocess (~1–2 s startup tax × group size). Measured: **~17 s/iter** at `GROUP_SIZE=4`, `MAX_COMPLETION=256` on the 30B-A3B q4 — so a 200-iter run is **~1 h** per model (the dedup cache + short completions keep it cheap). The dense 27B is heavier per step. If step time hurts, replace the per-completion subprocess with an in-process jaclang runner (optimization, not a blocker).

---

## Evaluation

Held-out `this_is_jac` tasks (disjoint from train, decontaminated), scored by [`rl/eval_rl.jac`](../../rl/eval_rl.jac): load the model once, generate per task, splice + isolated `jac run`, report **run% / behavior-pass% / avg idiom count**. Run base vs `+grpo` adapter per model; a win = behavior-pass% and/or idiom up with no compile-rate regression. Aggregate into `results/RL_RESULTS.md`.

---

## Harness state & remaining work

**Built + validated** (merged to `main`): repo isolated into `sft_dpo/` + `rl/`; reward logic in Jac (`rl/reward_logic.jac`, behind a ~5-line `.py` shim the trainer's loader requires), tests green (`jac run rl/test_reward.jac`); task pipeline (`build_tasks` / `build_rl_splits`); runner + eval. The full GRPO loop ran on a real 30B (`qwen-q4`, 2 iters): the `jac_behavioral` reward loaded, scored rollouts, frozen reference fit RAM, adapter produced. The whole harness is Jac (`*.jac`) except a `run_grpo.sh` launcher and the unavoidable reward shim. **No engineering risk remains.**

Drivers authored: **31 tasks** (24 train / 7 holdout), all determinism-verified
(two isolated rebuilds byte-identical). Warm-start (`run_rft.sh`) built and
smoke-validated on `qwen-q4` (sample → score → keep → SFT-format write).

**Remaining is pure compute (no scaffolding left):**

1. **Warm-start the two fresh bases** — `RL_BASE=models/qwen-q4 ./rl/run_rft.sh qwen3coder` (and `qwen36`).
2. **Run the three GRPO jobs**, sequential, ~24–36 h total.
3. **Eval each** base-vs-`+grpo`; write `results/RL_RESULTS.md`.

---

## Run order

Sequence: author drivers → `build_tasks` → `build_rl_splits` → (warm-start fresh bases) → `run_grpo.sh` per model → `eval_rl.jac` base-vs-`+grpo`. The exact commands, `RL_BASE` per model, env knobs, and gotchas live in **[`rl/README.md`](../../rl/README.md)** (the operational home — kept there to avoid drift). This doc owns the *why*; the README owns the *how*.
