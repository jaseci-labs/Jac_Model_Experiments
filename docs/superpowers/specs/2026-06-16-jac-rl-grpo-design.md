# Jac RL (GRPO) — Design Spec

**Date:** 2026-06-16
**Status:** Approved (design), pending spec review → implementation plan
**Phase:** Reinforcement learning on Jac code generation, after SFT + DPO (phase 1).

## Goal

Push three models toward idiomatic, compiler-correct Jac via **GRPO with a
verifiable reward** (the `jac run` behavioral gate), using the real open-source
Jac corpus in `this_is_jac/` as the task source. Run entirely **local on a 48 GB
Apple-Silicon machine via MLX** (`mlx-lm-lora` GRPO), mirroring the existing
SFT/DPO local workflow.

This is the natural continuation of the project's established philosophy: reward
is not a learned preference model but the **compiler + behavioral test gate**
already trusted in `jacgen/writer.jac` (`run_jac`, exit 0 + output match).

## Decisions (locked)

| Decision | Choice |
|---|---|
| RL method | GRPO (group-relative policy optimization) + verifiable jac reward |
| Reward | `jac run` exit-0 + behavioral output match + idiom bonus |
| Compute | Local MLX, 48 GB Apple Silicon, `mlx-lm-lora` GRPO |
| Task derivation | **Body-completion** of units harvested from `this_is_jac/` |
| Eval | New `this_is_jac` holdout (disjoint files, decontaminated) |
| Reorg | Full phase isolation — phase-1 code/config/docs moved to `sft_dpo/` |
| Harness style | Jac-native (harvest/split/eval) + Python reward shim (trainer-facing) |

## Model lineup (3 sequential GRPO runs)

| `NAME` | RL base | Source | Quant | Notes |
|---|---|---|---|---|
| `qwen-coder` | `models/qwen-jac-dpo-fused-q8` | local (phase-1 SFT+DPO best) | have q8 → make q4 for RL | continue from current best |
| `qwen3` | `Qwen/Qwen3-Coder-30B-A3B` | local `models/qwen-q4` | have q4 | same base, RL-only (no SFT/DPO) ablation |
| `qwen36` | `Qwen/Qwen3.6-27B` (dense, all-active) | download from HF | `mlx_lm.convert` → q4 | dense → ~2-3× slower rollouts; run last |

Qwen3.6 released 2026-04-21. The 27B variant is dense (every parameter active
per token) — stronger coding scores than its A3B sibling but heavier per-token
rollout cost on local hardware.

## Part A — Repository reorganization (full phase isolation)

### Target layout

```
ROOT after:
  rl/                    NEW: RL (GRPO) harness
  sft_dpo/               MOVED phase-1 code/config/docs/reports
    jacgen/              (from srccurrent/jacgen)
    run_probe.sh  run_dpo.sh  check.sh  process.md  make_comparison.py
    configs/lora.yaml
    docs/modeltesting/
    resultsft/
  jac_ml_studio/         stays at root (repoint only if a read path moved)
  models/   adapters/    stay at root (gitignored runtime, shared)
  dataset/               stays at root; NEW dataset/rl/ subtree
  results/               stays at root, phase-namespaced: results/$NAME/{sft,dpo,grpo}
  this_is_jac/  papers/  docs/{initmodelchoice,wholestack,superpowers}/
  context.md  README.md  setup_env.sh  .gitignore
  DELETED: presentation/
```

### Isolation principle

`dataset/`, `models/`, `adapters/`, `results/` are gitignored **runtime data**
shared by `jac_ml_studio` and both training phases. Physically relocating them
would break the studio. Therefore isolation moves **code, config, docs, and
committed result-reports** into `sft_dpo/` and `rl/`, while runtime dirs remain
at root and are **phase-namespaced**:

- `results/$NAME/{sft,dpo,grpo}/...`
- `dataset/rl/...` (alongside existing `dataset/mlx`, `dataset/mlx_dpo`)
- adapters already name-prefixed: `$NAME-{probe,dpo,grpo}`

Every moved phase-1 script has its relative paths rewritten and is re-tested
(`sft_dpo/check.sh` + a dry `jac check` parse sweep). The studio is repointed
only if a path it actually reads has moved.

### Deletions

- `presentation/` (LaTeX deck — not needed going forward)

### Path-coupling checklist (must verify after move)

- `sft_dpo/run_probe.sh`, `run_dpo.sh`, `check.sh` — internal relative paths
  (`dataset/mlx`, `results/$NAME`, `adapters/$NAME-probe`, `models/$NAME-*`).
- `sft_dpo/jacgen/*.jac` — any hardcoded `dataset/`, `results/`, `srccurrent/`
  path literals.
- `sft_dpo/configs/lora.yaml` — `data:`, `adapter_path:` keys.
- `setup_env.sh` — `srccurrent/jacgen` reference in the syntax-check line.
- `context.md`, `README.md` — links to moved paths (`srccurrent/jacgen`,
  `process.md`, `docs/modeltesting`).
- `jac_ml_studio` — confirm `JAC_STUDIO_DATA_ROOT` consumers still resolve
  `models/`, `results/`, `dataset/` at root.

## Part B — RL harness (`rl/`)

### Components

| File | Role |
|---|---|
| `rl/harvest.jac` | Walk `this_is_jac/**/*.jac`; extract completable + **testable** units (`def`, `can…with entry`, walker abilities). Filter to deterministic-runnable units (lexer, analytics, graph walkers, bench — exclude UI `.cl.jac` components with no comparable output). Emit `dataset/rl/tasks.jsonl`: `{file, unit_id, prompt (file-with-hole + instruction), reference_body, run_recipe, expected_output}`. Report yield count. |
| `rl/build_rl_splits.jac` | File-level disjoint train/holdout split + 14-gram decontam (reuse `decontam.jac` logic) → `dataset/rl/{train,holdout}.jsonl`. Holdout = whole files never trained on. |
| `rl/reward.py` | GRPO reward function imported by `mlx-lm-lora`. Splice completion into the hole → `jac run` via a **persistent jac worker** (avoid ~1-2 s jaclang import × group size) → score `= 0.3·compiles + 0.3·runs + 0.3·output_match + 0.1·idiom_bonus`; timeout/crash → 0. |
| `rl/run_grpo.sh` | Like `run_dpo.sh`: `NAME` arg, frozen-base reference via LoRA trick (one weight set resident, fits 48 GB), group size 4–8, Q4 base. → `adapters/$NAME-grpo` + `results/$NAME/grpo`. Env: `GRPO_ITERS GRPO_LR GRPO_BETA GROUP_SIZE`. |
| `rl/eval_rl.jac` | Score holdout (reuse `eval_probe.jac` load-once pattern): run% / behavior-pass% / idiom% per model vs its base for delta → `results/$NAME/grpo/eval.txt`. |
| `rl/README.md` | Run order, env vars, gotchas, model lineup. |

### Data flow

```
this_is_jac/*.jac ──harvest.jac──▶ dataset/rl/tasks.jsonl
                  ──build_rl_splits.jac──▶ dataset/rl/{train,holdout}.jsonl
train.jsonl + reward.py ──run_grpo.sh (mlx-lm-lora GRPO)──▶ adapters/$NAME-grpo
holdout.jsonl ──eval_rl.jac──▶ results/$NAME/grpo/eval.txt
```

### Reward function (detail)

For a candidate completion `c` of unit `u` in file `f`:

1. Splice `c` into `f`'s hole → temp file `f'`.
2. `jac run f'` (persistent worker, SIGALRM timeout ~2 s).
3. Score:
   - `compiles` (0/1): no parse/compile error.
   - `runs` (0/1): exit 0, no runtime exception.
   - `output_match` (0/1): stdout matches `expected_output` of the original.
   - `idiom_bonus` (0..1): `1 - rouge_l(c, py2jac(equivalent_python))` style
     signal, or graph-construct presence; low transpile-similarity = idiomatic.
4. Weighted sum (0.3 / 0.3 / 0.3 / 0.1). Any failure short-circuits to 0 below it.

### Run order (per model, sequential)

1. `qwen-coder`: quantize dpo-fused → q4, GRPO, eval.
2. `qwen3`: GRPO on local `qwen-q4`, eval.
3. `qwen36`: download `Qwen/Qwen3.6-27B` → `mlx_lm.convert` q4 → GRPO, eval.

## Risks / de-risking (front-loaded in the plan)

1. **mlx-lm-lora GRPO reward-fn API** — confirm it accepts a custom Python
   reward callable + prompt dataset. **Spike before building the harness.** If
   unsupported, fallback to TRL/Unsloth (changes the "local" decision — escalate
   to user).
2. **Dense-27B rollout cost** — all-active → slowest; may need group size 4 +
   fewer iters. Run last.
3. **jac startup tax** — persistent worker in `reward.py` is mandatory.
4. **Testable-unit yield** — if too few `this_is_jac` units are
   deterministic-runnable, augment with small wrapper test harnesses. Harvest
   reports yield first so we decide before training.

## Out of scope

- Cloud GRPO (A100/Unsloth) — local only unless risk #1 forces it.
- Gemma RL — Qwen family only this phase.
- Changes to `jac_ml_studio`, `models/`, `papers/` beyond path repointing.
- SFT/DPO retraining — phase 1 is frozen; RL builds on top.

## Success criteria

- Reorg: phase-1 scripts still run from `sft_dpo/` after path rewrite; studio
  unaffected; `presentation/` gone; clean `git status`.
- RL: each of the 3 models shows a positive holdout delta (behavior-pass% and/or
  idiom% up vs its RL base) with no compile-rate regression.
