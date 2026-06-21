# RL Results — GRPO on Jac (local MLX, 48GB)

Local results summary. Tracked record + full narrative: `docs/rl/runlog.md`.
Pass % = exact stdout match after body-completion. Holdout 7 tasks, train 24.

| model | base ho/tr | +warm ho/tr | +grpo ho/tr | notes |
|---|---|---|---|---|
| **jac-qwen3coder** (MoE, SFT+DPO) | 14.3% / 8.3% | n/a (near-ceiling) | 14.3% / 8.3% | base already near-gold; GRPO no greedy change |
| **qwen3coder** (MoE, fresh) | 0% / 8.3% | **14.3%** / 8.3% | 14.3% / 8.3% | **warm-start lifted holdout 0→14.3%**; GRPO flat |
| **qwen36** (dense 27B, fresh) | 0% / 0% | OOM | OOM | dense untrainable on 48GB (inference fits) |

## Headline

- **Warm-start (gold-SFT) is the measured win:** fresh Qwen3-Coder holdout pass
  0% → 14.3%.
- **LoRA-GRPO changed no greedy outputs** at feasible LR (≤1e-5)/300 iters on any
  model — reward variance + nonzero loss present, but KL≈0, greedy identical.
- **Dense 27B can't be LoRA-trained on 48GB** (all-active activation memory);
  MoE 30B-A3B trains fine (~38GB peak).

## The bug that mattered

All early runs were flat because of a **splice bug**: models emit the whole unit
(`can X { … }`) but the template hole sat *inside* the unit → nested, broken Jac →
nothing ran. Fixed with `unwrap_unit` in `extract_jac`. After the fix, runs ==
pass and the real (modest) numbers above appeared.

## Artifacts

- adapters: `adapters/{jac-qwen3coder,qwen3coder}-grpo`, `*-rft` (warm)
- models: `models/{jac-qwen3coder,qwen3coder}-rft-q4` (warm), `models/qwen36-q4`
- per-stage eval txt: `results/<model>/final/` and `results/jac-qwen3coder/{grpo3,fixed}/`

## If continuing

- Easier-graded tasks (line-level / partial pass) so base pass > 0 broadly →
  stronger RL gradient.
- More tasks (100s) for generalization.
- Higher-LR / full-finetune RL (not LoRA) to actually move greedy — needs more
  VRAM than 48GB for 30B.
- Dense 27B: needs a bigger box or a smaller/MoE variant.
