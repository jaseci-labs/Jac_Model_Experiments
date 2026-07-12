# ling (Ling-Coder-lite) — DROPPED (BailingMoE unsupported in this mlx pipeline)

**Date:** 2026-06-26

## What happened
inclusionAI/Ling-Coder-lite is a **BailingMoE** architecture. Two blockers:

1. **Standard convert failed** — `mlx_lm.convert` rejected the model under strict
   weight load: `ValueError: Received 1 parameters not in model: model.rotary_emb.inv_freq`.
   Salvaged by converting with strict load relaxed (mlx recomputes RoPE; the buffer is
   a benign derived value). Produced `models/ling-q4` + `models/ling-q8`.

2. **Runs but does not work** — the converted model loads into a *generic* class
   (`transformers` warning: "instantiate a model of type ``"), needs `trust_remote_code`
   (interactive prompt, fed via `yes |`). Generation is **runaway**: every eval emits
   exactly 512 tokens (the max cap) with no clean stop — the chat template / EOS handling
   for BailingMoE is not honored by the mlx_lm path.

## Evidence (learning curve, 50-task subset, all 6 checkpoints)
```
step 100-500: runs 0/50, test_pass 0%   (25600 gen tokens = 512/task = max cap every time)
step 600:     runs 1/50, test_pass 2%   (single fluke)
```
SFT never taught it to produce runnable Jac — not a learning failure, a
generation/integration failure (wrong model class + no EOS).

## Decision
**Dropped from the bake-off.** Unlike gpt-oss (which learned well on a working Q4 path),
Ling-Coder-lite does not run correctly under the installed mlx_lm at all. Supporting it
would need BailingMoE added to mlx_lm (a separate effort / version bump that would break
the one-variable control for the other candidates). Row = N/A.

Killed the chained run before DPO (pointless on a 0% SFT model).
