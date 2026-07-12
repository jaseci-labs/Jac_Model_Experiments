# gptoss — protocol deviation (MXFP4 fuse/Q8 broken)

**Date:** 2026-06-26

## What happened
gpt-oss-20b is MXFP4-native. In this mlx version:
- **Q8 quantization is broken** — `models/gptoss-q8` base eval = 0% runs (0/150). Model
  generates tokens but output is unusable (no compilable/runnable Jac). No adapter, no
  fuse involved — pure convert-to-Q8 is broken.
- **`mlx_lm.fuse` breaks the model at any precision** — fusing the SFT adapter into Q4
  (`models/gptoss-jac-fused-q4`) and into Q8 (`-q8`) both eval 0% (0/3 spot-check).
- **Q4 + separate adapter (unfused) works perfectly** — learning-curve subset hit 94%
  (peak 96% @ ckpt500), LIMIT-3 spot-check 100%. gpt-oss DID learn Jac well.

## Decision (per spec risk clause "if it blocks, drop gptoss and note it")
- **SFT row recorded via the Q4 + unfused-adapter eval path** (deviation from the
  standard Q8-fused eval used by all other candidates). Q4 ≤ Q8 precision, so this is
  a conservative/handicapped measurement, not an inflated one.
- **DPO = N/A.** `run_dpo` fuses SFT→base before DPO training; the fuse breaks the model.
  A fuse-free DPO path would be separate engineering, out of scope.

## Files
- `base.txt`, `finetuned.txt`, `idiom-metrics.jsonl`, `idiom-finetuned.txt` —
  regenerated on Q4 + unfused adapter (valid).
- `metrics.jsonl`, `train.log`, `*.png` — valid (Q4 training).
- The broken `models/gptoss-jac-fused-q8` and the fq4 test fuse were deleted.

## Bottom line for the bake-off
gpt-oss-20b is a **promising learner** (SFT ~94%) but **not deployable through the
standard MLX SFT→fuse→DPO pipeline** in this version. Ranked as SFT-only with an
asterisk; does not get a DPO idiom-gain number.
