#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

SFT_FUSED="models/sft-cptv2-fused-q4"
DPO_ADAPTER="model-experiments/04-cpt-sft/sft_cptv2_probe/adapters/dpo-on-sft-v3"
BEST_ADAPTER="model-experiments/04-cpt-sft/sft_cptv2_probe/adapters/dpo-on-sft-v3-best"
HOLDOUT="model-experiments/04-cpt-sft/sft_cptv2_probe/dataset/sft/valid.jsonl"
RDIR="model-experiments/04-cpt-sft/sft_cptv2_probe/results/dpo-v3"
METRICS="$RDIR/metrics_functional.jsonl"   # keep -- already has the per-snapshot subset rows

FINAL_STEP="$(cat "$RDIR/.dpo_progress_steps" 2>/dev/null || echo 250)"
BEST_STEP="$(cat "$RDIR/.best_step" 2>/dev/null || echo "$FINAL_STEP")"
LAST_EVAL_STEP=$(( FINAL_STEP + 1000000 ))
BEST_EVAL_STEP=$(( BEST_STEP + 500000 ))

echo ">>> FULL holdout eval: last checkpoint (step $FINAL_STEP)"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="$SFT_FUSED" JAC_EVAL_ADAPTER="$DPO_ADAPTER" \
  JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$LAST_EVAL_STEP" \
  jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac | tee "$RDIR/final_last.txt"

if [ -f "$BEST_ADAPTER/adapters.safetensors" ] && [ "$BEST_STEP" != "$FINAL_STEP" ]; then
  echo ">>> FULL holdout eval: best snapshot (step $BEST_STEP)"
  JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="$SFT_FUSED" JAC_EVAL_ADAPTER="$BEST_ADAPTER" \
    JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$BEST_EVAL_STEP" \
    jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac | tee "$RDIR/final_best.txt"
else
  echo ">>> best snapshot == last checkpoint (step $BEST_STEP), skipping duplicate full eval"
fi
