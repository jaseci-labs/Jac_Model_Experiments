#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

BASE_MODEL="models/qwen-q4"
DPO_ADAPTER="model-experiments/06-nitin-ds-sft/stock_probe/adapters/dpo-on-sft-nitin-nofuse"
BEST_ADAPTER="model-experiments/06-nitin-ds-sft/stock_probe/adapters/dpo-on-sft-nitin-nofuse-best"
HOLDOUT="${HOLDOUT:-model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl}"
RDIR="${RDIR:-model-experiments/06-nitin-ds-sft/stock_probe/results/dpo-nofuse}"
# Where run_dpo_nofuse.sh wrote its run state. RDIR is redirected per holdout
# (workflow.md §5.2 -- holdout (b) writes to results/dpo-nofuse-nitinholdout/),
# so the step counters must be read from the TRAINING dir, not from RDIR:
# reading them from a redirected RDIR silently falls back to 250/250 and then
# skips the DPO-best cell entirely via the BEST_STEP != FINAL_STEP test below.
TRAIN_RDIR="${TRAIN_RDIR:-model-experiments/06-nitin-ds-sft/stock_probe/results/dpo-nofuse}"
METRICS="$RDIR/metrics_functional.jsonl"
mkdir -p "$RDIR"

FINAL_STEP="$(cat "$TRAIN_RDIR/.dpo_progress_steps" 2>/dev/null || echo 250)"
BEST_STEP="$(cat "$TRAIN_RDIR/.best_step" 2>/dev/null || echo "$FINAL_STEP")"
LAST_EVAL_STEP=$(( FINAL_STEP + 1000000 ))
BEST_EVAL_STEP=$(( BEST_STEP + 500000 ))

echo ">>> FULL holdout eval: last checkpoint (step $FINAL_STEP)"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="$BASE_MODEL" JAC_EVAL_ADAPTER="$DPO_ADAPTER" \
  JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$LAST_EVAL_STEP" \
  jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac | tee "$RDIR/final_last.txt"

if [ -f "$BEST_ADAPTER/adapters.safetensors" ] && [ "$BEST_STEP" != "$FINAL_STEP" ]; then
  echo ">>> FULL holdout eval: best snapshot (step $BEST_STEP)"
  JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="$BASE_MODEL" JAC_EVAL_ADAPTER="$BEST_ADAPTER" \
    JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$BEST_EVAL_STEP" \
    jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac | tee "$RDIR/final_best.txt"
else
  echo ">>> best snapshot == last checkpoint (step $BEST_STEP), skipping duplicate full eval"
fi
