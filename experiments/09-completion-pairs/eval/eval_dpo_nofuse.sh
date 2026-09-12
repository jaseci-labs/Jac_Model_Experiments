#!/usr/bin/env bash
set -euo pipefail
EXP_ABS="$(cd "$(dirname "$0")/.." && pwd)"   # <repo>/experiments/<exp> (this file: <exp>/eval/)
cd "$EXP_ABS/../.."                           # repo root
EXP="experiments/$(basename "$EXP_ABS")"
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

BASE_MODEL="models/qwen-q4"
DPO_ADAPTER="$EXP/stock-dpo/adapter"
BEST_ADAPTER="$EXP/stock-dpo/adapter-best"
HOLDOUT="${HOLDOUT:-$EXP/dataset/holdout_a_shared855.jsonl}"
RDIR="${RDIR:-$EXP/stock-dpo/results}"
# Where run_dpo_nofuse.sh wrote its run state. RDIR is redirected per holdout
# (holdout (b) runs set e.g. RDIR=$EXP/stock-dpo/results/holdoutB),
# so the step counters must be read from the TRAINING dir, not from RDIR:
# reading them from a redirected RDIR silently falls back to 250/250 and then
# skips the DPO-best cell entirely via the BEST_STEP != FINAL_STEP test below.
TRAIN_RDIR="${TRAIN_RDIR:-$EXP/stock-dpo/results}"
METRICS="$RDIR/metrics_functional.jsonl"
mkdir -p "$RDIR"

FINAL_STEP="$(cat "$TRAIN_RDIR/.dpo_progress_steps" 2>/dev/null || echo 250)"
BEST_STEP="$(cat "$TRAIN_RDIR/.best_step" 2>/dev/null || echo "$FINAL_STEP")"
LAST_EVAL_STEP=$(( FINAL_STEP + 1000000 ))
BEST_EVAL_STEP=$(( BEST_STEP + 500000 ))

echo ">>> FULL holdout eval: last checkpoint (step $FINAL_STEP)"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="$BASE_MODEL" JAC_EVAL_ADAPTER="$DPO_ADAPTER" \
  JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$LAST_EVAL_STEP" \
  jac run "$EXP/eval/eval_functional.jac" | tee "$RDIR/final_last.txt"

if [ -f "$BEST_ADAPTER/adapters.safetensors" ] && [ "$BEST_STEP" != "$FINAL_STEP" ]; then
  echo ">>> FULL holdout eval: best snapshot (step $BEST_STEP)"
  JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="$BASE_MODEL" JAC_EVAL_ADAPTER="$BEST_ADAPTER" \
    JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$BEST_EVAL_STEP" \
    jac run "$EXP/eval/eval_functional.jac" | tee "$RDIR/final_best.txt"
else
  echo ">>> best snapshot == last checkpoint (step $BEST_STEP), skipping duplicate full eval"
fi
