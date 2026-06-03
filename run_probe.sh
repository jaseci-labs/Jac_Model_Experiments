#!/usr/bin/env bash
# Conversion-probe runner (SCAFFOLD — not executed by the data-prep work).
# Requires, on PATH in your (non-anaconda) Python env:
#   jaclang  (jac)          pip install jaclang
#   mlx-lm   (mlx_lm.*)     pip install mlx-lm
#   matplotlib (graphs)     pip install matplotlib
# Plus ~50-60GB disk per model.
#
# Usage: ./run_probe.sh <hf-model-id> <short-name>
#        ./run_probe.sh Qwen/Qwen3-Coder-30B-A3B qwen
#
# Flow: quantize -> base eval -> DRY RUN -> full train (live dashboard +
# per-checkpoint learning curve) -> fuse -> final eval -> graphs.
set -euo pipefail

HF_MODEL="${1:?hf model id, e.g. Qwen/Qwen3-Coder-30B-A3B}"
NAME="${2:?short name, e.g. qwen}"
mkdir -p models adapters results
TRAIN_LOG="results/${NAME}-train.log"
METRICS="results/${NAME}-metrics.jsonl"
ADAPTER="adapters/${NAME}-probe"
: > "$METRICS"

# 1. quantize: Q4 (train) + Q8 (eval)
mlx_lm.convert --hf-path "$HF_MODEL" --mlx-path "models/${NAME}-q4" -q --q-bits 4
mlx_lm.convert --hf-path "$HF_MODEL" --mlx-path "models/${NAME}-q8" -q --q-bits 8

# 2. base eval (full 150) — before any finetuning
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="models/${NAME}-q8" \
  jac run srccurrent/jacgen/eval_probe.jac | tee "results/${NAME}-base.txt"

# 3. DRY RUN (30 iters) — confirm it trains (loss drops, no NaN/OOM) before committing
echo ">>> dry-run (30 iters)"
mlx_lm.lora --config configs/lora.yaml --model "models/${NAME}-q4" \
  --iters 30 --adapter-path "adapters/${NAME}-dry" 2>&1 | tail -25
echo ">>> dry-run done — Ctrl-C within 8s to abort before the full run"
sleep 8

# 4. FULL TRAIN in background, tee to log
echo ">>> training (full)"
mlx_lm.lora --config configs/lora.yaml --model "models/${NAME}-q4" \
  --adapter-path "$ADAPTER" 2>&1 | tee "$TRAIN_LOG" &
TRAIN_PID=$!

# 5. live loop until training ends: per-checkpoint eval (50-subset, adapter — no
#    fuse) appends a learning-curve point, then redraw the ASCII dashboard.
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  STEP=$(grep -oE "Iter [0-9]+" "$TRAIN_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+" || echo 0)
  if [ -d "$ADAPTER" ]; then
    JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="models/${NAME}-q4" JAC_EVAL_ADAPTER="$ADAPTER" \
      JAC_EVAL_LIMIT=50 JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$STEP" \
      jac run srccurrent/jacgen/eval_probe.jac >/dev/null 2>&1 || true
  fi
  clear
  JAC_TRAIN_LOG="$TRAIN_LOG" JAC_METRICS="$METRICS" \
    jac run srccurrent/jacgen/dashboard.jac 2>/dev/null || true
  sleep 60
done
wait "$TRAIN_PID" || true

# 6. fuse adapter into Q8, final full eval (150)
mlx_lm.fuse --model "models/${NAME}-q8" --adapter-path "$ADAPTER" \
  --save-path "models/${NAME}-jac-fused-q8"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="models/${NAME}-jac-fused-q8" \
  jac run srccurrent/jacgen/eval_probe.jac | tee "results/${NAME}-finetuned.txt"

# 7. graphs (matplotlib PNGs in results/)
JAC_TRAIN_LOG="$TRAIN_LOG" JAC_METRICS="$METRICS" \
  jac run srccurrent/jacgen/plot_metrics.jac || echo "(pip install matplotlib for PNG graphs)"

echo "=== done ==="
echo "  base:      results/${NAME}-base.txt"
echo "  finetuned: results/${NAME}-finetuned.txt"
echo "  graphs:    results/*.png   (learning curve = results/learning_curve.png)"
