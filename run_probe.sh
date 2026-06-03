#!/usr/bin/env bash
# Conversion-probe runner (SCAFFOLD — not executed by the data-prep work).
# Requires, on PATH in your (non-anaconda) Python env:
#   jaclang  (jac)          pip install jaclang
#   mlx-lm   (mlx_lm.*)     pip install mlx-lm
#   matplotlib (PNG graphs) pip install matplotlib   # optional; ASCII works without
# Plus ~50-60GB disk per model.
#
# Usage: ./run_probe.sh <hf-model-id> <short-name>
#        ./run_probe.sh Qwen/Qwen3-Coder-30B-A3B qwen
# Optional env: EVAL_EVERY (secs, default 60), SUBSET (tasks/checkpoint, default 50),
#               DRY_ITERS (default 30), SKIP_DRY=1
#
# Flow: prereqs -> quantize -> base eval -> dry-run -> live-monitored train
#       (per-checkpoint learning curve + dashboard) -> fuse -> full eval -> graphs.
set -euo pipefail

# use the project venv toolchain (jaclang + mlx-lm) — no anaconda needed
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -d "$SELF_DIR/.venv/bin" ] && export PATH="$SELF_DIR/.venv/bin:$PATH"

HF_MODEL="${1:?hf model id, e.g. Qwen/Qwen3-Coder-30B-A3B}"
NAME="${2:?short name, e.g. qwen}"
EVAL_EVERY="${EVAL_EVERY:-60}"
SUBSET="${SUBSET:-50}"
DRY_ITERS="${DRY_ITERS:-30}"

# ---- prerequisites (fail loud, not opaque) ----
need() { command -v "$1" >/dev/null 2>&1 || { echo "MISSING: $1  (try: $2)"; exit 1; }; }
need jac "pip install jaclang"
for s in convert lora fuse generate; do need "mlx_lm.$s" "pip install mlx-lm"; done
for f in dataset/mlx/train.jsonl dataset/mlx/valid.jsonl \
         dataset/eval_holdout/conversion.jsonl configs/lora.yaml; do
  [ -f "$f" ] || { echo "MISSING: $f  (run build_splits.jac / holdout.jac first)"; exit 1; }
done

mkdir -p models adapters results
TRAIN_LOG="results/${NAME}-train.log"
METRICS="results/${NAME}-metrics.jsonl"
ADAPTER="adapters/${NAME}-probe"
: > "$METRICS"
TRAIN_PID=""
cleanup() { [ -n "$TRAIN_PID" ] && kill "$TRAIN_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# ---- 1. quantize (idempotent: reuse existing) ----
quantize() {  # <out-dir> <bits>
  if [ -d "$1" ]; then echo "  reuse $1"; else
    mlx_lm.convert --hf-path "$HF_MODEL" --mlx-path "$1" -q --q-bits "$2"
  fi
}
quantize "models/${NAME}-q4" 4
quantize "models/${NAME}-q8" 8

# ---- 2. base eval (full holdout) ----
echo ">>> base eval (pre-finetune)"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="models/${NAME}-q8" \
  jac run srccurrent/jacgen/eval_probe.jac | tee "results/${NAME}-base.txt"

# ---- 3. dry-run (bail check) ----
if [ "${SKIP_DRY:-0}" != "1" ]; then
  echo ">>> dry-run (${DRY_ITERS} iters) — confirms it trains (loss drops, no NaN/OOM)"
  mlx_lm.lora --config configs/lora.yaml --model "models/${NAME}-q4" \
    --iters "$DRY_ITERS" --adapter-path "adapters/${NAME}-dry" 2>&1 | tail -25
  echo ">>> dry-run done — Ctrl-C within 8s to abort before the full run"; sleep 8
fi

# ---- 4. full train in background ----
echo ">>> training"
mlx_lm.lora --config configs/lora.yaml --model "models/${NAME}-q4" \
  --adapter-path "$ADAPTER" 2>&1 | tee "$TRAIN_LOG" &
TRAIN_PID=$!

# ---- 5. live loop: per-checkpoint subset eval (adapter, no fuse) + dashboard ----
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  STEP=$(grep -oE "Iter [0-9]+" "$TRAIN_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+" || echo 0)
  if [ -d "$ADAPTER" ]; then
    JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="models/${NAME}-q4" JAC_EVAL_ADAPTER="$ADAPTER" \
      JAC_EVAL_LIMIT="$SUBSET" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$STEP" \
      jac run srccurrent/jacgen/eval_probe.jac >/dev/null 2>&1 || true
  fi
  clear
  JAC_TRAIN_LOG="$TRAIN_LOG" JAC_METRICS="$METRICS" \
    jac run srccurrent/jacgen/dashboard.jac 2>/dev/null || true
  sleep "$EVAL_EVERY"
done
wait "$TRAIN_PID" || true
TRAIN_PID=""

# ---- 6. fuse + final full eval ----
echo ">>> fuse + final eval"
mlx_lm.fuse --model "models/${NAME}-q8" --adapter-path "$ADAPTER" \
  --save-path "models/${NAME}-jac-fused-q8"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="models/${NAME}-jac-fused-q8" \
  jac run srccurrent/jacgen/eval_probe.jac | tee "results/${NAME}-finetuned.txt"

# ---- 7. graphs (PNG; needs matplotlib) ----
JAC_TRAIN_LOG="$TRAIN_LOG" JAC_METRICS="$METRICS" \
  jac run srccurrent/jacgen/plot_metrics.jac || echo "(pip install matplotlib for PNG graphs)"

echo "=== done ==="
echo "  base:      results/${NAME}-base.txt"
echo "  finetuned: results/${NAME}-finetuned.txt"
echo "  graphs:    results/*.png  (learning curve = results/learning_curve.png)"
