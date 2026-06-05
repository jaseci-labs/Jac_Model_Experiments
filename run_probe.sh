#!/usr/bin/env bash
# Conversion-probe runner — RESUMABLE. Safe to kill / shutdown / re-run: it skips
# finished stages and resumes LoRA training from the last saved checkpoint.
#
# Requires (in your non-anaconda Python env, on PATH):
#   jaclang (jac), mlx-lm (mlx_lm.*), matplotlib   # see setup_env.sh
#   ~50-60GB disk per model.
#
# Usage: ./run_probe.sh <hf-model-id> <short-name>
#        ./run_probe.sh Qwen/Qwen3-Coder-30B-A3B qwen
# Env: EVAL_EVERY (dashboard secs, 60) SUBSET (tasks/checkpoint, 50)
#      DRY_ITERS (30) SKIP_DRY=1
set -euo pipefail

# --- prevent idle sleep while running (lid-close still suspends, then resumes) ---
if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu env CAFFEINATED=1 "$0" "$@"
fi

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -d "$SELF_DIR/.venv/bin" ] && export PATH="$SELF_DIR/.venv/bin:$PATH"

HF_MODEL="${1:?hf model id, e.g. Qwen/Qwen3-Coder-30B-A3B}"
NAME="${2:?short name, e.g. qwen}"
EVAL_EVERY="${EVAL_EVERY:-60}"; SUBSET="${SUBSET:-50}"; DRY_ITERS="${DRY_ITERS:-30}"

# --- prereqs ---
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
TRAIN_PID=""
cleanup() { [ -n "$TRAIN_PID" ] && kill "$TRAIN_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
done_mark() { touch "results/.${NAME}.$1.done"; }
is_done() { [ -f "results/.${NAME}.$1.done" ]; }

# --- 1. quantize (idempotent + completeness check) ---
quantize() {  # <out> <bits>
  if [ -f "$1/config.json" ] && ls "$1"/*.safetensors >/dev/null 2>&1; then
    echo "  reuse $1"
  else
    rm -rf "$1"   # nuke any partial/corrupt quantize from a mid-kill
    mlx_lm.convert --hf-path "$HF_MODEL" --mlx-path "$1" -q --q-bits "$2"
  fi
}
quantize "models/${NAME}-q4" 4
quantize "models/${NAME}-q8" 8

# --- 2. base eval (skip if already recorded) ---
if is_done base; then
  echo ">>> base eval: already done"
else
  echo ">>> base eval (pre-finetune)"
  JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="models/${NAME}-q8" \
    jac run srccurrent/jacgen/eval_probe.jac | tee "results/${NAME}-base.txt"
  done_mark base
fi

# --- discover training progress from saved checkpoints ---
TOTAL_ITERS="$(grep -E '^[[:space:]]*iters:' configs/lora.yaml | grep -oE '[0-9]+' | head -1)"
TOTAL_ITERS="${TOTAL_ITERS:-600}"
LATEST_CKPT="$(ls "$ADAPTER"/*_adapters.safetensors 2>/dev/null | sort -V | tail -1 || true)"
DONE_STEPS=0
if [ -n "$LATEST_CKPT" ]; then
  DONE_STEPS="$(basename "$LATEST_CKPT" | grep -oE '^[0-9]+' | sed 's/^0*//')"
  DONE_STEPS="${DONE_STEPS:-0}"
  echo ">>> found checkpoint at step ${DONE_STEPS} -> resuming"
fi

# --- 3. dry-run (only on a truly fresh start) ---
if [ "$DONE_STEPS" -eq 0 ] && ! is_done dry && [ "${SKIP_DRY:-0}" != "1" ]; then
  echo ">>> dry-run (${DRY_ITERS} iters) — bail check"
  mlx_lm.lora --config configs/lora.yaml --model "models/${NAME}-q4" \
    --iters "$DRY_ITERS" --adapter-path "adapters/${NAME}-dry" 2>&1 | tail -25
  echo ">>> dry-run done — Ctrl-C within 8s to abort"; sleep 8
  done_mark dry
fi

# --- 4. train (resume from checkpoint; remaining iters only) ---
REMAIN=$(( TOTAL_ITERS - DONE_STEPS ))
ADAPTER_FILE="$ADAPTER/adapters.safetensors"
# "done" requires a real adapter on disk — self-heals a stale marker from an interrupt
if { is_done train || [ "$REMAIN" -le 0 ]; } && [ -f "$ADAPTER_FILE" ]; then
  echo ">>> training: already complete (${DONE_STEPS}/${TOTAL_ITERS})"
  done_mark train
else
  echo ">>> training ${REMAIN} more iters (from ${DONE_STEPS}/${TOTAL_ITERS})"
  [ "$DONE_STEPS" -eq 0 ] && : > "$METRICS"   # fresh start clears the learning curve
  : > "$TRAIN_LOG"
  # redirect (not pipe): $! is the TRAINER, not tee; the dashboard reads the log live.
  # two branches avoid an empty-array expansion (errors under set -u on macOS bash 3.2).
  if [ -n "$LATEST_CKPT" ]; then
    mlx_lm.lora --config configs/lora.yaml --model "models/${NAME}-q4" \
      --adapter-path "$ADAPTER" --iters "$REMAIN" --resume-adapter-file "$LATEST_CKPT" \
      > "$TRAIN_LOG" 2>&1 &
  else
    mlx_lm.lora --config configs/lora.yaml --model "models/${NAME}-q4" \
      --adapter-path "$ADAPTER" --iters "$REMAIN" > "$TRAIN_LOG" 2>&1 &
  fi
  TRAIN_PID=$!
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    STEP=$(grep -oE "Iter [0-9]+" "$TRAIN_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+" || echo 0)
    if [ -f "$ADAPTER_FILE" ]; then
      JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="models/${NAME}-q4" JAC_EVAL_ADAPTER="$ADAPTER" \
        JAC_EVAL_LIMIT="$SUBSET" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$((DONE_STEPS + STEP))" \
        jac run srccurrent/jacgen/eval_probe.jac >/dev/null 2>&1 || true
    fi
    clear
    JAC_TRAIN_LOG="$TRAIN_LOG" JAC_METRICS="$METRICS" \
      jac run srccurrent/jacgen/dashboard.jac 2>/dev/null || true
    # refresh the PNG graphs live too (open results/*.png in Preview to watch them update)
    JAC_TRAIN_LOG="$TRAIN_LOG" JAC_METRICS="$METRICS" \
      jac run srccurrent/jacgen/plot_metrics.jac >/dev/null 2>&1 || true
    sleep "$EVAL_EVERY"
  done
  RC=0; wait "$TRAIN_PID" || RC=$?
  TRAIN_PID=""
  if [ "$RC" -ne 0 ] || [ ! -f "$ADAPTER_FILE" ]; then
    echo "!!! training stopped early (exit $RC, no adapter at $ADAPTER_FILE)."
    echo "    last log lines:"; tail -20 "$TRAIN_LOG"
    echo "    re-run the same command to resume from the last checkpoint."
    exit 1
  fi
  done_mark train
fi

# --- 5. fuse (skip if present) ---
FUSED="models/${NAME}-jac-fused-q8"
[ -f "$ADAPTER_FILE" ] || { echo "!!! no trained adapter at $ADAPTER_FILE — train didn't finish"; exit 1; }
if [ -f "$FUSED/config.json" ]; then echo ">>> fuse: reuse $FUSED"; else
  echo ">>> fuse"
  mlx_lm.fuse --model "models/${NAME}-q8" --adapter-path "$ADAPTER" --save-path "$FUSED"
fi

# --- 6. finetuned eval (skip if recorded) ---
if is_done finetuned; then
  echo ">>> finetuned eval: already done"
else
  echo ">>> finetuned eval"
  JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="$FUSED" \
    jac run srccurrent/jacgen/eval_probe.jac | tee "results/${NAME}-finetuned.txt"
  done_mark finetuned
fi

# --- 7. graphs ---
JAC_TRAIN_LOG="$TRAIN_LOG" JAC_METRICS="$METRICS" \
  jac run srccurrent/jacgen/plot_metrics.jac || echo "(pip install matplotlib for PNGs)"

echo "=== done ==="
echo "  base:      results/${NAME}-base.txt"
echo "  finetuned: results/${NAME}-finetuned.txt"
echo "  graphs:    results/*.png  (learning curve = results/learning_curve.png)"
echo "  (re-running is safe: finished stages skip, training resumes from last checkpoint)"
