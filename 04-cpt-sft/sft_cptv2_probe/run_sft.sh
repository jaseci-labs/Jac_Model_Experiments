#!/usr/bin/env bash
# SFT-on-CPT-v2 probe runner. Mirrors 01-sft-dpo/sft_dpo/run_probe.sh's structure:
# resumable, dry-run-first, live log-tail dashboard (NOT a second model load --
# LIVE_EVAL stays off, confirmed OOM risk on 48GB for a 30B model).
set -euo pipefail

if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu env CAFFEINATED=1 "$0" "$@"
fi

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(cd "$SELF_DIR/../../.." && pwd)"   # repo root
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

CFG="model-experiments/04-cpt-sft/sft_cptv2_probe/configs/sft.yaml"
ADAPTER="model-experiments/04-cpt-sft/sft_cptv2_probe/adapters/sft-on-cptv2"
RDIR="model-experiments/04-cpt-sft/sft_cptv2_probe/results/sft"
TRAIN_LOG="$RDIR/train.log"
DRY_ITERS="${DRY_ITERS:-30}"
EVAL_EVERY="${EVAL_EVERY:-60}"

need() { command -v "$1" >/dev/null 2>&1 || { echo "MISSING: $1"; exit 1; }; }
need jac "pip install jaclang"; need mlx_lm.lora "pip install mlx-lm"
for f in model-experiments/04-cpt-sft/sft_cptv2_probe/dataset/sft/train.jsonl \
         model-experiments/04-cpt-sft/sft_cptv2_probe/dataset/sft/valid.jsonl \
         model-experiments/03-cpt-only/adapters/cpt-v2/adapters.safetensors "$CFG"; do
  [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }
done

mkdir -p "$RDIR" "$ADAPTER"
TRAIN_PID=""
cleanup() { [ -n "$TRAIN_PID" ] && kill "$TRAIN_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
done_mark() { touch "$RDIR/.$1.done"; }
is_done() { [ -f "$RDIR/.$1.done" ]; }

# --- dry-run (only on a truly fresh start) ---
LATEST_CKPT="$(ls "$ADAPTER"/*_adapters.safetensors 2>/dev/null | sort -V | tail -1 || true)"
if [ -z "$LATEST_CKPT" ] && ! is_done dry && [ "${SKIP_DRY:-0}" != "1" ]; then
  echo ">>> dry-run (${DRY_ITERS} iters) -- bail check"
  mlx_lm.lora --config "$CFG" --iters "$DRY_ITERS" \
    --adapter-path "model-experiments/04-cpt-sft/sft_cptv2_probe/adapters/dry" 2>&1 | tail -25
  echo ">>> dry-run done -- Ctrl-C within 8s to abort"; sleep 8
  done_mark dry
fi

TOTAL_ITERS="$(grep -E '^iters:' "$CFG" | grep -oE '[0-9]+' | head -1)"
DONE_STEPS=0
if [ -n "$LATEST_CKPT" ]; then
  DONE_STEPS="$(basename "$LATEST_CKPT" | grep -oE '^[0-9]+' | sed 's/^0*//')"; DONE_STEPS="${DONE_STEPS:-0}"
  echo ">>> resuming from step ${DONE_STEPS}"
fi
REMAIN=$(( TOTAL_ITERS - DONE_STEPS ))
ADAPTER_FILE="$ADAPTER/adapters.safetensors"

if { is_done train || [ "$REMAIN" -le 0 ]; } && [ -f "$ADAPTER_FILE" ]; then
  echo ">>> training: already complete (${DONE_STEPS}/${TOTAL_ITERS})"
else
  echo ">>> training ${REMAIN} more iters (from ${DONE_STEPS}/${TOTAL_ITERS})"
  : > "$TRAIN_LOG"
  if [ -n "$LATEST_CKPT" ]; then
    mlx_lm.lora --config "$CFG" --adapter-path "$ADAPTER" --iters "$REMAIN" \
      --resume-adapter-file "$LATEST_CKPT" > "$TRAIN_LOG" 2>&1 &
  else
    mlx_lm.lora --config "$CFG" --adapter-path "$ADAPTER" --iters "$REMAIN" \
      > "$TRAIN_LOG" 2>&1 &
  fi
  TRAIN_PID=$!
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    JAC_TRAIN_LOG="$TRAIN_LOG" JAC_METRICS="/dev/null" JAC_PLOT_DIR="$RDIR" \
      jac run model-experiments/01-sft-dpo/sft_dpo/jacgen/plot_metrics.jac >/dev/null 2>&1 || true
    sleep "$EVAL_EVERY"
  done
  RC=0; wait "$TRAIN_PID" || RC=$?
  TRAIN_PID=""
  if [ "$RC" -ne 0 ] || [ ! -f "$ADAPTER_FILE" ]; then
    echo "!!! training stopped early (exit $RC). Re-run this script to resume."
    tail -20 "$TRAIN_LOG"; exit 1
  fi
  done_mark train
fi
echo "=== SFT training done. Next: Task 3 (functional eval sweep) ==="
