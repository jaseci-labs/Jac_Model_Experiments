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
# Env: EVAL_EVERY (dashboard refresh secs, 60) SUBSET (tasks/checkpoint, 50)
#      DRY_ITERS (30) SKIP_DRY=1
#      LIVE_EVAL=1  -> run a holdout eval per checkpoint DURING training. This loads
#                     a SECOND full copy of the model in-process. On a 48GB box a 30B
#                     model + the training copy exceeds RAM -> swap thrash -> deadlock.
#                     DEFAULT 0 (off). Leave off for ~30B models; the live learning
#                     signal is VAL LOSS from the train log (free, same model). The
#                     holdout test-pass% is measured at base vs finetuned regardless.
#                     Only set LIVE_EVAL=1 for small models (≲8B) that fit twice.
set -euo pipefail

# --- prevent idle sleep while running (lid-close still suspends, then resumes) ---
if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu env CAFFEINATED=1 "$0" "$@"
fi

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(cd "$SELF_DIR/.." && pwd)"   # repo root: dataset/ models/ adapters/ results/ resolve here
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

HF_MODEL="${1:?hf model id, e.g. Qwen/Qwen3-Coder-30B-A3B}"
NAME="${2:?short name, e.g. qwen}"
EVAL_EVERY="${EVAL_EVERY:-60}"; SUBSET="${SUBSET:-50}"; DRY_ITERS="${DRY_ITERS:-30}"

# --- prereqs ---
need() { command -v "$1" >/dev/null 2>&1 || { echo "MISSING: $1  (try: $2)"; exit 1; }; }
need jac "pip install jaclang"
for s in convert lora fuse generate; do need "mlx_lm.$s" "pip install mlx-lm"; done
for f in dataset/mlx/train.jsonl dataset/mlx/valid.jsonl \
         dataset/eval_holdout/conversion.jsonl sft_dpo/configs/lora.yaml; do
  [ -f "$f" ] || { echo "MISSING: $f  (run build_splits.jac / holdout.jac first)"; exit 1; }
done

mkdir -p models adapters "results/${NAME}"   # per-model results dir (no gemma/qwen clash)
RDIR="results/${NAME}"
TRAIN_LOG="$RDIR/train.log"
METRICS="$RDIR/metrics.jsonl"
ADAPTER="adapters/${NAME}-probe"
TRAIN_PID=""
cleanup() { [ -n "$TRAIN_PID" ] && kill "$TRAIN_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
done_mark() { touch "$RDIR/.$1.done"; }
is_done() { [ -f "$RDIR/.$1.done" ]; }

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
    jac run sft_dpo/jacgen/eval_probe.jac | tee "$RDIR/base.txt"
  done_mark base
fi

# --- discover training progress from saved checkpoints ---
TOTAL_ITERS="$(grep -E '^[[:space:]]*iters:' sft_dpo/configs/lora.yaml | grep -oE '[0-9]+' | head -1)"
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
  mlx_lm.lora --config sft_dpo/configs/lora.yaml --model "models/${NAME}-q4" \
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
    mlx_lm.lora --config sft_dpo/configs/lora.yaml --model "models/${NAME}-q4" \
      --adapter-path "$ADAPTER" --iters "$REMAIN" --resume-adapter-file "$LATEST_CKPT" \
      > "$TRAIN_LOG" 2>&1 &
  else
    mlx_lm.lora --config sft_dpo/configs/lora.yaml --model "models/${NAME}-q4" \
      --adapter-path "$ADAPTER" --iters "$REMAIN" > "$TRAIN_LOG" 2>&1 &
  fi
  TRAIN_PID=$!
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    STEP=$(grep -oE "Iter [0-9]+" "$TRAIN_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+" || echo 0)
    # OPT-IN ONLY: a per-checkpoint holdout eval loads a SECOND full model in-process.
    # On 48GB this OOMs/deadlocks a 30B run (swap thrash). Default off — watch val loss.
    if [ "${LIVE_EVAL:-0}" = "1" ] && [ -f "$ADAPTER_FILE" ]; then
      JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="models/${NAME}-q4" JAC_EVAL_ADAPTER="$ADAPTER" \
        JAC_EVAL_LIMIT="$SUBSET" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$((DONE_STEPS + STEP))" \
        jac run sft_dpo/jacgen/eval_probe.jac >/dev/null 2>&1 || true
    fi
    clear
    JAC_TRAIN_LOG="$TRAIN_LOG" JAC_METRICS="$METRICS" \
      jac run sft_dpo/jacgen/dashboard.jac 2>/dev/null || true
    # refresh the PNG graphs live too (open results/<name>/*.png in Preview to watch them update)
    JAC_TRAIN_LOG="$TRAIN_LOG" JAC_METRICS="$METRICS" JAC_PLOT_DIR="$RDIR" \
      jac run sft_dpo/jacgen/plot_metrics.jac >/dev/null 2>&1 || true
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

# --- 6. learning curve: eval each saved checkpoint SEQUENTIALLY (one model in RAM
#        at a time -> no OOM, unlike a concurrent live eval). Builds the real curve. ---
if is_done curve; then
  echo ">>> learning curve: already done"
else
  echo ">>> learning curve: evaluating saved checkpoints on ${SUBSET} holdout tasks"
  : > "$METRICS"   # fresh curve
  CFG="$ADAPTER/adapter_config.json"
  TMPADP="adapters/${NAME}-ckpt-eval"
  for CK in "$ADAPTER"/*_adapters.safetensors; do
    [ -e "$CK" ] || continue
    STEP="$(basename "$CK" | grep -oE '^[0-9]+' | sed 's/^0*//')"; STEP="${STEP:-0}"
    rm -rf "$TMPADP"; mkdir -p "$TMPADP"
    cp "$CK" "$TMPADP/adapters.safetensors"
    [ -f "$CFG" ] && cp "$CFG" "$TMPADP/adapter_config.json"
    echo "  checkpoint ${STEP}"
    JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="models/${NAME}-q4" JAC_EVAL_ADAPTER="$TMPADP" \
      JAC_EVAL_LIMIT="$SUBSET" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$STEP" \
      jac run sft_dpo/jacgen/eval_probe.jac 2>/dev/null | tail -3 || true
  done
  rm -rf "$TMPADP"
  # final point: the end-of-training adapter (adapters.safetensors), step = total iters
  echo "  checkpoint ${TOTAL_ITERS} (final)"
  JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="models/${NAME}-q4" JAC_EVAL_ADAPTER="$ADAPTER" \
    JAC_EVAL_LIMIT="$SUBSET" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$TOTAL_ITERS" \
    jac run sft_dpo/jacgen/eval_probe.jac 2>/dev/null | tail -3 || true
  done_mark curve
fi

# --- 7. finetuned eval (skip if recorded) ---
if is_done finetuned; then
  echo ">>> finetuned eval: already done"
else
  echo ">>> finetuned eval"
  JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="$FUSED" \
    jac run sft_dpo/jacgen/eval_probe.jac | tee "$RDIR/finetuned.txt"
  done_mark finetuned
fi

# --- 8. graphs ---
JAC_TRAIN_LOG="$TRAIN_LOG" JAC_METRICS="$METRICS" JAC_PLOT_DIR="$RDIR" \
  jac run sft_dpo/jacgen/plot_metrics.jac || echo "(pip install matplotlib for PNGs)"

echo "=== done ==="
echo "  base:      $RDIR/base.txt"
echo "  finetuned: $RDIR/finetuned.txt"
echo "  graphs:    $RDIR/*.png  (learning curve = $RDIR/learning_curve.png)"
echo "  (re-running is safe: finished stages skip, training resumes from last checkpoint)"
