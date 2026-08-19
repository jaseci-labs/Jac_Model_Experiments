#!/usr/bin/env bash
# Spectrum-layer SFT probe runner (Phase 1, fresh arm) -- run_sft.sh with CFG /
# ADAPTER / RDIR retargeted and `mlx_lm.lora` swapped for spectrum_lora_layers.jac.
#
# The driver is a DROP-IN for `mlx_lm.lora`: it rebinds
# linear_to_lora_layers on both mlx_lm.lora and mlx_lm.tuner.utils, then
# delegates to mlx_lm.lora.main(), which parses sys.argv exactly as the stock
# CLI does. So every flag this loop passes (--config / --iters / --adapter-path
# / --resume-adapter-file) behaves stock, and the watchdog/stall/OOM/checkpoint
# machinery below is byte-for-byte the same logic as run_sft.sh's.
#
# One addition ahead of the gate: a --verify-layers preflight. spectrum-plan.md
# §6.4 makes the trainable-parameter count the probe's cheapest correctness
# check -- a count that differs from 281.838M means the selection changed
# CAPACITY, not just placement, and invalidates the whole comparison. Training
# is gated on it, not merely advised by it.
set -euo pipefail
# NOTE (07 jac port): the training/eval drivers are .jac now and are launched with
# `jac run <driver>.jac <flags>` -- jaclang sets sys.argv = [filename] + flags, so the
# drivers' own arg handling is unchanged. [TODO: confirm the `jac` resolved on PATH is
# the SAME interpreter/venv that has mlx / mlx_lm / mlx_lm_lora installed. workflow.md's
# conventions warn a bare `jac` can resolve to a different, stale venv; if so, pin the
# absolute venv jac here.]

if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu env CAFFEINATED=1 "$0" "$@"
fi

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(cd "$SELF_DIR/../../.." && pwd)"   # repo root
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

SPEC_DIR="model-experiments/07-nitin-ds-new-sft/spectrum_probe/spectrum"
DRIVER="$SPEC_DIR/spectrum_lora_layers.jac"
LAYERS="$SPEC_DIR/configs/spectrum_layers.json"
CFG="$SPEC_DIR/configs/sft_spectrum.yaml"
ADAPTER="model-experiments/07-nitin-ds-new-sft/spectrum_probe/adapters/sft-on-nitin-spectrum"
CKPT_DIR="$ADAPTER/checkpoints"
RDIR="model-experiments/07-nitin-ds-new-sft/spectrum_probe/results/sft-spectrum"
TRAIN_LOG="$RDIR/train.log"
DRY_ITERS="${DRY_ITERS:-30}"
EVAL_EVERY="${EVAL_EVERY:-60}"
STALL_SECS="${SFT_STALL_SECS:-900}"
OOM_RECOVERY_ITERS="${SFT_OOM_RECOVERY_ITERS:-100}"

need() { command -v "$1" >/dev/null 2>&1 || { echo "MISSING: $1"; exit 1; }; }
need jac; need python
for f in model-experiments/07-nitin-ds-new-sft/dataset/sft/train.jsonl \
         model-experiments/07-nitin-ds-new-sft/dataset/sft/valid.jsonl \
         "$CFG" "$DRIVER"; do
  [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }
done
if [ ! -f "$LAYERS" ]; then
  echo "MISSING: $LAYERS"
  echo "  The layer selection is produced by the SNR scan, not by this script."
  echo "  Run spectrum-workflow.md Phases 1-2 first:"
  echo "    python $SPEC_DIR/snr_scan.py --snapshot <bf16 snapshot> --out $SPEC_DIR/snr/snr_raw.json"
  echo "    python $SPEC_DIR/layer_select.py --snr $SPEC_DIR/snr/snr_raw.json \\"
  echo "      --layer-scores-out $SPEC_DIR/snr/layer_scores.json --selection-out $LAYERS"
  exit 1
fi

# Preflight: a competing resident model OOMs this 48GB machine regardless of
# anything this script does (documented dual-model-load gotcha from the CPT work).
if pgrep -f "jac start" >/dev/null 2>&1 || pgrep -f "mlx_lm" >/dev/null 2>&1; then
  echo "!!! another jac/mlx_lm process is already running -- stop it first."
  pgrep -fl "jac start|mlx_lm" || true
  exit 1
fi

mkdir -p "$RDIR" "$ADAPTER" "$CKPT_DIR"
done_mark() { touch "$RDIR/.$1.done"; }
is_done() { [ -f "$RDIR/.$1.done" ]; }
ADAPTER_FILE="$ADAPTER/adapters.safetensors"
PROGRESS_FILE="$RDIR/.sft_progress_steps"

# --- self-test gate (spectrum-plan.md §6.4) -------------------------------
if ! is_done verify && [ "${SKIP_VERIFY:-0}" != "1" ]; then
  echo ">>> --verify-layers self-test (loads models/qwen-q4 twice; a few minutes)"
  jac run "$DRIVER" --verify-layers --spectrum-layers "$LAYERS" \
    --model models/qwen-q4 2>&1 | tee "$RDIR/verify_layers.txt"
  if ! grep -q "^VERIFY: PASS" "$RDIR/verify_layers.txt"; then
    echo "!!! self-test FAILED -- see $RDIR/verify_layers.txt. Not training."
    exit 1
  fi
  done_mark verify
fi

# --- dry-run (advisory, does not gate) -------------------------------------
if [ ! -f "$ADAPTER_FILE" ] && ! is_done dry && [ "${SKIP_DRY:-0}" != "1" ]; then
  echo ">>> dry-run (${DRY_ITERS} iters) -- bail check"
  jac run "$DRIVER" --spectrum-layers "$LAYERS" --config "$CFG" --iters "$DRY_ITERS" \
    --adapter-path "model-experiments/07-nitin-ds-new-sft/spectrum_probe/adapters/dry-spectrum" 2>&1 | tail -25
  echo ">>> dry-run complete."
  done_mark dry
fi

# --- the actual safety gate. UNCONDITIONAL on ADAPTER_FILE existence and NOT
# additionally gated on "did a dry-run happen" -- run_sft.sh's comment records
# why: with SKIP_DRY=1 on a fresh start an AND-gate never fires and training
# launches with ZERO confirmation required.
if [ ! -f "$ADAPTER_FILE" ] && [ "${CONFIRM_FULL_RUN:-}" != "1" ]; then
  echo "Self-test and dry-run complete (or skipped). Re-run with CONFIRM_FULL_RUN=1 to start the real multi-hour training run."
  exit 0
fi

TOTAL_ITERS="$(grep -E '^iters:' "$CFG" | grep -oE '[0-9]+' | head -1)"
TOTAL_ITERS="${TOTAL_ITERS:-8200}"
[ -f "$PROGRESS_FILE" ] || echo 0 > "$PROGRESS_FILE"

if is_done train && [ -f "$ADAPTER_FILE" ]; then
  echo ">>> training: already complete"
  exit 0
fi

consecutive_fails=0
oom_shrinks=0
attempt_iters_override=""
while true; do
  DONE_STEPS="$(cat "$PROGRESS_FILE")"
  REMAIN=$(( TOTAL_ITERS - DONE_STEPS ))
  if [ "$REMAIN" -le 0 ] && [ -f "$ADAPTER_FILE" ]; then
    done_mark train
    echo "=== spectrum SFT training done ($DONE_STEPS/$TOTAL_ITERS). Next: eval_sft_spectrum.sh ==="
    break
  fi
  ATTEMPT_ITERS="$REMAIN"
  if [ -n "$attempt_iters_override" ] && [ "$attempt_iters_override" -lt "$REMAIN" ]; then
    ATTEMPT_ITERS="$attempt_iters_override"
  fi
  echo ">>> SFT attempt: requesting ${ATTEMPT_ITERS} iters (${DONE_STEPS}/${TOTAL_ITERS} done, ${REMAIN} remaining)" | tee -a "$TRAIN_LOG"

  BEFORE_CKPTS="$(ls "$ADAPTER"/*_adapters.safetensors 2>/dev/null | xargs -n1 basename 2>/dev/null || true)"
  : > "$RDIR/.segment.log"
  # Two explicit branches, NOT an empty-array expansion -- macOS bash 3.2 treats
  # "${ARR[@]}" as an unbound-variable error under `set -u` when ARR is empty.
  if [ "$DONE_STEPS" -gt 0 ] && [ -f "$ADAPTER_FILE" ]; then
    jac run "$DRIVER" --spectrum-layers "$LAYERS" --config "$CFG" --adapter-path "$ADAPTER" \
      --iters "$ATTEMPT_ITERS" --resume-adapter-file "$ADAPTER_FILE" >> "$RDIR/.segment.log" 2>&1 &
  else
    jac run "$DRIVER" --spectrum-layers "$LAYERS" --config "$CFG" --adapter-path "$ADAPTER" \
      --iters "$ATTEMPT_ITERS" >> "$RDIR/.segment.log" 2>&1 &
  fi
  SEG_PID=$!

  last_growth=$(date +%s); last_size=0
  while kill -0 "$SEG_PID" 2>/dev/null; do
    sleep "$EVAL_EVERY"
    cur_size="$(wc -l < "$RDIR/.segment.log" 2>/dev/null || echo 0)"; now="$(date +%s)"
    if [ "$cur_size" -gt "$last_size" ]; then last_size="$cur_size"; last_growth="$now"; fi
    if [ $(( now - last_growth )) -ge "$STALL_SECS" ]; then
      echo "!!! stalled: no log growth for ${STALL_SECS}s -- killing PID $SEG_PID and treating as a failed attempt" | tee -a "$TRAIN_LOG"
      kill -9 "$SEG_PID" 2>/dev/null || true
      break
    fi
    JAC_TRAIN_LOG="$RDIR/.segment.log" JAC_METRICS="/dev/null" JAC_PLOT_DIR="$RDIR" \
      jac run model-experiments/01-sft-dpo/sft_dpo/jacgen/plot_metrics.jac >/dev/null 2>&1 || true
  done
  RC=0; wait "$SEG_PID" 2>/dev/null || RC=$?
  cat "$RDIR/.segment.log" >> "$TRAIN_LOG"

  AFTER_CKPTS="$(ls "$ADAPTER"/*_adapters.safetensors 2>/dev/null | xargs -n1 basename 2>/dev/null || true)"
  NEW_CKPTS="$(comm -13 <(echo "$BEFORE_CKPTS" | sort) <(echo "$AFTER_CKPTS" | sort) 2>/dev/null || true)"
  MAX_NEW_LOCAL=0
  for f in $NEW_CKPTS; do
    ln="$(echo "$f" | grep -oE '^[0-9]+' | sed 's/^0*//')"; ln="${ln:-0}"
    true_step=$(( DONE_STEPS + ln ))
    cp "$ADAPTER/$f" "$CKPT_DIR/$(printf '%07d' "$true_step")_adapters.safetensors"
    [ "$ln" -gt "$MAX_NEW_LOCAL" ] && MAX_NEW_LOCAL="$ln"
  done

  if [ "$RC" -ne 0 ]; then
    consecutive_fails=$(( consecutive_fails + 1 ))
    echo "!!! attempt failed (attempt ${consecutive_fails})" | tee -a "$TRAIN_LOG"
    if [ "$MAX_NEW_LOCAL" -gt 0 ]; then
      NEW_DONE=$(( DONE_STEPS + MAX_NEW_LOCAL ))
      echo "$NEW_DONE" > "$PROGRESS_FILE"
      echo "  real progress persisted before the crash: now at ${NEW_DONE}/${TOTAL_ITERS}" | tee -a "$TRAIN_LOG"
    fi
    if grep -qEi "out of memory|OutOfMemory|kIOGPUCommandBuffer|MTL::.*(OOM|Insufficient)" "$RDIR/.segment.log"; then
      echo "!!! OOM signature detected in segment log" | tee -a "$TRAIN_LOG"
      if [ "$oom_shrinks" -lt 2 ]; then
        oom_shrinks=$(( oom_shrinks + 1 ))
        attempt_iters_override="$OOM_RECOVERY_ITERS"
        echo "!!! OOM-recovery ${oom_shrinks}/2: next attempt(s) capped at ${OOM_RECOVERY_ITERS} iters" | tee -a "$TRAIN_LOG"
      else
        echo "!!! OOM persisted through the shrink ladder -- giving up." | tee -a "$TRAIN_LOG"
        exit 1
      fi
    fi
    if [ "$consecutive_fails" -ge 5 ]; then
      echo "!!! attempt failed 5x in a row at the same ${DONE_STEPS}/${TOTAL_ITERS} point -- giving up." | tee -a "$TRAIN_LOG"
      tail -20 "$TRAIN_LOG"; exit 1
    fi
    continue
  fi

  consecutive_fails=0
  attempt_iters_override=""
  NEW_DONE=$(( DONE_STEPS + ATTEMPT_ITERS ))
  echo "$NEW_DONE" > "$PROGRESS_FILE"
done
