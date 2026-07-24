#!/usr/bin/env bash
# SFT-on-fresh-Qwen probe runner, watchdog-supervised. Unlike
# sft_cptv2_probe/run_sft.sh (one long-lived process for the whole remaining
# iter count -- that arm's NaN-crash-and-resume had to be caught and resumed
# by hand), this segments training into save_every-sized chunks and actively
# polls each segment: log growth (not the child's self-report) proves real
# progress, a stalled segment gets killed and restarted, a crashed segment
# gets retried, and a detected OOM shrinks the segment size (then, if that's
# not enough, would need a human -- this script does not silently degrade
# past a documented floor).
set -euo pipefail

if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu env CAFFEINATED=1 "$0" "$@"
fi

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(cd "$SELF_DIR/../../.." && pwd)"   # repo root
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

CFG="model-experiments/04-cpt-sft/sft_fresh_probe/configs/sft.yaml"
ADAPTER="model-experiments/04-cpt-sft/sft_fresh_probe/adapters/sft-on-fresh"
CKPT_DIR="$ADAPTER/checkpoints"    # true-global-step-named snapshots, see the loop comment below for why
RDIR="model-experiments/04-cpt-sft/sft_fresh_probe/results/sft"
TRAIN_LOG="$RDIR/train.log"
DRY_ITERS="${DRY_ITERS:-30}"
EVAL_EVERY="${EVAL_EVERY:-60}"
SEGMENT_ITERS="${SFT_SEGMENT_ITERS:-820}"    # = save_every, so every segment boundary is a real checkpoint
STALL_SECS="${SFT_STALL_SECS:-900}"          # 15min with no new log line => treat as hung

need() { command -v "$1" >/dev/null 2>&1 || { echo "MISSING: $1"; exit 1; }; }
need jac "pip install jaclang"; need mlx_lm.lora "pip install mlx-lm"
for f in model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/train.jsonl \
         model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl "$CFG"; do
  [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }
done

# Preflight: the CPT-v1/v2 dual-model-load scripts OOM'd on this same 48GB
# machine when the studio dev server (its own resident model) was left
# running -- catch that proactively instead of discovering it via an OOM.
if pgrep -f "jac start" >/dev/null 2>&1 || pgrep -f "mlx_lm" >/dev/null 2>&1; then
  echo "!!! another jac/mlx_lm process is already running -- stop it first (a competing"
  echo "!!! resident model will OOM a 48GB machine regardless of this script's own settings)."
  pgrep -fl "jac start|mlx_lm" || true
  exit 1
fi

mkdir -p "$RDIR" "$ADAPTER" "$CKPT_DIR"
done_mark() { touch "$RDIR/.$1.done"; }
is_done() { [ -f "$RDIR/.$1.done" ]; }
ADAPTER_FILE="$ADAPTER/adapters.safetensors"
PROGRESS_FILE="$RDIR/.sft_progress_steps"

# --- dry-run (only on a truly fresh start) ---
if [ ! -f "$ADAPTER_FILE" ] && ! is_done dry && [ "${SKIP_DRY:-0}" != "1" ]; then
  echo ">>> dry-run (${DRY_ITERS} iters) -- bail check"
  mlx_lm.lora --config "$CFG" --iters "$DRY_ITERS" \
    --adapter-path "model-experiments/04-cpt-sft/sft_fresh_probe/adapters/dry" 2>&1 | tail -25
  echo ">>> dry-run complete."
  done_mark dry
fi

# --- gate: only proceed to real training if explicitly confirmed ---
if [ ! -f "$ADAPTER_FILE" ] && is_done dry && [ "${CONFIRM_FULL_RUN:-}" != "1" ]; then
  echo "Dry-run complete. Re-run with CONFIRM_FULL_RUN=1 to start the real multi-hour training run."
  exit 0
fi

TOTAL_ITERS="$(grep -E '^iters:' "$CFG" | grep -oE '[0-9]+' | head -1)"
TOTAL_ITERS="${TOTAL_ITERS:-8200}"
[ -f "$PROGRESS_FILE" ] || echo 0 > "$PROGRESS_FILE"

if is_done train && [ -f "$ADAPTER_FILE" ]; then
  echo ">>> training: already complete"
  exit 0
fi

# Watchdog-supervised segmented loop. Progress is tracked via an EXPLICIT
# counter file, NOT by parsing mlx_lm.lora's own checkpoint filenames.
# VERIFIED against the installed package (.venv/lib/.../mlx_lm/tuner/
# trainer.py:374-375): checkpoints are named "{it:07d}_adapters.safetensors"
# where `it` is the LOCAL per-invocation loop counter, not a global step --
# since every segment here calls `--iters "$SEG" --save-every "$SEG"`
# (forcing exactly one numbered save per segment, always at local iter
# $SEG), a naive "parse the latest numbered file" resume would read the
# SAME filename every segment and never detect real progress. This is the
# same bug CLASS already hit and fixed in this repo's DPO runner (a
# checkpoint-bookkeeping bug that silently re-resumed from the same stale
# checkpoint 5 times) -- fixed here the same way DPO fixed it: an explicit
# progress file, and resuming from the POINTER file (adapters.safetensors,
# confirmed elsewhere in this repo via mtime to always be the real current
# state), never from a numbered file.
consecutive_fails=0
oom_shrinks=0
while true; do
  DONE_STEPS="$(cat "$PROGRESS_FILE")"
  REMAIN=$(( TOTAL_ITERS - DONE_STEPS ))
  if [ "$REMAIN" -le 0 ] && [ -f "$ADAPTER_FILE" ]; then
    done_mark train
    echo "=== SFT training done ($DONE_STEPS/$TOTAL_ITERS). Next: functional eval sweep ==="
    break
  fi
  SEG=$(( REMAIN < SEGMENT_ITERS ? REMAIN : SEGMENT_ITERS ))
  echo ">>> SFT segment: ${SEG} iters (${DONE_STEPS}/${TOTAL_ITERS} done, ${REMAIN} remaining)" | tee -a "$TRAIN_LOG"

  : > "$RDIR/.segment.log"
  RESUME_FLAGS=()
  if [ "$DONE_STEPS" -gt 0 ] && [ -f "$ADAPTER_FILE" ]; then
    RESUME_FLAGS=(--resume-adapter-file "$ADAPTER_FILE")
  fi
  # --save-every "$SEG" (CLI override, not the config's static 820) guarantees
  # exactly one numbered checkpoint per segment even if the OOM ladder below
  # later shrinks SEGMENT_ITERS to something that no longer divides evenly.
  mlx_lm.lora --config "$CFG" --adapter-path "$ADAPTER" --iters "$SEG" --save-every "$SEG" \
    "${RESUME_FLAGS[@]}" >> "$RDIR/.segment.log" 2>&1 &
  SEG_PID=$!

  # Poll: refresh the live graphs every EVAL_EVERY; kill+treat-as-failed if
  # no new log line for STALL_SECS -- this is the "no task dies unattended"
  # mechanism, driven by real file growth, not the child's exit status alone.
  last_growth=$(date +%s); last_size=0
  while kill -0 "$SEG_PID" 2>/dev/null; do
    sleep "$EVAL_EVERY"
    cur_size="$(wc -l < "$RDIR/.segment.log" 2>/dev/null || echo 0)"; now="$(date +%s)"
    if [ "$cur_size" -gt "$last_size" ]; then last_size="$cur_size"; last_growth="$now"; fi
    if [ $(( now - last_growth )) -ge "$STALL_SECS" ]; then
      echo "!!! stalled: no log growth for ${STALL_SECS}s -- killing PID $SEG_PID and treating as a failed segment" | tee -a "$TRAIN_LOG"
      kill -9 "$SEG_PID" 2>/dev/null || true
      break
    fi
    JAC_TRAIN_LOG="$RDIR/.segment.log" JAC_METRICS="/dev/null" JAC_PLOT_DIR="$RDIR" \
      jac run model-experiments/01-sft-dpo/sft_dpo/jacgen/plot_metrics.jac >/dev/null 2>&1 || true
  done
  RC=0; wait "$SEG_PID" 2>/dev/null || RC=$?
  cat "$RDIR/.segment.log" >> "$TRAIN_LOG"

  if [ "$RC" -ne 0 ]; then
    consecutive_fails=$(( consecutive_fails + 1 ))
    echo "!!! segment failed (attempt ${consecutive_fails})" | tee -a "$TRAIN_LOG"
    if grep -qEi "out of memory|OutOfMemory|kIOGPUCommandBuffer|MTL::.*(OOM|Insufficient)" "$RDIR/.segment.log"; then
      echo "!!! OOM signature detected in segment log" | tee -a "$TRAIN_LOG"
      if [ "$oom_shrinks" -lt 2 ]; then
        oom_shrinks=$(( oom_shrinks + 1 ))
        SEGMENT_ITERS=$(( SEGMENT_ITERS / 2 )); [ "$SEGMENT_ITERS" -lt 50 ] && SEGMENT_ITERS=50
        echo "!!! OOM-recovery ${oom_shrinks}/2: shrinking SEGMENT_ITERS to ${SEGMENT_ITERS} and retrying" | tee -a "$TRAIN_LOG"
      else
        echo "!!! OOM persisted through the shrink ladder (2/2 already applied) -- giving up. Check for a competing process (the preflight check above should have caught one that started BEFORE this run; a new one may have started since)." | tee -a "$TRAIN_LOG"
        exit 1
      fi
    fi
    if [ "$consecutive_fails" -ge 5 ]; then
      echo "!!! segment failed 5x in a row at the same ${DONE_STEPS}/${TOTAL_ITERS} point -- giving up." | tee -a "$TRAIN_LOG"
      tail -20 "$TRAIN_LOG"; exit 1
    fi
    continue
  fi
  consecutive_fails=0

  NEW_DONE=$(( DONE_STEPS + SEG ))
  echo "$NEW_DONE" > "$PROGRESS_FILE"
  # Snapshot this segment's checkpoint under its TRUE global step. The
  # numbered file mlx_lm.lora just wrote is named by LOCAL iteration count
  # (see the comment above the loop) -- copy it immediately under an
  # unambiguous global-step name so Task 5's sweep can trust filenames
  # directly, no "+N" offset correction needed afterward.
  LOCAL_NAME="$(printf '%07d' "$SEG")_adapters.safetensors"
  if [ -f "$ADAPTER/$LOCAL_NAME" ]; then
    cp "$ADAPTER/$LOCAL_NAME" "$CKPT_DIR/$(printf '%07d' "$NEW_DONE")_adapters.safetensors"
    echo "  checkpoint snapshot: $CKPT_DIR/$(printf '%07d' "$NEW_DONE")_adapters.safetensors" | tee -a "$TRAIN_LOG"
  fi
done
