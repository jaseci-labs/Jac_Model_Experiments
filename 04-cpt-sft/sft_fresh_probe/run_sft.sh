#!/usr/bin/env bash
# SFT-on-fresh-Qwen probe runner, watchdog-supervised. Runs training as ONE
# continuous mlx_lm.lora process whenever possible -- an earlier draft of
# this script artificially chunked training into fixed-size segments every
# save_every=820 iters (like the DPO runner does) and was caught in code
# review: mlx_lm's LoRA optimizer rebuilds fresh on every process launch,
# and --resume-adapter-file only restores WEIGHTS, not optimizer/schedule
# state (verified directly against .venv/lib/.../mlx/optimizers/
# optimizers.py -- this is the SAME documented risk already known in this
# repo for CPT-v2's multi-leg training, see project memory). With
# warmup==820==the old segment size, that design would have kept EVERY
# segment inside the LR warmup ramp for the entire 8200-iter run, never
# reaching the configured cosine decay -- a silent, no-error defect that
# would have made this arm's SFT run not comparable to the CPT-v2 arm's
# (which ran as one continuous process, resuming only once after a real
# crash). Fixed: DPO's segmenting is unaffected by this (it uses a flat
# `--learning-rate`, no schedule to reset) so it's untouched; SFT here
# only relaunches (which DOES reset the schedule -- same accepted tradeoff
# the CPT-v2 arm's own one-time crash-resume already had) on a REAL stall
# or crash, never as a routine cadence. Log growth (not the child's exit
# status alone) proves real progress; a stalled process gets killed and
# retried; a detected OOM caps the next attempt(s) at a small recovery
# size until one succeeds, then reverts to requesting the full remainder.
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
STALL_SECS="${SFT_STALL_SECS:-900}"          # 15min with no new log line => treat as hung
OOM_RECOVERY_ITERS="${SFT_OOM_RECOVERY_ITERS:-100}"  # capped attempt size right after an OOM; reverts to full-remaining once one attempt succeeds

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

# --- dry-run (only on a truly fresh start; purely advisory, does not gate) ---
if [ ! -f "$ADAPTER_FILE" ] && ! is_done dry && [ "${SKIP_DRY:-0}" != "1" ]; then
  echo ">>> dry-run (${DRY_ITERS} iters) -- bail check"
  mlx_lm.lora --config "$CFG" --iters "$DRY_ITERS" \
    --adapter-path "model-experiments/04-cpt-sft/sft_fresh_probe/adapters/dry" 2>&1 | tail -25
  echo ">>> dry-run complete."
  done_mark dry
fi

# --- gate: only proceed to real training if explicitly confirmed. This check
# is UNCONDITIONAL on ADAPTER_FILE existence -- NOT additionally gated on
# "did a dry-run happen" (is_done dry). An earlier draft chained both with
# &&, which review caught as a real bypass: with SKIP_DRY=1 on a fresh
# start, is_done dry stays false forever, so that AND-gate never fired and
# training launched with ZERO confirmation required. Fixed by dropping the
# is_done dry condition entirely -- the dry-run above is an advisory sanity
# check, this is the actual safety gate and must hold regardless.
if [ ! -f "$ADAPTER_FILE" ] && [ "${CONFIRM_FULL_RUN:-}" != "1" ]; then
  echo "Dry-run complete (or skipped). Re-run with CONFIRM_FULL_RUN=1 to start the real multi-hour training run."
  exit 0
fi

TOTAL_ITERS="$(grep -E '^iters:' "$CFG" | grep -oE '[0-9]+' | head -1)"
TOTAL_ITERS="${TOTAL_ITERS:-8200}"
[ -f "$PROGRESS_FILE" ] || echo 0 > "$PROGRESS_FILE"

if is_done train && [ -f "$ADAPTER_FILE" ]; then
  echo ">>> training: already complete"
  exit 0
fi

# Watchdog-supervised loop: launches ONE continuous mlx_lm.lora process
# sized to run the REST of training in one shot (preserving the LR
# schedule -- see the file header comment for why this matters), and only
# relaunches on a REAL stall or crash, never as a routine cadence. Progress
# is tracked via an EXPLICIT counter file, driven only by REAL persisted
# state (a clean RC==0 exit, or newly-written numbered checkpoint files on
# a crash) -- never by parsing a numbered checkpoint filename directly,
# since that number is LOCAL to each invocation (verified against
# .venv/lib/.../mlx_lm/tuner/trainer.py:374-375) and would be ambiguous
# across more than one resume.
consecutive_fails=0
oom_shrinks=0
attempt_iters_override=""
while true; do
  DONE_STEPS="$(cat "$PROGRESS_FILE")"
  REMAIN=$(( TOTAL_ITERS - DONE_STEPS ))
  if [ "$REMAIN" -le 0 ] && [ -f "$ADAPTER_FILE" ]; then
    done_mark train
    echo "=== SFT training done ($DONE_STEPS/$TOTAL_ITERS). Next: functional eval sweep ==="
    break
  fi
  ATTEMPT_ITERS="$REMAIN"
  if [ -n "$attempt_iters_override" ] && [ "$attempt_iters_override" -lt "$REMAIN" ]; then
    ATTEMPT_ITERS="$attempt_iters_override"
  fi
  echo ">>> SFT attempt: requesting ${ATTEMPT_ITERS} iters (${DONE_STEPS}/${TOTAL_ITERS} done, ${REMAIN} remaining)" | tee -a "$TRAIN_LOG"

  BEFORE_CKPTS="$(ls "$ADAPTER"/*_adapters.safetensors 2>/dev/null | xargs -n1 basename 2>/dev/null || true)"
  : > "$RDIR/.segment.log"
  RESUME_FLAGS=()
  if [ "$DONE_STEPS" -gt 0 ] && [ -f "$ADAPTER_FILE" ]; then
    RESUME_FLAGS=(--resume-adapter-file "$ADAPTER_FILE")
  fi
  mlx_lm.lora --config "$CFG" --adapter-path "$ADAPTER" --iters "$ATTEMPT_ITERS" \
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
      echo "!!! stalled: no log growth for ${STALL_SECS}s -- killing PID $SEG_PID and treating as a failed attempt" | tee -a "$TRAIN_LOG"
      kill -9 "$SEG_PID" 2>/dev/null || true
      break
    fi
    JAC_TRAIN_LOG="$RDIR/.segment.log" JAC_METRICS="/dev/null" JAC_PLOT_DIR="$RDIR" \
      jac run model-experiments/01-sft-dpo/sft_dpo/jacgen/plot_metrics.jac >/dev/null 2>&1 || true
  done
  RC=0; wait "$SEG_PID" 2>/dev/null || RC=$?
  cat "$RDIR/.segment.log" >> "$TRAIN_LOG"

  # Archive any newly-written numbered checkpoints under their TRUE global
  # step name, regardless of success or failure. The new local number is
  # LOCAL to this invocation; true_global = DONE_STEPS (the value BEFORE
  # this invocation started) + new_local_number.
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
        echo "!!! OOM-recovery ${oom_shrinks}/2: next attempt(s) capped at ${OOM_RECOVERY_ITERS} iters until one succeeds, then reverting to full-remaining requests" | tee -a "$TRAIN_LOG"
      else
        echo "!!! OOM persisted through the shrink ladder (2/2 already applied) -- giving up. Check for a competing process (the preflight check above should have caught one that started BEFORE this run; a new one may have started since)." | tee -a "$TRAIN_LOG"
        exit 1
      fi
    fi
    if [ "$consecutive_fails" -ge 5 ]; then
      echo "!!! attempt failed 5x in a row at the same ${DONE_STEPS}/${TOTAL_ITERS} point -- giving up." | tee -a "$TRAIN_LOG"
      tail -20 "$TRAIN_LOG"; exit 1
    fi
    continue
  fi

  # Success: RC==0 means mlx_lm.lora completed all ATTEMPT_ITERS iters and
  # wrote the final pointer (trainer.py always does a final save at loop
  # end, regardless of save_every alignment) -- trust the requested count
  # directly rather than re-deriving it from checkpoint filenames.
  consecutive_fails=0
  attempt_iters_override=""
  NEW_DONE=$(( DONE_STEPS + ATTEMPT_ITERS ))
  echo "$NEW_DONE" > "$PROGRESS_FILE"
done
