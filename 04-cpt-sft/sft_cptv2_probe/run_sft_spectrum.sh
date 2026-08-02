#!/usr/bin/env bash
# Spectrum-layer SFT probe runner (Phase 2, cptv2 arm).
#
# Structurally identical to sft_fresh_probe/run_sft_spectrum.sh -- same watchdog,
# stall detector, OOM shrink ladder, checkpoint archiver, and the same
# UNCONDITIONAL CONFIRM_FULL_RUN=1 gate. Two arm-specific differences:
#
#   1. The driver is cptv2_spectrum_lora_layers.py, not spectrum_lora_layers.py.
#      It converts the union `picks | {32..47}` and freezes `{32..47} \ picks` so
#      resume_adapter_file's 256 CPT-v2 keys all land and none of them consume
#      the 16-block trainable budget (spectrum-plan.md §8.3 steps 1-4).
#      resume_adapter_file comes from the YAML, exactly as the incumbent
#      run_sft.sh gets it -- the lineage logic is unchanged, only which blocks
#      carry gradients.
#
#   2. A POST-TRAINING MERGE (§8.3 step 5). mlx_lm/tuner/trainer.py saves only
#      model.trainable_parameters(), so the frozen CPT-v2 blocks are absent from
#      every file training writes. merge_frozen_keys.py puts them back, into the
#      final adapter AND into every archived checkpoint, before eval ever sees
#      them. Without it the CPT-v2 delta vanishes at eval and this arm is scored
#      as a partially-trained model -- the same silent-loss class as §7.
#
# spectrum-plan.md §12 marks this arm CONDITIONAL on the Phase-1 gate
# (spectrum-workflow.md Phase 6: paired McNemar p<0.05 AND |delta|>=2.8pp vs
# sft-on-fresh). This script does not enforce that -- it is a human decision
# recorded in results/sft-spectrum/GATE.md on the fresh arm. Do not launch this
# before that file says the gate opened.
set -euo pipefail

if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu env CAFFEINATED=1 "$0" "$@"
fi

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(cd "$SELF_DIR/../../.." && pwd)"   # repo root
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

SPEC_DIR="model-experiments/04-cpt-sft/sft_cptv2_probe/spectrum"
DRIVER="$SPEC_DIR/cptv2_spectrum_lora_layers.py"
MERGER="$SPEC_DIR/merge_frozen_keys.py"
LAYERS="$SPEC_DIR/configs/spectrum_layers.json"
CFG="$SPEC_DIR/configs/sft_spectrum.yaml"
CPT_ADAPTER="model-experiments/03-cpt-only/adapters/cpt-v2/adapters.safetensors"
ADAPTER="model-experiments/04-cpt-sft/sft_cptv2_probe/adapters/sft-on-cptv2-spectrum"
CKPT_DIR="$ADAPTER/checkpoints"
RDIR="model-experiments/04-cpt-sft/sft_cptv2_probe/results/sft-spectrum"
TRAIN_LOG="$RDIR/train.log"
DRY_ITERS="${DRY_ITERS:-30}"
EVAL_EVERY="${EVAL_EVERY:-60}"
STALL_SECS="${SFT_STALL_SECS:-900}"
OOM_RECOVERY_ITERS="${SFT_OOM_RECOVERY_ITERS:-100}"

need() { command -v "$1" >/dev/null 2>&1 || { echo "MISSING: $1"; exit 1; }; }
need jac; need python
for f in model-experiments/04-cpt-sft/sft_cptv2_probe/dataset/sft/train.jsonl \
         model-experiments/04-cpt-sft/sft_cptv2_probe/dataset/sft/valid.jsonl \
         "$CPT_ADAPTER" "$CFG" "$DRIVER" "$MERGER"; do
  [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }
done
if [ ! -f "$LAYERS" ]; then
  echo "MISSING: $LAYERS"
  echo "  It is a symlink to the fresh arm's frozen selection -- both arms use the"
  echo "  SAME picks (the SNR ranking is a property of the base model, not the arm)."
  echo "  Run spectrum-workflow.md Phases 1-2 in sft_fresh_probe/spectrum/ first."
  exit 1
fi

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

# --- self-test gate (spectrum-plan.md §6.4 + §8.3 steps 1-4) ---------------
# Asserts, on the real 30B model: the union is converted, all 256 CPT-v2 keys
# land, only the 16 picks are trainable, and trainable == 281.838M exactly.
if ! is_done verify && [ "${SKIP_VERIFY:-0}" != "1" ]; then
  echo ">>> --verify-layers self-test (loads models/qwen-q4 twice; a few minutes)"
  python "$DRIVER" --verify-layers --spectrum-layers "$LAYERS" \
    --cpt-adapter "$CPT_ADAPTER" --model models/qwen-q4 2>&1 | tee "$RDIR/verify_layers.txt"
  if ! grep -q "^VERIFY: PASS" "$RDIR/verify_layers.txt"; then
    echo "!!! self-test FAILED -- see $RDIR/verify_layers.txt. Not training."
    exit 1
  fi
  done_mark verify
fi

# --- dry-run (advisory, does not gate) -------------------------------------
if [ ! -f "$ADAPTER_FILE" ] && ! is_done dry && [ "${SKIP_DRY:-0}" != "1" ]; then
  echo ">>> dry-run (${DRY_ITERS} iters) -- bail check"
  python "$DRIVER" --spectrum-layers "$LAYERS" --cpt-adapter "$CPT_ADAPTER" \
    --config "$CFG" --iters "$DRY_ITERS" \
    --adapter-path "model-experiments/04-cpt-sft/sft_cptv2_probe/adapters/dry-spectrum" 2>&1 | tail -25
  echo ">>> dry-run complete."
  done_mark dry
fi

# --- the actual safety gate. UNCONDITIONAL on ADAPTER_FILE existence, for the
# reason run_sft.sh records: with SKIP_DRY=1 on a fresh start an AND-gate never
# fires and training launches with ZERO confirmation required.
if [ ! -f "$ADAPTER_FILE" ] && [ "${CONFIRM_FULL_RUN:-}" != "1" ]; then
  echo "Self-test and dry-run complete (or skipped). Re-run with CONFIRM_FULL_RUN=1 to start the real multi-hour training run."
  exit 0
fi

TOTAL_ITERS="$(grep -E '^iters:' "$CFG" | grep -oE '[0-9]+' | head -1)"
TOTAL_ITERS="${TOTAL_ITERS:-8200}"
[ -f "$PROGRESS_FILE" ] || echo 0 > "$PROGRESS_FILE"

# NOTE: this does NOT exit -- it falls through to the §8.3 step-5 merge below,
# which has its own done-mark. Exiting here would strand a completed run whose
# merge crashed, and the un-merged artifact is the one that silently loses the
# CPT-v2 delta at eval.
if is_done train && [ -f "$ADAPTER_FILE" ]; then
  echo ">>> training: already complete"
  echo "$TOTAL_ITERS" > "$PROGRESS_FILE"
fi

RESUME_MERGED="$RDIR/.resume_merged.safetensors"
consecutive_fails=0
oom_shrinks=0
attempt_iters_override=""
while true; do
  DONE_STEPS="$(cat "$PROGRESS_FILE")"
  REMAIN=$(( TOTAL_ITERS - DONE_STEPS ))
  if [ "$REMAIN" -le 0 ] && [ -f "$ADAPTER_FILE" ]; then
    done_mark train
    echo "=== cptv2 spectrum SFT training done ($DONE_STEPS/$TOTAL_ITERS) ==="
    break
  fi
  ATTEMPT_ITERS="$REMAIN"
  if [ -n "$attempt_iters_override" ] && [ "$attempt_iters_override" -lt "$REMAIN" ]; then
    ATTEMPT_ITERS="$attempt_iters_override"
  fi
  echo ">>> SFT attempt: requesting ${ATTEMPT_ITERS} iters (${DONE_STEPS}/${TOTAL_ITERS} done, ${REMAIN} remaining)" | tee -a "$TRAIN_LOG"

  BEFORE_CKPTS="$(ls "$ADAPTER"/*_adapters.safetensors 2>/dev/null | xargs -n1 basename 2>/dev/null || true)"
  : > "$RDIR/.segment.log"
  # THE RESUME TRAP, and why this is not just `--resume-adapter-file $ADAPTER_FILE`
  # as the fresh arm does: the CLI flag OVERRIDES the YAML's resume_adapter_file,
  # and mid-training `$ADAPTER_FILE` holds only trainable (= picks) keys. Resuming
  # straight from it would leave the frozen {32..47}\picks blocks at LoRA init --
  # zero delta -- so a single crash+resume would silently drop the CPT-v2 lineage
  # for the remainder of the run and nothing downstream would notice. Resume from
  # a merged file instead: picks' trained state + the frozen CPT-v2 blocks, which
  # is exactly the state the previous segment ended in.
  # Two explicit branches, NOT an empty-array expansion -- macOS bash 3.2 treats
  # "${ARR[@]}" as an unbound-variable error under `set -u` when ARR is empty.
  if [ "$DONE_STEPS" -gt 0 ] && [ -f "$ADAPTER_FILE" ]; then
    rm -f "$RESUME_MERGED"
    python "$MERGER" --in "$ADAPTER_FILE" --out "$RESUME_MERGED" \
      --cpt-adapter "$CPT_ADAPTER" --spectrum-layers "$LAYERS" | tee -a "$TRAIN_LOG"
    python "$DRIVER" --spectrum-layers "$LAYERS" --cpt-adapter "$CPT_ADAPTER" \
      --config "$CFG" --adapter-path "$ADAPTER" \
      --iters "$ATTEMPT_ITERS" --resume-adapter-file "$RESUME_MERGED" >> "$RDIR/.segment.log" 2>&1 &
  else
    python "$DRIVER" --spectrum-layers "$LAYERS" --cpt-adapter "$CPT_ADAPTER" \
      --config "$CFG" --adapter-path "$ADAPTER" \
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

rm -f "$RESUME_MERGED"

# --- §8.3 step 5: put the frozen CPT-v2 blocks back --------------------------
# merge_frozen_keys.py refuses any input that is not exactly the picks, so a
# re-run after a successful merge fails loudly rather than double-merging; the
# done-mark keeps that from happening in the first place.
if ! is_done merge; then
  echo ">>> merging the frozen CPT-v2 blocks back into every saved artifact (§8.3 step 5)"
  {
    python "$MERGER" --in "$ADAPTER_FILE" --out "$ADAPTER_FILE" \
      --cpt-adapter "$CPT_ADAPTER" --spectrum-layers "$LAYERS"
    for CK in "$CKPT_DIR"/*_adapters.safetensors; do
      [ -e "$CK" ] || continue
      echo "-- $(basename "$CK")"
      python "$MERGER" --in "$CK" --out "$CK" \
        --cpt-adapter "$CPT_ADAPTER" --spectrum-layers "$LAYERS"
    done
  } 2>&1 | tee "$RDIR/merge_frozen_keys.txt"
  done_mark merge
fi

echo "=== done. Next: eval_sft_spectrum.sh ==="
