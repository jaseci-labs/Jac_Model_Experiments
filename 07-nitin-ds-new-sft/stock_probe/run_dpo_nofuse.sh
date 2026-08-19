#!/usr/bin/env bash
# ============================================================================
# ROOT-CAUSE FIX for the DPO collapse (12% -> ~2% across every prior variant,
# including the chat-template fix in run_dpo_fixed.sh). Diagnosed by reading
# RAW generations from a "fixed" run's step_0020 snapshot: it output plain
# React/JSX, not Jac -- identical to the UNTOUCHED base model's own output on
# the same prompt. Traced to `mlx_lm.fuse` on an int4-quantized model: fusing
# the SFT LoRA delta into 4-bit-quantized weights re-quantizes afterward, and
# the SFT delta is too fine-grained to survive that -- ~15% of packed weight
# elements changed bit-pattern (not a no-op at the byte level) but the
# resulting fused-model-with-no-adapter generates functionally identical to
# raw base (verified side by side). Every DPO run to date (both arms, every
# hyperparameter variant, the chat-template fix) trained on top of this
# silently-reverted "SFT" base -- nothing to do with add_generation_prompt,
# beta, or lr.
#
# THE FIX: never fuse before DPO. Train directly against models/qwen-q4
# (raw, quantized, untouched) and seed DPO's LoRA from the SFT adapter via
# --resume-adapter-file instead of baking it into weights first. Verified:
# a manual 20-iter test this way scored 73% subset functional pass rate
# (beats the 69% SFT baseline) with val_accuracy 0.888 and real reward
# magnitudes -- vs the fused path's permanent ~2% collapse and chance-level
# (~0.5) val_accuracy.
#
# Byte-identical to run_dpo_fixed.sh otherwise (still uses the chat-template
# fix via dpo_fixed_train.jac -- both fixes are real and both are kept):
#   - no fuse step, no SFT_FUSED var -- BASE_MODEL=models/qwen-q4 throughout
#   - dry-run and the first real segment both pass
#     --resume-adapter-file "$SFT_ADAPTER/adapters.safetensors" to seed from
#     the SFT-trained LoRA instead of starting DPO from scratch
#   - new output paths (adapters/dpo-on-sft-nitin-nofuse{,-best}, results/dpo-nofuse/)
#     so no prior (broken) result is touched
# ----------------------------------------------------------------------------
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
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

need() { command -v "$1" >/dev/null 2>&1 || { echo "MISSING: $1"; exit 1; }; }
need jac "pip install jaclang"
python3 -c "import mlx_lm_lora" || { echo "MISSING: mlx-lm-lora (pip install mlx-lm-lora)"; exit 1; }

if pgrep -f "jac start" >/dev/null 2>&1 || pgrep -f "mlx_lm" >/dev/null 2>&1; then
  echo "!!! another jac/mlx_lm process is already running -- stop it first (a competing resident model will OOM a 48GB machine)."
  pgrep -fl "jac start|mlx_lm" || true
  exit 1
fi

DRIVER="model-experiments/07-nitin-ds-new-sft/stock_probe/dpo_fixed_train.jac"
BASE_MODEL="models/qwen-q4"
SFT_ADAPTER="model-experiments/07-nitin-ds-new-sft/stock_probe/adapters/sft-on-nitin"
DPO_ADAPTER="model-experiments/07-nitin-ds-new-sft/stock_probe/adapters/dpo-on-sft-nitin-nofuse"
BEST_ADAPTER="model-experiments/07-nitin-ds-new-sft/stock_probe/adapters/dpo-on-sft-nitin-nofuse-best"
RDIR="model-experiments/07-nitin-ds-new-sft/stock_probe/results/dpo-nofuse"
SNAP_DIR="$RDIR/snapshots"
DPO_ITERS="${DPO_ITERS:-250}"
DPO_LR="${DPO_LR:-1e-6}"
DPO_BETA="${DPO_BETA:-0.1}"
DPO_MAXLEN="${DPO_MAXLEN:-512}"
DPO_VAL_BATCHES="${DPO_VAL_BATCHES:-10}"
SEGMENT_ITERS="${DPO_SEGMENT_ITERS:-20}"
STALL_SECS="${DPO_STALL_SECS:-900}"
EVAL_SUBSET="${DPO_EVAL_SUBSET:-100}"
COLLAPSE_ABS_FLOOR="${DPO_COLLAPSE_ABS_FLOOR:-30}"
HOLDOUT="${HOLDOUT:-model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl}"
DPO_METRICS="$RDIR/metrics_functional.jsonl"

mkdir -p "$RDIR" "$SNAP_DIR"
[ -f "$SFT_ADAPTER/adapters.safetensors" ] || { echo "!!! SFT adapter missing"; exit 1; }
[ -f "$DPO_METRICS" ] || : > "$DPO_METRICS"

# NOTE (07-nitin-new retarget): the lookup below filters on `total == 855`, i.e. it
# only recognises an SFT baseline scored on holdout (a) (the shared 855-row
# code-graded set), which is what HOLDOUT above defaults to. If HOLDOUT is
# overridden to the Nitin holdout for a training run, this returns 0.0 and the
# collapse gate silently degrades to the COLLAPSE_ABS_FLOOR (30%) alone. Left
# verbatim on purpose -- the default path is the replication path.
SFT_FINAL_PCT="$(python3 -c "
import json
best = 0.0
best_step = -1
try:
    with open('model-experiments/07-nitin-ds-new-sft/stock_probe/results/sft/metrics_functional.jsonl') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{'): continue
            r = json.loads(line)
            if r.get('category') == '__overall__' and r.get('total') == 855:
                if r.get('step', -1) > best_step:
                    best_step = r.get('step', -1)
                    best = float(r.get('runs_pct', 0))
except FileNotFoundError:
    pass
print(best)
")"
COLLAPSE_REL_THRESHOLD="$(python3 -c "print(round(float('${SFT_FINAL_PCT}') * 0.5, 1))")"
echo ">>> SFT baseline: ${SFT_FINAL_PCT}% -- collapse gate fires at 2 consecutive snapshots below max(${COLLAPSE_ABS_FLOOR}%, ${COLLAPSE_REL_THRESHOLD}%)"

DRY_DONE_MARK="$RDIR/.dry.done"
if [ ! -f "$DRY_DONE_MARK" ]; then
  echo ">>> DPO dry-run (8 iters), seeded from SFT adapter, no fuse -- bail check"
  jac run "$DRIVER" --model "$BASE_MODEL" --train --data model-experiments/07-nitin-ds-new-sft/dataset/dpo \
    --train-mode dpo --config model-experiments/07-nitin-ds-new-sft/stock_probe/configs/dpo_lora.yaml \
    --adapter-path model-experiments/07-nitin-ds-new-sft/stock_probe/adapters/dpo-dry-nofuse \
    --resume-adapter-file "$SFT_ADAPTER/adapters.safetensors" \
    --train-type lora --num-layers 16 --grad-checkpoint --batch-size 1 --max-seq-length "$DPO_MAXLEN" \
    --iters 8 --learning-rate "$DPO_LR" --beta "$DPO_BETA" --dpo-cpo-loss-type sigmoid \
    --steps-per-report 2 --steps-per-eval 100000 --val-batches "$DPO_VAL_BATCHES" --save-every 100 2>&1 | tail -25
  touch "$DRY_DONE_MARK"
fi

if [ ! -f "$DPO_ADAPTER/adapters.safetensors" ] && [ "${CONFIRM_FULL_RUN:-}" != "1" ]; then
  echo "Dry-run complete. Re-run with CONFIRM_FULL_RUN=1 to start the real DPO training run."
  exit 0
fi

PROGRESS_FILE="$RDIR/.dpo_progress_steps"
[ -f "$PROGRESS_FILE" ] || echo 0 > "$PROGRESS_FILE"
[ -f "$RDIR/train.log" ] || : > "$RDIR/train.log"
consecutive_fails=0
oom_shrinks=0
low_streak=0
best_pct=-1.0
while true; do
  DONE_STEPS="$(cat "$PROGRESS_FILE")"
  REMAIN=$(( DPO_ITERS - DONE_STEPS ))
  if [ "$REMAIN" -le 0 ]; then echo "=== DPO reached full ${DPO_ITERS} iters ===" | tee -a "$RDIR/train.log"; break; fi
  SEG_ITERS=$(( REMAIN < SEGMENT_ITERS ? REMAIN : SEGMENT_ITERS ))
  echo ">>> DPO segment: ${SEG_ITERS} iters (${DONE_STEPS}/${DPO_ITERS} done)" | tee -a "$RDIR/train.log"

  : > "$RDIR/.segment.log"
  if [ "$DONE_STEPS" -gt 0 ] && [ -f "$DPO_ADAPTER/adapters.safetensors" ]; then
    jac run "$DRIVER" --model "$BASE_MODEL" --train --data model-experiments/07-nitin-ds-new-sft/dataset/dpo \
      --train-mode dpo --config model-experiments/07-nitin-ds-new-sft/stock_probe/configs/dpo_lora.yaml \
      --adapter-path "$DPO_ADAPTER" --train-type lora --num-layers 16 --grad-checkpoint \
      --batch-size 1 --max-seq-length "$DPO_MAXLEN" --iters "$SEG_ITERS" \
      --resume-adapter-file "$DPO_ADAPTER/adapters.safetensors" \
      --learning-rate "$DPO_LR" --beta "$DPO_BETA" --dpo-cpo-loss-type sigmoid \
      --steps-per-report 5 --steps-per-eval "$SEG_ITERS" --val-batches "$DPO_VAL_BATCHES" --save-every "$SEG_ITERS" \
      > "$RDIR/.segment.log" 2>&1 &
  else
    # First real segment -- seed from the SFT adapter, not from scratch.
    jac run "$DRIVER" --model "$BASE_MODEL" --train --data model-experiments/07-nitin-ds-new-sft/dataset/dpo \
      --train-mode dpo --config model-experiments/07-nitin-ds-new-sft/stock_probe/configs/dpo_lora.yaml \
      --adapter-path "$DPO_ADAPTER" --train-type lora --num-layers 16 --grad-checkpoint \
      --batch-size 1 --max-seq-length "$DPO_MAXLEN" --iters "$SEG_ITERS" \
      --resume-adapter-file "$SFT_ADAPTER/adapters.safetensors" \
      --learning-rate "$DPO_LR" --beta "$DPO_BETA" --dpo-cpo-loss-type sigmoid \
      --steps-per-report 5 --steps-per-eval "$SEG_ITERS" --val-batches "$DPO_VAL_BATCHES" --save-every "$SEG_ITERS" \
      > "$RDIR/.segment.log" 2>&1 &
  fi
  SEG_PID=$!

  last_growth=$(date +%s); last_size=0
  while kill -0 "$SEG_PID" 2>/dev/null; do
    sleep 30
    cur_size="$(wc -l < "$RDIR/.segment.log" 2>/dev/null || echo 0)"; now="$(date +%s)"
    if [ "$cur_size" -gt "$last_size" ]; then last_size="$cur_size"; last_growth="$now"; fi
    if [ $(( now - last_growth )) -ge "$STALL_SECS" ]; then
      echo "!!! stalled: no log growth for ${STALL_SECS}s -- killing PID $SEG_PID" | tee -a "$RDIR/train.log"
      kill -9 "$SEG_PID" 2>/dev/null || true
      break
    fi
  done
  RC=0; wait "$SEG_PID" 2>/dev/null || RC=$?
  cat "$RDIR/.segment.log" >> "$RDIR/train.log"

  if [ "$RC" -ne 0 ]; then
    consecutive_fails=$(( consecutive_fails + 1 ))
    if grep -qEi "out of memory|OutOfMemory|kIOGPUCommandBuffer|MTL::.*(OOM|Insufficient)" "$RDIR/.segment.log"; then
      echo "!!! OOM signature detected" | tee -a "$RDIR/train.log"
      if [ "$oom_shrinks" -lt 3 ]; then
        oom_shrinks=$(( oom_shrinks + 1 ))
        if [ "$oom_shrinks" -le 2 ]; then
          SEGMENT_ITERS=$(( SEGMENT_ITERS / 2 )); [ "$SEGMENT_ITERS" -lt 5 ] && SEGMENT_ITERS=5
          echo "!!! OOM-recovery ${oom_shrinks}/3: shrinking DPO_SEGMENT_ITERS to ${SEGMENT_ITERS}" | tee -a "$RDIR/train.log"
        else
          DPO_MAXLEN=384
          echo "!!! OOM-recovery 3/3: dropping DPO_MAXLEN to ${DPO_MAXLEN} (last resort -- flag this deviation in the final report)" | tee -a "$RDIR/train.log"
        fi
      else
        echo "!!! OOM persisted through the full shrink ladder -- giving up." | tee -a "$RDIR/train.log"
        tail -20 "$RDIR/train.log"; exit 1
      fi
    fi
    if [ "$consecutive_fails" -ge 5 ]; then
      echo "!!! segment failed 5x in a row -- giving up." | tee -a "$RDIR/train.log"
      tail -20 "$RDIR/train.log"; exit 1
    fi
    continue
  fi
  consecutive_fails=0

  NEW_DONE=$(( DONE_STEPS + SEG_ITERS ))
  echo "$NEW_DONE" > "$PROGRESS_FILE"
  STEP_TAG="$(printf '%04d' "$NEW_DONE")"
  SNAP="$SNAP_DIR/step_${STEP_TAG}"
  mkdir -p "$SNAP"
  cp "$DPO_ADAPTER/adapters.safetensors" "$SNAP/adapters.safetensors"
  cp "$DPO_ADAPTER/adapter_config.json" "$SNAP/adapter_config.json"
  echo "  snapshot saved: $SNAP" | tee -a "$RDIR/train.log"

  echo ">>> subset functional eval on snapshot step ${NEW_DONE} (n=${EVAL_SUBSET})"
  JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="$BASE_MODEL" JAC_EVAL_ADAPTER="$SNAP" \
    JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_LIMIT="$EVAL_SUBSET" JAC_EVAL_METRICS_OUT="$DPO_METRICS" JAC_EVAL_STEP="$NEW_DONE" \
    jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac 2>&1 | tail -5 | tee -a "$RDIR/train.log"
  STEP_PCT="$(python3 -c "
import json
p = 0.0
try:
    with open('$DPO_METRICS') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{'): continue
            r = json.loads(line)
            if r.get('category') == '__overall__' and r.get('step') == $NEW_DONE:
                p = float(r.get('runs_pct', 0))
except Exception:
    pass
print(p)
")"
  echo "  step ${NEW_DONE} subset runs_pct: ${STEP_PCT}%" | tee -a "$RDIR/train.log"

  IS_BEST="$(python3 -c "print(1 if ${STEP_PCT} > ${best_pct} else 0)")"
  if [ "$IS_BEST" = "1" ]; then
    best_pct="$STEP_PCT"
    rm -rf "$BEST_ADAPTER"; mkdir -p "$BEST_ADAPTER"
    cp "$SNAP/adapters.safetensors" "$BEST_ADAPTER/adapters.safetensors"
    cp "$SNAP/adapter_config.json" "$BEST_ADAPTER/adapter_config.json"
    echo "$NEW_DONE" > "$RDIR/.best_step"
    echo "  new best: step ${NEW_DONE} (${STEP_PCT}%)" | tee -a "$RDIR/train.log"
  fi

  BELOW_GATE="$(python3 -c "print(1 if ${STEP_PCT} < max(${COLLAPSE_ABS_FLOOR}, ${COLLAPSE_REL_THRESHOLD}) else 0)")"
  if [ "$BELOW_GATE" = "1" ]; then low_streak=$(( low_streak + 1 )); else low_streak=0; fi
  if [ "$low_streak" -ge 2 ]; then
    GATE_VAL="$(python3 -c "print(max(${COLLAPSE_ABS_FLOOR}, ${COLLAPSE_REL_THRESHOLD}))")"
    if [ ! -f "$RDIR/.would_have_stopped" ]; then
      {
        echo "# DPO gate would have fired at step ${NEW_DONE}/${DPO_ITERS}"
        echo ""
        echo "Two consecutive snapshots scored below the collapse gate (${GATE_VAL}%, ="
        echo "max(${COLLAPSE_ABS_FLOOR}% absolute floor, 50% of this arm's SFT baseline ${SFT_FINAL_PCT}%))."
        echo "Best snapshot seen so far: step $(cat "$RDIR/.best_step" 2>/dev/null || echo "$NEW_DONE") at ${best_pct}%."
      } > "$RDIR/WOULD_HAVE_STOPPED.md"
      touch "$RDIR/.would_have_stopped"
      echo "!!! gate crossed at step ${NEW_DONE} (recorded, not stopping -- continuing to full budget)" | tee -a "$RDIR/train.log"
    fi
    if [ "${DPO_DISABLE_EARLY_STOP:-1}" != "1" ]; then
      {
        echo "# DPO early-stopped at step ${NEW_DONE}/${DPO_ITERS}"
        echo ""
        echo "Two consecutive snapshots scored below the collapse gate (${GATE_VAL}%, ="
        echo "max(${COLLAPSE_ABS_FLOOR}% absolute floor, 50% of this arm's SFT baseline ${SFT_FINAL_PCT}%))."
        echo "Best snapshot seen: step $(cat "$RDIR/.best_step" 2>/dev/null || echo "$NEW_DONE") at ${best_pct}%."
        echo "Stopped early instead of burning the remaining $(( DPO_ITERS - NEW_DONE )) iters on an already-collapsed policy."
      } > "$RDIR/EARLY_STOP.md"
      touch "$RDIR/.early_stopped"
      echo "!!! EARLY STOP: 2 consecutive snapshots below collapse gate. See $RDIR/EARLY_STOP.md" | tee -a "$RDIR/train.log"
      break
    fi
  fi
done
echo "=== DPO training loop done: $RDIR/train.log (best snapshot: step $(cat "$RDIR/.best_step" 2>/dev/null || echo N/A) at ${best_pct}%) ==="
