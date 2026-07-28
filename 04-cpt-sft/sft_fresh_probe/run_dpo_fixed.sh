#!/usr/bin/env bash
# ============================================================================
# CONFIRMING TEST for the DPO chat-template fix. This is run_dpo.sh with three
# targeted changes and NOTHING else (diff it -- watchdog, OOM ladder, snapshot
# cadence, subset eval, collapse gate and preflight are byte-identical):
#
#   1. training invoked through ./dpo_fixed_train.py instead of
#      `python -m mlx_lm_lora.train`. That driver rebinds
#      mlx_lm_lora.trainer.datasets.DPODataset to a copy that tokenizes with
#      add_generation_prompt=False, removing the 3 spurious
#      `<|im_start|>assistant\n` tokens the stock class appends AFTER the real
#      <|im_end|> of every chosen AND rejected sequence (verified 161->158 /
#      156->153 tokens on real rows). The installed package is never edited --
#      same call this repo made for CPT-v2 (03-cpt-only/docs/cpt-2/design.md
#      section 4.2).
#   2. all outputs go to NEW paths (adapters/dpo-on-sft-fixed{,-best},
#      results/dpo-fixed/) so the already-analyzed dpo-on-sft results this
#      investigation is about are never touched.
#   3. DPO_ITERS defaults to 60 (3 x 20-iter segments) and early stop is OFF by
#      default -- the question is only "does the functional pass rate still
#      collapse to ~2% by iter 20", which all 3 snapshots answer. This is a
#      confirming test, not the full experiment.
#
# Everything below this banner is run_dpo.sh's original header and logic.
# ----------------------------------------------------------------------------
# DPO-on-SFT-on-fresh-base runner, v1 hyperparams (beta=0.1, lr=1e-6, 250
# iters) per scope decision -- the cptv2 arm's v2 follow-up (stronger
# regularization) still collapsed to the same functional floor, so this arm
# doesn't repeat that rerun. What IS new here, fixing two mistakes the
# cptv2 arm's own report flagged:
#   - val_batches raised 1->10 (FULL-RESULTS.md: n=1 validation reads during
#     training are noise, not a trustworthy checkpoint-selection signal)
#   - functional subset-eval runs INLINE after every snapshot, with an
#     early-stop gate -- v2's real finding was that collapse had already
#     happened by the earliest checkpoint tested (iter 15/250); this arm
#     detects that in-flight instead of discovering it only after a full
#     blind 250-iter run
# Plus watchdog/OOM-ladder/preflight, matching run_sft.sh.
set -euo pipefail
if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu env CAFFEINATED=1 "$0" "$@"
fi
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

need() { command -v "$1" >/dev/null 2>&1 || { echo "MISSING: $1"; exit 1; }; }
need mlx_lm.fuse "pip install mlx-lm"; need jac "pip install jaclang"
python3 -c "import mlx_lm_lora" || { echo "MISSING: mlx-lm-lora (pip install mlx-lm-lora)"; exit 1; }

if pgrep -f "jac start" >/dev/null 2>&1 || pgrep -f "mlx_lm" >/dev/null 2>&1; then
  echo "!!! another jac/mlx_lm process is already running -- stop it first (a competing resident model will OOM a 48GB machine)."
  pgrep -fl "jac start|mlx_lm" || true
  exit 1
fi

DRIVER="model-experiments/04-cpt-sft/sft_fresh_probe/dpo_fixed_train.py"
SFT_ADAPTER="model-experiments/04-cpt-sft/sft_fresh_probe/adapters/sft-on-fresh"
SFT_FUSED="models/sft-fresh-fused-q4"
DPO_ADAPTER="model-experiments/04-cpt-sft/sft_fresh_probe/adapters/dpo-on-sft-fixed"
BEST_ADAPTER="model-experiments/04-cpt-sft/sft_fresh_probe/adapters/dpo-on-sft-fixed-best"
RDIR="model-experiments/04-cpt-sft/sft_fresh_probe/results/dpo-fixed"
SNAP_DIR="$RDIR/snapshots"
DPO_ITERS="${DPO_ITERS:-60}"
DPO_LR="${DPO_LR:-1e-6}"
DPO_BETA="${DPO_BETA:-0.1}"
DPO_MAXLEN="${DPO_MAXLEN:-512}"
DPO_VAL_BATCHES="${DPO_VAL_BATCHES:-10}"
SEGMENT_ITERS="${DPO_SEGMENT_ITERS:-20}"
STALL_SECS="${DPO_STALL_SECS:-900}"
EVAL_SUBSET="${DPO_EVAL_SUBSET:-100}"
COLLAPSE_ABS_FLOOR="${DPO_COLLAPSE_ABS_FLOOR:-30}"   # absolute runs_pct fallback threshold
HOLDOUT="model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl"
DPO_METRICS="$RDIR/metrics_functional.jsonl"

mkdir -p "$RDIR" "$SNAP_DIR"
[ -f "$SFT_ADAPTER/adapters.safetensors" ] || { echo "!!! SFT adapter missing, finish Task 4 first"; exit 1; }
[ -f "$DPO_METRICS" ] || : > "$DPO_METRICS"

echo ">>> fuse SFT adapter -> $SFT_FUSED"
if [ -f "$SFT_FUSED/config.json" ]; then echo "  reuse $SFT_FUSED"; else
  mlx_lm.fuse --model models/qwen-q4 --adapter-path "$SFT_ADAPTER" --save-path "$SFT_FUSED"
fi

# SFT baseline for the collapse gate -- read from Task 5's real result, not
# hardcoded, so this arm's gate always compares against ITS OWN SFT number.
# Filters to full-holdout rows (total==855) and takes the highest step among
# THOSE -- metrics_functional.jsonl also has the base-model row (step 0,
# total 855) and 10 interim subset rows (total 100, sometimes scoring HIGHER
# than the true final on the code_gen-only subset, e.g. this arm's own
# step-4100 interim read 72% vs the true final's 69%) -- a plain max(runs_pct)
# across all rows would silently pick a noisier interim number instead of the
# true final (caught in Task 6's review, inherited from this exact snippet).
SFT_FINAL_PCT="$(python3 -c "
import json
best = 0.0
best_step = -1
try:
    with open('model-experiments/04-cpt-sft/sft_fresh_probe/results/sft/metrics_functional.jsonl') as f:
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
  echo ">>> DPO dry-run (8 iters) -- bail check"
  python "$DRIVER" --model "$SFT_FUSED" --train --data model-experiments/04-cpt-sft/sft_fresh_probe/dataset/dpo \
    --train-mode dpo --config model-experiments/04-cpt-sft/sft_fresh_probe/configs/dpo_lora.yaml \
    --adapter-path model-experiments/04-cpt-sft/sft_fresh_probe/adapters/dpo-dry-fixed \
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
  # Two explicit branches, NOT an empty-array expansion -- macOS's system
  # bash 3.2 treats "${ARR[@]}" as an unbound-variable error under `set -u`
  # when ARR has zero elements. This exact bug was hit LIVE on the SFT
  # runner's real launch attempt (a fresh start always has DONE_STEPS=0,
  # so every attempt crashed instantly) and fixed there the same way --
  # applying the fix here too before this script ever runs for real,
  # instead of waiting to hit it a second time.
  if [ "$DONE_STEPS" -gt 0 ] && [ -f "$DPO_ADAPTER/adapters.safetensors" ]; then
    python "$DRIVER" --model "$SFT_FUSED" --train --data model-experiments/04-cpt-sft/sft_fresh_probe/dataset/dpo \
      --train-mode dpo --config model-experiments/04-cpt-sft/sft_fresh_probe/configs/dpo_lora.yaml \
      --adapter-path "$DPO_ADAPTER" --train-type lora --num-layers 16 --grad-checkpoint \
      --batch-size 1 --max-seq-length "$DPO_MAXLEN" --iters "$SEG_ITERS" \
      --resume-adapter-file "$DPO_ADAPTER/adapters.safetensors" \
      --learning-rate "$DPO_LR" --beta "$DPO_BETA" --dpo-cpo-loss-type sigmoid \
      --steps-per-report 5 --steps-per-eval "$SEG_ITERS" --val-batches "$DPO_VAL_BATCHES" --save-every "$SEG_ITERS" \
      > "$RDIR/.segment.log" 2>&1 &
  else
    python "$DRIVER" --model "$SFT_FUSED" --train --data model-experiments/04-cpt-sft/sft_fresh_probe/dataset/dpo \
      --train-mode dpo --config model-experiments/04-cpt-sft/sft_fresh_probe/configs/dpo_lora.yaml \
      --adapter-path "$DPO_ADAPTER" --train-type lora --num-layers 16 --grad-checkpoint \
      --batch-size 1 --max-seq-length "$DPO_MAXLEN" --iters "$SEG_ITERS" \
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
  JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="$SFT_FUSED" JAC_EVAL_ADAPTER="$SNAP" \
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
    # DPO_DISABLE_EARLY_STOP=1: run the full budget anyway (per user request, to
    # see the whole curve and find the real sweet spot instead of trusting the
    # gate's first trigger point) -- record WHERE the gate would have fired the
    # first time it's crossed, but only actually break the loop when unset.
    if [ ! -f "$RDIR/.would_have_stopped" ]; then
      {
        echo "# DPO gate would have fired at step ${NEW_DONE}/${DPO_ITERS}"
        echo ""
        echo "Two consecutive snapshots scored below the collapse gate (${GATE_VAL}%, ="
        echo "max(${COLLAPSE_ABS_FLOOR}% absolute floor, 50% of this arm's SFT baseline ${SFT_FINAL_PCT}%))."
        echo "Best snapshot seen so far: step $(cat "$RDIR/.best_step" 2>/dev/null || echo "$NEW_DONE") at ${best_pct}%."
      } > "$RDIR/WOULD_HAVE_STOPPED.md"
      touch "$RDIR/.would_have_stopped"
      echo "!!! gate crossed at step ${NEW_DONE} (recorded, not stopping -- DPO_DISABLE_EARLY_STOP or continuing to full budget)" | tee -a "$RDIR/train.log"
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
