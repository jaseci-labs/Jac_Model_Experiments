#!/usr/bin/env bash
# DPO-on-SFT-on-CPT-v2 runner, v2: stronger regularization + early-stopping via
# checkpoint snapshots. The v1 run (results/dpo/) at beta=0.1/lr=1e-6/250 iters
# catastrophically overfit: functional pass rate 72%(SFT)->12%(DPO), train acc
# pinned at 1.000 while val loss diverged 0.705->2.000 the whole run. Real fix
# per that diagnosis: much stronger beta (weaker beta = weaker KL pull toward
# the reference = more reward-hacking room), lower LR, and -- since a FIXED
# iteration count can't be trusted in advance -- snapshot every segment and
# pick the best one by functional eval afterward, instead of guessing a stop
# point up front.
#
# Separate adapter/results dirs from v1 (does not touch or overwrite it).
set -euo pipefail
if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu env CAFFEINATED=1 "$0" "$@"
fi
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

need() { command -v "$1" >/dev/null 2>&1 || { echo "MISSING: $1"; exit 1; }; }
need mlx_lm.fuse "pip install mlx-lm"
python3 -c "import mlx_lm_lora" || { echo "MISSING: mlx-lm-lora (pip install mlx-lm-lora)"; exit 1; }

SFT_ADAPTER="model-experiments/04-cpt-sft/sft_cptv2_probe/adapters/sft-on-cptv2"
SFT_FUSED="models/sft-cptv2-fused-q4"
DPO_ADAPTER="model-experiments/04-cpt-sft/sft_cptv2_probe/adapters/dpo-on-sft-v2"
RDIR="model-experiments/04-cpt-sft/sft_cptv2_probe/results/dpo-v2"
SNAP_DIR="$RDIR/snapshots"
DPO_ITERS="${DPO_ITERS:-150}"        # ceiling; segments + snapshots let us stop earlier by inspection
DPO_LR="${DPO_LR:-5e-7}"             # v1 was 1e-6 -- halved, smaller policy steps
DPO_BETA="${DPO_BETA:-0.4}"          # v1 was 0.1 -- much stronger KL pull toward the SFT reference
DPO_MAXLEN="${DPO_MAXLEN:-512}"      # unchanged from v1, verified safe

mkdir -p "$RDIR" "$SNAP_DIR"
[ -f "$SFT_ADAPTER/adapters.safetensors" ] || { echo "!!! SFT adapter missing, finish Task 3 first"; exit 1; }

echo ">>> fuse SFT adapter -> $SFT_FUSED"
if [ -f "$SFT_FUSED/config.json" ]; then echo "  reuse $SFT_FUSED"; else
  mlx_lm.fuse --model models/qwen-q4 --adapter-path "$SFT_ADAPTER" --save-path "$SFT_FUSED"
fi

DRY_DONE_MARK="$RDIR/.dry.done"
if [ ! -f "$DRY_DONE_MARK" ]; then
  echo ">>> DPO dry-run (8 iters) -- bail check"
  python -m mlx_lm_lora.train --model "$SFT_FUSED" --train --data model-experiments/04-cpt-sft/sft_cptv2_probe/dataset/dpo \
    --train-mode dpo --config model-experiments/04-cpt-sft/sft_cptv2_probe/configs/dpo_lora.yaml \
    --adapter-path model-experiments/04-cpt-sft/sft_cptv2_probe/adapters/dpo-dry-v2 \
    --train-type lora --num-layers 16 --grad-checkpoint --batch-size 1 --max-seq-length "$DPO_MAXLEN" \
    --iters 8 --learning-rate "$DPO_LR" --beta "$DPO_BETA" --dpo-cpo-loss-type sigmoid \
    --steps-per-report 2 --steps-per-eval 100000 --val-batches 1 --save-every 100 2>&1 | tail -25
  touch "$DRY_DONE_MARK"
fi

if [ ! -f "$DPO_ADAPTER/adapters.safetensors" ] && [ "${CONFIRM_FULL_RUN:-}" != "1" ]; then
  echo "Dry-run complete. Re-run with CONFIRM_FULL_RUN=1 to start the real DPO training run."
  exit 0
fi

# Segmented resumable loop (same architecture proven in v1's run_dpo.sh), PLUS
# a snapshot copy of each segment's result keyed by our own trusted progress
# counter (not the library's per-invocation-local checkpoint filenames, which
# is the exact bookkeeping bug v1 hit and fixed).
SEGMENT_ITERS="${DPO_SEGMENT_ITERS:-15}"
PROGRESS_FILE="$RDIR/.dpo_progress_steps"
[ -f "$PROGRESS_FILE" ] || echo 0 > "$PROGRESS_FILE"
: > "$RDIR/train.log"
CONSECUTIVE_FAILS=0
while true; do
  DONE_STEPS="$(cat "$PROGRESS_FILE")"
  REMAIN=$(( DPO_ITERS - DONE_STEPS ))
  if [ "$REMAIN" -le 0 ]; then break; fi
  SEG_ITERS=$(( REMAIN < SEGMENT_ITERS ? REMAIN : SEGMENT_ITERS ))
  echo ">>> DPO training segment: ${SEG_ITERS} iters this process (${DONE_STEPS}/${DPO_ITERS} done, ${REMAIN} remaining overall)" | tee -a "$RDIR/train.log"
  RC=0
  # Two branches (not a possibly-empty array expansion) -- macOS's system bash
  # 3.2 treats "${ARR[@]}" as an unbound-variable error under `set -u` when
  # ARR has zero elements, same gotcha already documented in run_sft.sh.
  if [ "$DONE_STEPS" -gt 0 ] && [ -f "$DPO_ADAPTER/adapters.safetensors" ]; then
    python -m mlx_lm_lora.train --model "$SFT_FUSED" --train --data model-experiments/04-cpt-sft/sft_cptv2_probe/dataset/dpo \
      --train-mode dpo --config model-experiments/04-cpt-sft/sft_cptv2_probe/configs/dpo_lora.yaml \
      --adapter-path "$DPO_ADAPTER" --train-type lora --num-layers 16 --grad-checkpoint \
      --batch-size 1 --max-seq-length "$DPO_MAXLEN" --iters "$SEG_ITERS" \
      --resume-adapter-file "$DPO_ADAPTER/adapters.safetensors" \
      --learning-rate "$DPO_LR" --beta "$DPO_BETA" --dpo-cpo-loss-type sigmoid \
      --steps-per-report 5 --steps-per-eval "$SEG_ITERS" --val-batches 1 --save-every "$SEG_ITERS" \
      >> "$RDIR/train.log" 2>&1 || RC=$?
  else
    python -m mlx_lm_lora.train --model "$SFT_FUSED" --train --data model-experiments/04-cpt-sft/sft_cptv2_probe/dataset/dpo \
      --train-mode dpo --config model-experiments/04-cpt-sft/sft_cptv2_probe/configs/dpo_lora.yaml \
      --adapter-path "$DPO_ADAPTER" --train-type lora --num-layers 16 --grad-checkpoint \
      --batch-size 1 --max-seq-length "$DPO_MAXLEN" --iters "$SEG_ITERS" \
      --learning-rate "$DPO_LR" --beta "$DPO_BETA" --dpo-cpo-loss-type sigmoid \
      --steps-per-report 5 --steps-per-eval "$SEG_ITERS" --val-batches 1 --save-every "$SEG_ITERS" \
      >> "$RDIR/train.log" 2>&1 || RC=$?
  fi
  if [ "$RC" -ne 0 ]; then
    CONSECUTIVE_FAILS=$(( CONSECUTIVE_FAILS + 1 ))
    if [ "$CONSECUTIVE_FAILS" -ge 5 ]; then
      echo "!!! segment crashed ${CONSECUTIVE_FAILS}x in a row at the same ${DONE_STEPS}/${DPO_ITERS} point -- giving up." | tee -a "$RDIR/train.log"
      tail -20 "$RDIR/train.log"; exit 1
    fi
    echo "!!! segment crashed (exit $RC, attempt ${CONSECUTIVE_FAILS}/5) -- retrying same segment from last good state" | tee -a "$RDIR/train.log"
    continue
  fi
  CONSECUTIVE_FAILS=0
  NEW_DONE=$(( DONE_STEPS + SEG_ITERS ))
  echo "$NEW_DONE" > "$PROGRESS_FILE"
  STEP_TAG="$(printf '%04d' "$NEW_DONE")"
  mkdir -p "$SNAP_DIR/step_${STEP_TAG}"
  cp "$DPO_ADAPTER/adapters.safetensors" "$SNAP_DIR/step_${STEP_TAG}/adapters.safetensors"
  cp "$DPO_ADAPTER/adapter_config.json" "$SNAP_DIR/step_${STEP_TAG}/adapter_config.json"
  echo "  snapshot saved: $SNAP_DIR/step_${STEP_TAG}/" | tee -a "$RDIR/train.log"
done
echo "=== DPO v2 training done: $RDIR/train.log ==="
