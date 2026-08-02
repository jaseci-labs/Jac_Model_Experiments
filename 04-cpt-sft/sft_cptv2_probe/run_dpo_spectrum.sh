#!/usr/bin/env bash
# ============================================================================
# DPO on the Spectrum-picked layers -- cptv2 arm.
#
# Same one-variable change as the fresh arm's run_dpo_spectrum.sh (which 16
# blocks carry the LoRA), on top of the same corrected no-fuse recipe
# (comparison report §3: BASE_MODEL=models/qwen-q4, --resume-adapter-file, never
# mlx_lm.fuse, beta 0.1, lr 1e-6, 654 pairs, 250 iters, chat-template fix).
#
# PLUS the union/freeze/merge discipline run_sft_spectrum.sh introduced, because
# DPO continues the SAME LoRA lineage and that lineage is 21 blocks, not 16:
#
#   * The SFT-spectrum adapter this run resumes from covers the union
#     `picks | {32..47}` = 336 keys (16 trained picks + 5 frozen CPT-v2 blocks
#     merged back by §8.3 step 5). mlx_lm_lora/train.py:527 applies
#     --resume-adapter-file with strict=False AFTER conversion, so a picks-only
#     DPO conversion would silently drop those 80 frozen tensors -- exactly the
#     bug the SFT side already fixed, arriving one stage later. The driver
#     cptv2_dpo_spectrum_train.py converts the union and freezes
#     `{32..47} \ picks`, keeping the trainable budget at 281.838M.
#   * mlx_lm_lora/trainer/dpo_trainer.py:672-685 saves only
#     model.trainable_parameters(), so every file DPO writes is picks-only again.
#     merge_frozen_keys.py puts the frozen blocks back into each snapshot BEFORE
#     it is scored, into the resume file between segments, and into the final
#     adapter -- the same three places run_sft_spectrum.sh handles.
#   * THE RESUME TRAP, same as SFT's: mid-training $DPO_ADAPTER/adapters.safetensors
#     holds only the picks. Resuming straight from it would leave the frozen
#     blocks at LoRA init (zero delta), so one crash+resume would silently drop
#     the CPT-v2 lineage for the rest of the run. Resume from a merged file.
#
# Gates before any compute: --verify-patches (both monkey-patches, three call
# sites), --verify-resume (the SFT artifact is union-shaped AND its frozen blocks
# are bit-identical to the CPT-v2 source -- proof the freeze survived SFT), and
# --verify-layers (the real 30B model through the real DPO conversion path).
#
# Outputs: adapters/dpo-on-sft-cptv2-spectrum{,-best}, results/dpo-spectrum/.
#
# LAUNCH ONLY AFTER this arm's run_sft_spectrum.sh AND eval_sft_spectrum.sh have
# completed -- and note spectrum-plan.md §12 makes this whole arm CONDITIONAL on
# the Phase-1 gate recorded in the fresh arm's results/sft-spectrum/GATE.md.
# ----------------------------------------------------------------------------
set -euo pipefail
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

SPEC_DIR="model-experiments/04-cpt-sft/sft_cptv2_probe/spectrum"
FRESH_SPEC_DIR="model-experiments/04-cpt-sft/sft_fresh_probe/spectrum"
DRIVER="$SPEC_DIR/cptv2_dpo_spectrum_train.py"
MERGER="$SPEC_DIR/merge_frozen_keys.py"
CFGFIX="$FRESH_SPEC_DIR/adapter_config_fix.py"
LAYERS="$SPEC_DIR/configs/spectrum_layers.json"
CPT_ADAPTER="model-experiments/03-cpt-only/adapters/cpt-v2/adapters.safetensors"
BASE_MODEL="models/qwen-q4"
SFT_ADAPTER="model-experiments/04-cpt-sft/sft_cptv2_probe/adapters/sft-on-cptv2-spectrum"
DPO_ADAPTER="model-experiments/04-cpt-sft/sft_cptv2_probe/adapters/dpo-on-sft-cptv2-spectrum"
BEST_ADAPTER="model-experiments/04-cpt-sft/sft_cptv2_probe/adapters/dpo-on-sft-cptv2-spectrum-best"
RDIR="model-experiments/04-cpt-sft/sft_cptv2_probe/results/dpo-spectrum"
SNAP_DIR="$RDIR/snapshots"
UNION="$RDIR/union_layers.json"
RESUME_MERGED="$RDIR/.resume_merged.safetensors"
DPO_ITERS="${DPO_ITERS:-250}"
DPO_LR="${DPO_LR:-1e-6}"
DPO_BETA="${DPO_BETA:-0.1}"
DPO_MAXLEN="${DPO_MAXLEN:-512}"
DPO_VAL_BATCHES="${DPO_VAL_BATCHES:-10}"
SEGMENT_ITERS="${DPO_SEGMENT_ITERS:-20}"
STALL_SECS="${DPO_STALL_SECS:-900}"
EVAL_SUBSET="${DPO_EVAL_SUBSET:-100}"
COLLAPSE_ABS_FLOOR="${DPO_COLLAPSE_ABS_FLOOR:-30}"
HOLDOUT="model-experiments/04-cpt-sft/sft_cptv2_probe/dataset/sft/valid.jsonl"
DPO_METRICS="$RDIR/metrics_functional.jsonl"
DATA="model-experiments/04-cpt-sft/sft_cptv2_probe/dataset/dpo"
DPO_CFG="model-experiments/04-cpt-sft/sft_cptv2_probe/configs/dpo_lora.yaml"

mkdir -p "$RDIR" "$SNAP_DIR"
for f in "$DRIVER" "$MERGER" "$CFGFIX" "$LAYERS" "$CPT_ADAPTER" "$DPO_CFG" \
         "$DATA/train.jsonl" "$HOLDOUT"; do
  [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }
done
if [ ! -f "$SFT_ADAPTER/adapters.safetensors" ]; then
  echo "!!! MISSING: $SFT_ADAPTER/adapters.safetensors"
  echo "    Run this arm's run_sft_spectrum.sh (including its §8.3 step-5 merge)"
  echo "    and eval_sft_spectrum.sh first."
  exit 1
fi
if [ ! -f "model-experiments/04-cpt-sft/sft_cptv2_probe/results/sft-spectrum/.merge.done" ]; then
  echo "!!! the SFT-spectrum §8.3 step-5 merge has not run -- $SFT_ADAPTER holds"
  echo "    ONLY the trainable picks, so the CPT-v2 delta is missing and DPO would"
  echo "    start from a lineage that already lost it. Finish run_sft_spectrum.sh."
  exit 1
fi
[ -f "$DPO_METRICS" ] || : > "$DPO_METRICS"

done_mark() { touch "$RDIR/.$1.done"; }
is_done() { [ -f "$RDIR/.$1.done" ]; }

# --- the union, derived from the real CPT-v2 file (not from a doc claim) ----
echo ">>> deriving the union picks | {32..47} -- what this arm's adapters cover"
PYTHONPATH="$SPEC_DIR:$FRESH_SPEC_DIR:${PYTHONPATH:-}" python - "$LAYERS" "$UNION" <<'PY' | tee "$RDIR/union.txt"
import json, sys
from cptv2_spectrum_lora_layers import assert_cpt_v2_shape, union_layers, frozen_layers
picks = json.loads(open(sys.argv[1]).read())["layers"]
base = assert_cpt_v2_shape()
u, fz = union_layers(picks, base), frozen_layers(picks, base)
json.dump({"layers": u, "count": len(u), "picks": picks,
           "cpt_v2_layers": base, "frozen_layers": fz}, open(sys.argv[2], "w"), indent=2)
print(f"picks({len(picks)})={picks}")
print(f"union({len(u)})={u}")
print(f"frozen({len(fz)})={fz}")
PY

# --- gate 1: both patches live, all three sites, union + freeze (seconds) ---
if ! is_done verify_patches; then
  echo ">>> --verify-patches: chat-template fix AND union layer rebind, all three call sites"
  python "$DRIVER" --verify-patches --spectrum-layers "$LAYERS" --cpt-adapter "$CPT_ADAPTER" \
    2>&1 | tee "$RDIR/verify_patches.txt"
  grep -q "^VERIFY: PASS" "$RDIR/verify_patches.txt" || {
    echo "!!! patch composition FAILED -- see $RDIR/verify_patches.txt. Not training."; exit 1; }
  done_mark verify_patches
fi

# --- gate 2: the SFT artifact still carries the frozen CPT-v2 blocks --------
# The single most important preflight on this arm: proves the freeze survived
# the SFT stage byte-for-byte, before DPO inherits it.
if ! is_done verify_resume; then
  echo ">>> --verify-resume: the SFT-spectrum adapter is union-shaped and its frozen blocks bit-match CPT-v2"
  python "$DRIVER" --verify-resume "$SFT_ADAPTER/adapters.safetensors" \
    --spectrum-layers "$LAYERS" --cpt-adapter "$CPT_ADAPTER" 2>&1 | tee "$RDIR/verify_resume.txt"
  grep -q "^VERIFY: PASS" "$RDIR/verify_resume.txt" || {
    echo "!!! the SFT-spectrum lineage did NOT survive -- see $RDIR/verify_resume.txt. Not training."; exit 1; }
  done_mark verify_resume
fi

# --- gate 3: the real conversion path on the real model (minutes) -----------
if ! is_done verify_layers && [ "${SKIP_VERIFY:-0}" != "1" ]; then
  echo ">>> --verify-layers self-test (loads models/qwen-q4 twice; a few minutes)"
  python "$DRIVER" --verify-layers --spectrum-layers "$LAYERS" --cpt-adapter "$CPT_ADAPTER" \
    --model "$BASE_MODEL" --resume-adapter-file "$SFT_ADAPTER/adapters.safetensors" \
    2>&1 | tee "$RDIR/verify_layers.txt"
  grep -q "^VERIFY: PASS" "$RDIR/verify_layers.txt" || {
    echo "!!! self-test FAILED -- see $RDIR/verify_layers.txt. Not training."; exit 1; }
  done_mark verify_layers
fi

SFT_FINAL_PCT="$(python3 -c "
import json
best = 0.0
best_step = -1
try:
    with open('model-experiments/04-cpt-sft/sft_cptv2_probe/results/sft-spectrum/metrics_functional.jsonl') as f:
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
echo ">>> SFT-spectrum baseline: ${SFT_FINAL_PCT}% -- collapse gate fires at 2 consecutive snapshots below max(${COLLAPSE_ABS_FLOOR}%, ${COLLAPSE_REL_THRESHOLD}%)"

DRY_DONE_MARK="$RDIR/.dry.done"
if [ ! -f "$DRY_DONE_MARK" ]; then
  echo ">>> DPO dry-run (8 iters), seeded from the SFT-spectrum adapter, no fuse -- bail check"
  python "$DRIVER" --spectrum-layers "$LAYERS" --cpt-adapter "$CPT_ADAPTER" \
    --model "$BASE_MODEL" --train --data "$DATA" \
    --train-mode dpo --config "$DPO_CFG" \
    --adapter-path model-experiments/04-cpt-sft/sft_cptv2_probe/adapters/dpo-dry-spectrum \
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

  # After the final §8.3 step-5 merge below, $DPO_ADAPTER is union-shaped and is
  # no longer a valid mid-training resume source. Say so plainly instead of
  # letting merge_frozen_keys.py fail with a "covers layers" message that reads
  # like a training bug.
  if [ "$DONE_STEPS" -gt 0 ] && is_done merge; then
    echo "!!! $DPO_ADAPTER/adapters.safetensors has already had the final merge applied"
    echo "    (union-shaped, $RDIR/.merge.done exists), so it cannot be resumed from."
    echo "    This run reached ${DONE_STEPS}/${DPO_ITERS} iters. To extend it, restart"
    echo "    from a snapshot under $SNAP_DIR (those are merged AND scored) into a NEW"
    echo "    adapter path -- do not re-merge this one."
    exit 1
  fi

  : > "$RDIR/.segment.log"
  # THE RESUME TRAP (see header): mid-training $DPO_ADAPTER holds only the picks,
  # so resume from picks + frozen CPT-v2 blocks merged, which is the state the
  # previous segment actually ended in.
  # Two explicit branches, NOT an empty-array expansion -- macOS bash 3.2 treats
  # "${ARR[@]}" as an unbound-variable error under `set -u` when ARR is empty.
  if [ "$DONE_STEPS" -gt 0 ] && [ -f "$DPO_ADAPTER/adapters.safetensors" ]; then
    rm -f "$RESUME_MERGED"
    python "$MERGER" --in "$DPO_ADAPTER/adapters.safetensors" --out "$RESUME_MERGED" \
      --cpt-adapter "$CPT_ADAPTER" --spectrum-layers "$LAYERS" | tee -a "$RDIR/train.log"
    python "$DRIVER" --spectrum-layers "$LAYERS" --cpt-adapter "$CPT_ADAPTER" \
      --model "$BASE_MODEL" --train --data "$DATA" \
      --train-mode dpo --config "$DPO_CFG" \
      --adapter-path "$DPO_ADAPTER" --train-type lora --num-layers 16 --grad-checkpoint \
      --batch-size 1 --max-seq-length "$DPO_MAXLEN" --iters "$SEG_ITERS" \
      --resume-adapter-file "$RESUME_MERGED" \
      --learning-rate "$DPO_LR" --beta "$DPO_BETA" --dpo-cpo-loss-type sigmoid \
      --steps-per-report 5 --steps-per-eval "$SEG_ITERS" --val-batches "$DPO_VAL_BATCHES" --save-every "$SEG_ITERS" \
      > "$RDIR/.segment.log" 2>&1 &
  else
    # First real segment -- seed from the SFT-spectrum adapter, which is already
    # union-shaped and was verified bit-identical to CPT-v2 on its frozen blocks.
    python "$DRIVER" --spectrum-layers "$LAYERS" --cpt-adapter "$CPT_ADAPTER" \
      --model "$BASE_MODEL" --train --data "$DATA" \
      --train-mode dpo --config "$DPO_CFG" \
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
  # §8.3 step 5 on the snapshot: the trainer wrote picks only, so put the frozen
  # CPT-v2 blocks back BEFORE this snapshot is scored, copied to -best, or kept.
  python "$MERGER" --in "$DPO_ADAPTER/adapters.safetensors" --out "$SNAP/adapters.safetensors" \
    --cpt-adapter "$CPT_ADAPTER" --spectrum-layers "$LAYERS" | tee -a "$RDIR/train.log"
  cp "$DPO_ADAPTER/adapter_config.json" "$SNAP/adapter_config.json"
  # §7 on the snapshot, over the UNION (that is what the merged file covers).
  python "$CFGFIX" --adapter-dir "$SNAP" --spectrum-layers "$UNION" \
    >> "$RDIR/adapter_config_rewrite.txt" 2>&1
  echo "  snapshot merged + adapter_config rewritten (union): $SNAP" | tee -a "$RDIR/train.log"

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
    cp "$SNAP/adapters.safetensors" "$BEST_ADAPTER/adapters.safetensors"   # already merged
    cp "$SNAP/adapter_config.json" "$BEST_ADAPTER/adapter_config.json"     # already rewritten
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
        echo "max(${COLLAPSE_ABS_FLOOR}% absolute floor, 50% of this arm's SFT-spectrum baseline ${SFT_FINAL_PCT}%))."
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
        echo "max(${COLLAPSE_ABS_FLOOR}% absolute floor, 50% of this arm's SFT-spectrum baseline ${SFT_FINAL_PCT}%))."
        echo "Best snapshot seen: step $(cat "$RDIR/.best_step" 2>/dev/null || echo "$NEW_DONE") at ${best_pct}%."
        echo "Stopped early instead of burning the remaining $(( DPO_ITERS - NEW_DONE )) iters on an already-collapsed policy."
      } > "$RDIR/EARLY_STOP.md"
      touch "$RDIR/.early_stopped"
      echo "!!! EARLY STOP: 2 consecutive snapshots below collapse gate. See $RDIR/EARLY_STOP.md" | tee -a "$RDIR/train.log"
      break
    fi
  fi
done

rm -f "$RESUME_MERGED"

# --- §8.3 step 5 on the FINAL artifact --------------------------------------
# merge_frozen_keys.py refuses any input that is not exactly the picks, so a
# re-run after a successful merge fails loudly rather than double-merging; the
# done-mark keeps that from happening in the first place. Snapshots and -best
# were merged in-loop, before they were ever scored.
if ! is_done merge; then
  echo ">>> merging the frozen CPT-v2 blocks back into the final DPO adapter (§8.3 step 5)"
  python "$MERGER" --in "$DPO_ADAPTER/adapters.safetensors" --out "$DPO_ADAPTER/adapters.safetensors" \
    --cpt-adapter "$CPT_ADAPTER" --spectrum-layers "$LAYERS" 2>&1 | tee "$RDIR/merge_frozen_keys.txt"
  done_mark merge
fi

echo "=== DPO-spectrum training loop done: $RDIR/train.log (best snapshot: step $(cat "$RDIR/.best_step" 2>/dev/null || echo N/A) at ${best_pct}%) ==="
echo "Next: eval_dpo_spectrum.sh"
