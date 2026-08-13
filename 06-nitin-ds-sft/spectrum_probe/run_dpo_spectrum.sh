#!/usr/bin/env bash
# ============================================================================
# DPO on the Spectrum-picked layers -- fresh arm.
#
# This is run_dpo_nofuse.sh with ONE variable changed: which 16 blocks carry the
# LoRA. Everything else is the corrected recipe verbatim (comparison report §3):
# no `mlx_lm.fuse`, BASE_MODEL=models/qwen-q4 throughout, DPO's LoRA seeded from
# the SFT adapter via --resume-adapter-file, beta 0.1, lr 1e-6, 654 pairs,
# max_seq_length 512, 250 iters, the same segmented watchdog / OOM shrink ladder
# / snapshot + subset-eval / collapse-gate machinery, and the same
# chat-template fix. The driver is spectrum/dpo_spectrum_train.py, which carries
# BOTH the chat-template fix and the layer rebind (see its header for why the
# rebind needs a THIRD site, mlx_lm_lora.utils, that the SFT driver does not).
#
# THREE ADDITIONS over run_dpo_nofuse.sh, all of them gates:
#
#   1. --verify-patches (seconds, no model load): both monkey-patches live, all
#      three call sites rebound, on the frozen picks. Composing two patches is
#      where one silently loses; this refuses to launch if it did.
#   2. --verify-layers (minutes, loads models/qwen-q4 twice): the REAL DPO
#      conversion path (mlx_lm_lora.utils.from_pretrained) converts the picks,
#      trainable == 281.838M exactly, and -- the DPO-specific one -- every key of
#      the SFT-spectrum adapter finds a home in the converted model.
#      mlx_lm_lora/train.py:527 loads it with strict=False and will NOT tell you.
#   3. adapter_config.json is rewritten on every snapshot BEFORE it is scored
#      (spectrum-plan.md §7). The in-loop subset evals go through
#      mlx_lm.load(..., adapter_path=SNAP) -> load_adapters, which rebuilds LoRA
#      from the adapter's own `num_layers: 16` = blocks 32-47. Without the
#      rewrite, block 0 (and 22/23/27/30) would be dropped silently and every
#      in-loop number -- including the best-snapshot selection and the collapse
#      gate -- would be measuring a partially-loaded model.
#
# Baseline for the collapse gate is this arm's SFT-SPECTRUM result, not sft/.
#
# Outputs: adapters/dpo-on-sft-nitin-spectrum{,-best}, results/dpo-spectrum/.
# Nothing existing is touched.
#
# LAUNCH ONLY AFTER run_sft_spectrum.sh AND eval_sft_spectrum.sh have completed.
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

SPEC_DIR="model-experiments/06-nitin-ds-sft/spectrum_probe/spectrum"
DRIVER="$SPEC_DIR/dpo_spectrum_train.py"
LAYERS="$SPEC_DIR/configs/spectrum_layers.json"
CFGFIX="$SPEC_DIR/adapter_config_fix.py"
BASE_MODEL="models/qwen-q4"
SFT_ADAPTER="model-experiments/06-nitin-ds-sft/spectrum_probe/adapters/sft-on-nitin-spectrum"
DPO_ADAPTER="model-experiments/06-nitin-ds-sft/spectrum_probe/adapters/dpo-on-sft-nitin-spectrum"
BEST_ADAPTER="model-experiments/06-nitin-ds-sft/spectrum_probe/adapters/dpo-on-sft-nitin-spectrum-best"
RDIR="model-experiments/06-nitin-ds-sft/spectrum_probe/results/dpo-spectrum"
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
DATA="model-experiments/06-nitin-ds-sft/dataset/dpo"
DPO_CFG="model-experiments/06-nitin-ds-sft/spectrum_probe/configs/dpo_lora.yaml"

mkdir -p "$RDIR" "$SNAP_DIR"
for f in "$DRIVER" "$LAYERS" "$CFGFIX" "$DPO_CFG" "$DATA/train.jsonl" "$HOLDOUT"; do
  [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }
done
if [ ! -f "$SFT_ADAPTER/adapters.safetensors" ]; then
  echo "!!! MISSING: $SFT_ADAPTER/adapters.safetensors"
  echo "    DPO on the spectrum layers continues the SFT-spectrum LoRA lineage."
  echo "    Run run_sft_spectrum.sh (and eval_sft_spectrum.sh) first."
  exit 1
fi
[ -f "$DPO_METRICS" ] || : > "$DPO_METRICS"

done_mark() { touch "$RDIR/.$1.done"; }
is_done() { [ -f "$RDIR/.$1.done" ]; }

# --- gate 1: both patches live, all three sites, right layers (seconds) -----
if ! is_done verify_patches; then
  echo ">>> --verify-patches: chat-template fix AND layer rebind, all three call sites"
  python "$DRIVER" --verify-patches --spectrum-layers "$LAYERS" 2>&1 | tee "$RDIR/verify_patches.txt"
  grep -q "^VERIFY: PASS" "$RDIR/verify_patches.txt" || {
    echo "!!! patch composition FAILED -- see $RDIR/verify_patches.txt. Not training."; exit 1; }
  done_mark verify_patches
fi

# --- gate 2: the real conversion path on the real model (minutes) -----------
if ! is_done verify_layers && [ "${SKIP_VERIFY:-0}" != "1" ]; then
  echo ">>> --verify-layers self-test (loads models/qwen-q4 twice; a few minutes)"
  python "$DRIVER" --verify-layers --spectrum-layers "$LAYERS" --model "$BASE_MODEL" \
    --resume-adapter-file "$SFT_ADAPTER/adapters.safetensors" 2>&1 | tee "$RDIR/verify_layers.txt"
  grep -q "^VERIFY: PASS" "$RDIR/verify_layers.txt" || {
    echo "!!! self-test FAILED -- see $RDIR/verify_layers.txt. Not training."; exit 1; }
  done_mark verify_layers
fi

# NOTE (06-nitin retarget): the lookup below filters on `total == 855`, i.e. it
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
    with open('model-experiments/06-nitin-ds-sft/spectrum_probe/results/sft-spectrum/metrics_functional.jsonl') as f:
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
  python "$DRIVER" --spectrum-layers "$LAYERS" --model "$BASE_MODEL" --train --data "$DATA" \
    --train-mode dpo --config "$DPO_CFG" \
    --adapter-path model-experiments/06-nitin-ds-sft/spectrum_probe/adapters/dpo-dry-spectrum \
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
  # Two explicit branches, NOT an empty-array expansion -- macOS bash 3.2 treats
  # "${ARR[@]}" as an unbound-variable error under `set -u` when ARR is empty.
  if [ "$DONE_STEPS" -gt 0 ] && [ -f "$DPO_ADAPTER/adapters.safetensors" ]; then
    python "$DRIVER" --spectrum-layers "$LAYERS" --model "$BASE_MODEL" --train --data "$DATA" \
      --train-mode dpo --config "$DPO_CFG" \
      --adapter-path "$DPO_ADAPTER" --train-type lora --num-layers 16 --grad-checkpoint \
      --batch-size 1 --max-seq-length "$DPO_MAXLEN" --iters "$SEG_ITERS" \
      --resume-adapter-file "$DPO_ADAPTER/adapters.safetensors" \
      --learning-rate "$DPO_LR" --beta "$DPO_BETA" --dpo-cpo-loss-type sigmoid \
      --steps-per-report 5 --steps-per-eval "$SEG_ITERS" --val-batches "$DPO_VAL_BATCHES" --save-every "$SEG_ITERS" \
      > "$RDIR/.segment.log" 2>&1 &
  else
    # First real segment -- seed from the SFT-spectrum adapter, not from scratch.
    python "$DRIVER" --spectrum-layers "$LAYERS" --model "$BASE_MODEL" --train --data "$DATA" \
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
  cp "$DPO_ADAPTER/adapters.safetensors" "$SNAP/adapters.safetensors"
  cp "$DPO_ADAPTER/adapter_config.json" "$SNAP/adapter_config.json"

  # spectrum-plan.md §7, and it MUST happen before the subset eval below.
  # from_pretrained rewrites $DPO_ADAPTER/adapter_config.json to `num_layers: 16`
  # at the start of every segment, so this is done on the SNAP copy each time
  # rather than once on the live adapter dir.
  python "$CFGFIX" --adapter-dir "$SNAP" --spectrum-layers "$LAYERS" \
    >> "$RDIR/adapter_config_rewrite.txt" 2>&1
  echo "  snapshot saved + adapter_config rewritten: $SNAP" | tee -a "$RDIR/train.log"

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
    cp "$SNAP/adapter_config.json" "$BEST_ADAPTER/adapter_config.json"   # already rewritten
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
echo "=== DPO-spectrum training loop done: $RDIR/train.log (best snapshot: step $(cat "$RDIR/.best_step" 2>/dev/null || echo N/A) at ${best_pct}%) ==="
echo "Next: eval_dpo_spectrum.sh"
