#!/usr/bin/env bash
# ============================================================================
# Arm 5 (CPT-on-Spectrum, "clean chain") -- ONE chained orchestrator.
#
#   CPT (12 legs) -> snapshot -> SFT -> SFT eval -> DPO -> DPO eval -> stats
#
# ~37-41h end to end (cpt-spectrum-plan.md §6.2).  Every stage is gated on the
# PREVIOUS stage's real, verified output -- never a blind `&&`.  Verification is
# `arm5_checks.py`, which materialises safetensors with mx.eval before
# inspecting them and requires every tensor finite and not all-zero, because a
# right-sized file with the right key count has ALREADY been garbage once in
# this repo (results/sft-spectrum-BROKEN-zeroweights, comparison report §3.1).
#
# Resumable: each stage drops a `.done` marker under $STATE.  Re-running the
# script skips completed stages and picks up where it left off.  The CPT loop is
# additionally resumable per-leg by construction (resume_point() re-derives from
# on-disk checkpoints), so a kill costs at most the in-flight leg (~2.2h).
#
# NEVER deletes or overwrites any other arm's adapters/results.  Every path it
# writes is under adapters/cpt-v2-spectrum*, results/cpt-v2-spectrum, or
# sft_cptspectrum_probe/.
#
# Launch:  bash .../run_arm5_full_chain.sh
# with ONE backgrounding mechanism only (the harness's) -- no trailing `&`,
# no nohup, no disown (lesson 4: two mechanisms orphan duplicate processes).
# ============================================================================
set -uo pipefail   # deliberately NOT -e: stages are gated explicitly below.

if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu env CAFFEINATED=1 "$0" "$@"
fi

ROOT="/Volumes/ExtremePro/JaseciLabs/jac_model_studio"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"
PY="$ROOT/.venv/bin/python3"

PROBE="model-experiments/04-cpt-sft/sft_cptspectrum_probe"
CPT_DIR="model-experiments/03-cpt-only"
CHECKS="$PY $PROBE/arm5_checks.py"

STATE="$PROBE/results/arm5-chain"
CHAIN_LOG="$STATE/chain.log"
mkdir -p "$STATE" "$CPT_DIR/results/cpt-v2-spectrum/logs" "$PROBE/results/sft-spectrum" "$PROBE/results/dpo-spectrum"

ADAPTER_CPT="$CPT_DIR/adapters/cpt-v2-spectrum"
ADAPTER_FINAL="$CPT_DIR/adapters/cpt-v2-spectrum-final"
STATE_JSON="$CPT_DIR/results/cpt-v2-spectrum/json/training_state.json"
LAYERS="$CPT_DIR/cpt_train/spectrum/configs/spectrum_layers.json"
CFGFIX="model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/adapter_config_fix.py"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$CHAIN_LOG"; }
stage_done() { [ -f "$STATE/.$1.done" ]; }
mark_done()  { touch "$STATE/.$1.done"; }

die() {
  log "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  log "!!! CHAIN ABORTED at stage: ${CURRENT_STAGE:-unknown}"
  log "!!! $*"
  log "!!! Not proceeding to any later stage. Nothing was deleted."
  log "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo "${CURRENT_STAGE:-unknown}" > "$STATE/.FAILED_AT_STAGE"
  echo "$*" > "$STATE/.FAILURE_REASON"
  exit 1
}

run_checked() {  # run_checked <description> <cmd...>
  local desc="$1"; shift
  log ">>> $desc"
  "$@" >> "$CHAIN_LOG" 2>&1
  local rc=$?
  [ $rc -eq 0 ] || die "$desc exited with code $rc (see $CHAIN_LOG)"
}

CHAIN_T0=$(date +%s)
log "================================================================"
log "Arm 5 full chain starting. Estimated 37-41h (cpt-spectrum-plan.md §6.2)."
log "PID $$  |  log: $CHAIN_LOG"
log "================================================================"

# --- preflight ---------------------------------------------------------------
CURRENT_STAGE="preflight"
if pgrep -f "jac start" >/dev/null 2>&1 || pgrep -f "mlx_lm" >/dev/null 2>&1; then
  pgrep -fl "jac start|mlx_lm" | tee -a "$CHAIN_LOG"
  die "another jac/mlx_lm process is already running -- a competing resident model OOMs this 48GB machine"
fi
AVAIL_GB=$(df -g "$ROOT" | awk 'NR==2 {print $4}')
log "disk available: ${AVAIL_GB} GB (need ~70 GB, R11)"
[ "${AVAIL_GB:-0}" -ge 100 ] || die "only ${AVAIL_GB} GB free; need >=100 GB headroom for ~70 GB of checkpoints"
for f in "$ROOT/models/qwen-q4/config.json" "$LAYERS" "$CPT_DIR/cpt_train/spectrum/configs/config_v2_leg1.yaml" \
         "$CPT_DIR/dataset/cpt-v2/packed/train.jsonl" "$CPT_DIR/dataset/cpt-v2/manifest.json"; do
  [ -e "$f" ] || die "missing required input: $f"
done
log "preflight OK: no competing job, base model present, corpus present, layers present"

# ============================================================ STAGE 1 -- CPT
CURRENT_STAGE="1-cpt"
if stage_done 1-cpt; then
  log "STAGE 1 (CPT): already complete, skipping"
else
  log "STAGE 1 (CPT): 12 legs on Spectrum's picks, ~26-30h. Resumes from on-disk checkpoints."
  S1_T0=$(date +%s)
  CONFIRM_FULL_RUN=1 bash "$CPT_DIR/cpt_train/spectrum/run_cpt_spectrum.sh" \
    >> "$CPT_DIR/results/cpt-v2-spectrum/logs/orchestrator.log" 2>&1
  RC=$?
  S1_MIN=$(( ($(date +%s) - S1_T0) / 60 ))
  log "STAGE 1: epoch loop exited rc=$RC after ${S1_MIN} min"
  [ $RC -eq 0 ] || die "CPT epoch loop exited $RC -- see $CPT_DIR/results/cpt-v2-spectrum/logs/orchestrator.log"

  # Real verification: the loop reached a halt decision, >= floor legs, finite losses.
  run_checked "STAGE 1 verify: training_state.json" $CHECKS state "$STATE_JSON" --min-legs 6
  # And the accepted rolling checkpoint is a real, non-degenerate adapter.
  run_checked "STAGE 1 verify: accepted adapter" $CHECKS adapter "$ADAPTER_CPT/adapters.safetensors" --min-bytes 1000000000
  log "STAGE 1 (CPT): COMPLETE and verified"
  mark_done 1-cpt
fi

# ======================================================= STAGE 2 -- SNAPSHOT
CURRENT_STAGE="2-snapshot"
if stage_done 2-snapshot; then
  log "STAGE 2 (snapshot): already complete, skipping"
else
  log "STAGE 2: freezing the accepted CPT checkpoint into cpt-v2-spectrum-final/ (§2.7)"
  mkdir -p "$ADAPTER_FINAL"
  # shutil/cp of BYTES to a DIFFERENT path -- never an in-place mx.load+write,
  # which is the exact shape of the BROKEN-zeroweights bug (§2.2c).
  cp "$ADAPTER_CPT/adapters.safetensors" "$ADAPTER_FINAL/adapters.safetensors" \
    || die "failed to copy the accepted CPT adapter"
  cp "$ADAPTER_CPT/adapter_config.json" "$ADAPTER_FINAL/adapter_config.json" \
    || die "failed to copy adapter_config.json"
  run_checked "STAGE 2: rewrite adapter_config for the picks" \
    $PY "$CFGFIX" --adapter-dir "$ADAPTER_FINAL" --spectrum-layers "$LAYERS"
  # Verify the SNAPSHOT itself, not the source -- catches any copy/mmap damage.
  run_checked "STAGE 2 verify: snapshot adapter is intact (not zeroed)" \
    $CHECKS adapter "$ADAPTER_FINAL/adapters.safetensors" --min-bytes 1000000000
  $PY - <<PY >> "$CHAIN_LOG" 2>&1 || die "snapshot vs source byte-comparison failed"
import hashlib, pathlib
a = pathlib.Path("$ADAPTER_CPT/adapters.safetensors").read_bytes()
b = pathlib.Path("$ADAPTER_FINAL/adapters.safetensors").read_bytes()
assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest(), "snapshot differs from source!"
print("snapshot sha256 matches source exactly")
PY
  log "STAGE 2 (snapshot): COMPLETE and verified"
  mark_done 2-snapshot
fi

# ============================================================ STAGE 3 -- SFT
CURRENT_STAGE="3-sft"
if stage_done 3-sft; then
  log "STAGE 3 (SFT): already complete, skipping"
else
  # 3a: dry gate. The script runs --verify-layers + a 30-iter dry run, then
  # exits 0 WITHOUT training unless CONFIRM_FULL_RUN=1.
  if ! stage_done 3a-sft-dry; then
    log "STAGE 3a: SFT dry gate (--verify-layers + 30-iter dry run, no CONFIRM_FULL_RUN)"
    bash "$PROBE/run_sft_spectrum.sh" >> "$CHAIN_LOG" 2>&1
    RC=$?
    [ $RC -eq 0 ] || die "SFT dry gate exited $RC"
    grep -q "^VERIFY: PASS" "$PROBE/results/sft-spectrum/verify_layers.txt" \
      || die "SFT --verify-layers did not print 'VERIFY: PASS'"
    [ -f "$PROBE/adapters/dry-spectrum/adapters.safetensors" ] \
      || die "SFT dry run produced no adapter at $PROBE/adapters/dry-spectrum/"
    run_checked "STAGE 3a verify: dry adapter is sane" \
      $CHECKS adapter "$PROBE/adapters/dry-spectrum/adapters.safetensors" --min-bytes 1000000000
    log "STAGE 3a: dry gate PASSED"
    mark_done 3a-sft-dry
  fi

  log "STAGE 3b: real SFT, 8200 iters, ~3.3h"
  S3_T0=$(date +%s)
  CONFIRM_FULL_RUN=1 bash "$PROBE/run_sft_spectrum.sh" >> "$CHAIN_LOG" 2>&1
  RC=$?
  log "STAGE 3b: SFT exited rc=$RC after $(( ($(date +%s) - S3_T0) / 60 )) min"
  [ $RC -eq 0 ] || die "SFT training exited $RC"
  DONE_STEPS=$(cat "$PROBE/results/sft-spectrum/.sft_progress_steps" 2>/dev/null || echo 0)
  [ "$DONE_STEPS" -ge 8200 ] || die "SFT only reached ${DONE_STEPS}/8200 iters"
  run_checked "STAGE 3b verify: SFT adapter" \
    $CHECKS adapter "$PROBE/adapters/sft-on-cptspectrum/adapters.safetensors" --min-bytes 1000000000
  NCK=$(ls "$PROBE/adapters/sft-on-cptspectrum/checkpoints"/*_adapters.safetensors 2>/dev/null | wc -l | tr -d ' ')
  log "STAGE 3b: ${NCK} archived checkpoints, ${DONE_STEPS} iters done"
  [ "$NCK" -ge 5 ] || die "only $NCK SFT checkpoints archived, expected ~10"
  log "STAGE 3 (SFT): COMPLETE and verified"
  mark_done 3-sft
fi

# ======================================================= STAGE 4 -- SFT EVAL
CURRENT_STAGE="4-sft-eval"
if stage_done 4-sft-eval; then
  log "STAGE 4 (SFT eval): already complete, skipping"
else
  log "STAGE 4: SFT functional eval sweep, ~1.3h"
  S4_T0=$(date +%s)
  bash "$PROBE/eval_sft_spectrum.sh" >> "$CHAIN_LOG" 2>&1
  RC=$?
  log "STAGE 4: eval exited rc=$RC after $(( ($(date +%s) - S4_T0) / 60 )) min"
  [ $RC -eq 0 ] || die "SFT eval exited $RC"
  run_checked "STAGE 4 verify: full-holdout SFT row present" \
    $CHECKS metrics "$PROBE/results/sft-spectrum/metrics_functional.jsonl" --min-full-rows 1
  log "STAGE 4 (SFT eval): COMPLETE and verified"
  mark_done 4-sft-eval
fi

# ============================================================ STAGE 5 -- DPO
# run_dpo_spectrum.sh is derived from run_dpo_nofuse.sh -- VERIFIED before this
# chain was launched: no `mlx_lm.fuse` anywhere, `fuse: false` in dpo_lora.yaml,
# and the driver carries the FixedDPODataset chat-template fix. The known int4
# requantisation-collapse bug (commit 6cd8b39) cannot recur here.
CURRENT_STAGE="5-dpo"
if stage_done 5-dpo; then
  log "STAGE 5 (DPO): already complete, skipping"
else
  if ! stage_done 5a-dpo-dry; then
    log "STAGE 5a: DPO dry gate (--verify-patches, --verify-layers, 8-iter dry at DPO_MAXLEN=512)"
    bash "$PROBE/run_dpo_spectrum.sh" >> "$CHAIN_LOG" 2>&1
    RC=$?
    if [ $RC -ne 0 ]; then
      if tail -400 "$CHAIN_LOG" | grep -qEi "out of memory|OutOfMemory|kIOGPUCommandBuffer"; then
        log "STAGE 5a: dry run OOMed at maxlen 512 -- retrying at 384 (§5.3 step 2, the documented fallback)"
        rm -f "$PROBE/results/dpo-spectrum/.dry.done"
        echo "384" > "$STATE/.dpo_maxlen"
        DPO_MAXLEN=384 bash "$PROBE/run_dpo_spectrum.sh" >> "$CHAIN_LOG" 2>&1 \
          || die "DPO dry run OOMed at BOTH 512 and 384 -- stopping per §5.3 step 4, do not improvise other knobs"
        log "STAGE 5a: dry gate passed at DPO_MAXLEN=384 (RECIPE DEVIATION -- must be flagged in the report)"
      else
        die "DPO dry gate exited $RC (not an OOM signature)"
      fi
    fi
    grep -q "^VERIFY: PASS" "$PROBE/results/dpo-spectrum/verify_patches.txt" || die "DPO --verify-patches failed"
    grep -q "^VERIFY: PASS" "$PROBE/results/dpo-spectrum/verify_layers.txt"  || die "DPO --verify-layers failed"
    log "STAGE 5a: DPO dry gate PASSED"
    mark_done 5a-dpo-dry
  fi

  MAXLEN=$(cat "$STATE/.dpo_maxlen" 2>/dev/null || echo 512)
  log "STAGE 5b: real DPO, 250 iters at DPO_MAXLEN=${MAXLEN}, ~1.2h"
  S5_T0=$(date +%s)
  CONFIRM_FULL_RUN=1 DPO_MAXLEN="$MAXLEN" bash "$PROBE/run_dpo_spectrum.sh" >> "$CHAIN_LOG" 2>&1
  RC=$?
  log "STAGE 5b: DPO exited rc=$RC after $(( ($(date +%s) - S5_T0) / 60 )) min"
  [ $RC -eq 0 ] || die "DPO training exited $RC"
  run_checked "STAGE 5b verify: DPO final adapter" \
    $CHECKS adapter "$PROBE/adapters/dpo-on-sft-cptspectrum/adapters.safetensors" --min-bytes 1000000000
  run_checked "STAGE 5b verify: DPO best adapter" \
    $CHECKS adapter "$PROBE/adapters/dpo-on-sft-cptspectrum-best/adapters.safetensors" --min-bytes 1000000000
  log "STAGE 5 (DPO): COMPLETE and verified"
  mark_done 5-dpo
fi

# ======================================================= STAGE 6 -- DPO EVAL
CURRENT_STAGE="6-dpo-eval"
if stage_done 6-dpo-eval; then
  log "STAGE 6 (DPO eval): already complete, skipping"
else
  log "STAGE 6: DPO full-holdout eval (final + best), ~1.2h"
  S6_T0=$(date +%s)
  bash "$PROBE/eval_dpo_spectrum.sh" >> "$CHAIN_LOG" 2>&1
  RC=$?
  log "STAGE 6: eval exited rc=$RC after $(( ($(date +%s) - S6_T0) / 60 )) min"
  [ $RC -eq 0 ] || die "DPO eval exited $RC"
  run_checked "STAGE 6 verify: full-holdout DPO rows present" \
    $CHECKS metrics "$PROBE/results/dpo-spectrum/metrics_functional.jsonl" --min-full-rows 1
  log "STAGE 6 (DPO eval): COMPLETE and verified"
  mark_done 6-dpo-eval
fi

# ========================================================== STAGE 7 -- STATS
CURRENT_STAGE="7-stats"
log "STAGE 7: three-way comparison statistics (six z-tests, §7.2)"
$PY "$PROBE/arm5_stats.py" --out "$PROBE/results/arm5_stats.json" >> "$CHAIN_LOG" 2>&1 \
  || die "arm5_stats.py failed"
log "STAGE 7: wrote $PROBE/results/arm5_stats.json"
mark_done 7-stats

TOTAL_H=$(( ($(date +%s) - CHAIN_T0) / 3600 ))
TOTAL_M=$(( (($(date +%s) - CHAIN_T0) % 3600) / 60 ))
log "================================================================"
log "ARM 5 CHAIN COMPLETE. Wall clock this invocation: ${TOTAL_H}h ${TOTAL_M}m"
log "Remaining (agent, not this script): write the three-way report,"
log "update five-arms-overview.md and accuracy-summary.md."
log "================================================================"
touch "$STATE/.ALL_STAGES_COMPLETE"
