#!/usr/bin/env bash
# Full-holdout eval for the fresh arm's DPO-spectrum run.
#
# eval_dpo_nofuse.sh with the same one addition eval_sft_spectrum.sh has:
# adapter_config.json is rewritten BEFORE any scoring (spectrum-plan.md §7), and
# the rewrite is then PROVEN by asserting every adapter key lands in the loaded
# model. Without it, load_adapters rebuilds LoRA on blocks 32-47 from the
# adapter's own `num_layers: 16` and load_weights(strict=False) silently drops
# blocks 0/22/23/27/30 -- the same failure class as the mlx_lm.fuse bug
# (comparison report §3), and the arm would be scored as a partially-loaded model.
set -euo pipefail
# NOTE (07 jac port): the training/eval drivers are .jac now and are launched with
# `jac run <driver>.jac <flags>` -- jaclang sets sys.argv = [filename] + flags, so the
# drivers' own arg handling is unchanged. [TODO: confirm the `jac` resolved on PATH is
# the SAME interpreter/venv that has mlx / mlx_lm / mlx_lm_lora installed. workflow.md's
# conventions warn a bare `jac` can resolve to a different, stale venv; if so, pin the
# absolute venv jac here.]
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

SPEC_DIR="model-experiments/07-nitin-ds-new-sft/spectrum_probe/spectrum"
LAYERS="$SPEC_DIR/configs/spectrum_layers.json"
BASE_MODEL="models/qwen-q4"
DPO_ADAPTER="model-experiments/07-nitin-ds-new-sft/spectrum_probe/adapters/dpo-on-sft-nitin-spectrum"
BEST_ADAPTER="model-experiments/07-nitin-ds-new-sft/spectrum_probe/adapters/dpo-on-sft-nitin-spectrum-best"
HOLDOUT="${HOLDOUT:-model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl}"
RDIR="${RDIR:-model-experiments/07-nitin-ds-new-sft/spectrum_probe/results/dpo-spectrum}"
# Where run_dpo_spectrum.sh wrote its run state. RDIR is redirected per holdout
# (workflow.md §5.2 -- holdout (b) writes to results/dpo-spectrum-nitinholdout/),
# so the step counters must be read from the TRAINING dir, not from RDIR:
# reading them from a redirected RDIR silently falls back to 250/250 and then
# skips the DPO-best cell entirely via the BEST_STEP != FINAL_STEP test below.
TRAIN_RDIR="${TRAIN_RDIR:-model-experiments/07-nitin-ds-new-sft/spectrum_probe/results/dpo-spectrum}"
METRICS="$RDIR/metrics_functional.jsonl"
mkdir -p "$RDIR"

[ -f "$LAYERS" ] || { echo "MISSING: $LAYERS"; exit 1; }
[ -f "$DPO_ADAPTER/adapter_config.json" ] || { echo "MISSING: $DPO_ADAPTER/adapter_config.json (train first)"; exit 1; }

FINAL_STEP="$(cat "$TRAIN_RDIR/.dpo_progress_steps" 2>/dev/null || echo 250)"
BEST_STEP="$(cat "$TRAIN_RDIR/.best_step" 2>/dev/null || echo "$FINAL_STEP")"
LAST_EVAL_STEP=$(( FINAL_STEP + 1000000 ))
BEST_EVAL_STEP=$(( BEST_STEP + 500000 ))

# --- §7 rewrite + the assertion that proves it, per adapter dir --------------
for A in "$DPO_ADAPTER" "$BEST_ADAPTER"; do
  [ -f "$A/adapters.safetensors" ] || continue
  echo ">>> rewriting $A/adapter_config.json so load_adapters covers the spectrum picks (§7)"
  jac run "$SPEC_DIR/adapter_config_fix.jac" --adapter-dir "$A" --spectrum-layers "$LAYERS" \
    | tee -a "$RDIR/adapter_config_rewrite.txt"

  echo ">>> asserting every adapter key lands in the loaded model (strict=False will not tell you)"
  # NOTE (07 jac port): the heredoc below imports `adapter_config_fix` from $SPEC_DIR --
  # that is now adapter_config_fix.JAC. jaclang registers a .pth import hook at Python
  # startup, so `from adapter_config_fix import ...` resolves the .jac file exactly like a
  # .py one -- provided this `python` is the interpreter jaclang is installed into.
  PYTHONPATH="$SPEC_DIR:${PYTHONPATH:-}" python - "$A" <<'PY' | tee -a "$RDIR/key_assertion.txt"
import sys
from mlx_lm.utils import load
from adapter_config_fix import assert_adapter_keys_present
adapter = sys.argv[1]
model, _ = load("models/qwen-q4", adapter_path=adapter)
n = assert_adapter_keys_present(model, adapter + "/adapters.safetensors")
assert n == 256, f"adapter holds {n} keys, expected 256 (16/block x 16 picked blocks)"
print(f"OK: all {n} adapter keys present in the loaded model -- {adapter}")
PY
done

echo ">>> FULL holdout eval: last checkpoint (step $FINAL_STEP)"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="$BASE_MODEL" JAC_EVAL_ADAPTER="$DPO_ADAPTER" \
  JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$LAST_EVAL_STEP" \
  jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac | tee "$RDIR/final_last.txt"

if [ -f "$BEST_ADAPTER/adapters.safetensors" ] && [ "$BEST_STEP" != "$FINAL_STEP" ]; then
  echo ">>> FULL holdout eval: best snapshot (step $BEST_STEP)"
  JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="$BASE_MODEL" JAC_EVAL_ADAPTER="$BEST_ADAPTER" \
    JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$BEST_EVAL_STEP" \
    jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac | tee "$RDIR/final_best.txt"
else
  echo ">>> best snapshot == last checkpoint (step $BEST_STEP), skipping duplicate full eval"
fi

echo "=== DPO-spectrum eval done: $METRICS ==="
echo "Next: paired McNemar vs this arm's SFT-spectrum result, vs this phase's own"
echo "dpo-nofuse (stock) arm, and cross-phase vs 06-nitin-ds-sft spectrum DPO-final"
echo "636/855 (74.4%) / DPO-best 631/855 (73.8%) -- the incumbent baselines."
