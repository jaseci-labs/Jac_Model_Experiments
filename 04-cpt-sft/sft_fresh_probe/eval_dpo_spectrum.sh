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
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

SPEC_DIR="model-experiments/04-cpt-sft/sft_fresh_probe/spectrum"
LAYERS="$SPEC_DIR/configs/spectrum_layers.json"
BASE_MODEL="models/qwen-q4"
DPO_ADAPTER="model-experiments/04-cpt-sft/sft_fresh_probe/adapters/dpo-on-sft-fresh-spectrum"
BEST_ADAPTER="model-experiments/04-cpt-sft/sft_fresh_probe/adapters/dpo-on-sft-fresh-spectrum-best"
HOLDOUT="model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl"
RDIR="model-experiments/04-cpt-sft/sft_fresh_probe/results/dpo-spectrum"
METRICS="$RDIR/metrics_functional.jsonl"

[ -f "$LAYERS" ] || { echo "MISSING: $LAYERS"; exit 1; }
[ -f "$DPO_ADAPTER/adapter_config.json" ] || { echo "MISSING: $DPO_ADAPTER/adapter_config.json (train first)"; exit 1; }

FINAL_STEP="$(cat "$RDIR/.dpo_progress_steps" 2>/dev/null || echo 250)"
BEST_STEP="$(cat "$RDIR/.best_step" 2>/dev/null || echo "$FINAL_STEP")"
LAST_EVAL_STEP=$(( FINAL_STEP + 1000000 ))
BEST_EVAL_STEP=$(( BEST_STEP + 500000 ))

# --- §7 rewrite + the assertion that proves it, per adapter dir --------------
for A in "$DPO_ADAPTER" "$BEST_ADAPTER"; do
  [ -f "$A/adapters.safetensors" ] || continue
  echo ">>> rewriting $A/adapter_config.json so load_adapters covers the spectrum picks (§7)"
  python "$SPEC_DIR/adapter_config_fix.py" --adapter-dir "$A" --spectrum-layers "$LAYERS" \
    | tee -a "$RDIR/adapter_config_rewrite.txt"

  echo ">>> asserting every adapter key lands in the loaded model (strict=False will not tell you)"
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
echo "Next: paired McNemar vs this arm's SFT-spectrum result and vs dpo-nofuse"
echo "(597/855, 69.8%, step 20) -- the trailing-16 DPO incumbent."
