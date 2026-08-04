#!/usr/bin/env bash
# Sequential per-checkpoint functional eval for the cptspectrum arm (CPT itself
# trained on Spectrum's picks, then SFT on the same picks -- cpt-spectrum-plan.md
# §4.3). Reuses sft_cptv2_probe/jacgen/eval_functional.jac unmodified (env-var
# driven), exactly as every other arm does, so this arm is scored by the same
# harness that produced every other number in RESULTS.md.
#
# THE ONE ADDITION: adapter_config.json is rewritten BEFORE any scoring. Without
# it, load_adapters rebuilds LoRA on blocks 32-47 from the adapter's own
# `num_layers: 16` and load_weights(strict=False) SILENTLY DROPS every
# out-of-slice layer -- the same failure class as the mlx_lm.fuse bug in
# comparison report §3.
#
# Base row is SKIPPED deliberately (cpt-spectrum-plan.md §4.3): the plain-base
# score on this holdout is already fixed at 10.5% (90/855), re-measuring it here
# adds nothing and costs ~25 minutes.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

SPEC_DIR="model-experiments/04-cpt-sft/sft_cptspectrum_probe/spectrum"
FRESH_SPEC_DIR="model-experiments/04-cpt-sft/sft_fresh_probe/spectrum"
LAYERS="$SPEC_DIR/configs/spectrum_layers.json"
ADAPTER="model-experiments/04-cpt-sft/sft_cptspectrum_probe/adapters/sft-on-cptspectrum"
RDIR="model-experiments/04-cpt-sft/sft_cptspectrum_probe/results/sft-spectrum"
HOLDOUT="model-experiments/04-cpt-sft/sft_cptspectrum_probe/dataset/sft/valid.jsonl"
METRICS="$RDIR/metrics_functional.jsonl"
SUBSET="${SUBSET:-100}"

[ -f "$LAYERS" ] || { echo "MISSING: $LAYERS"; exit 1; }
[ -f "$ADAPTER/adapter_config.json" ] || { echo "MISSING: $ADAPTER/adapter_config.json (train first)"; exit 1; }

mkdir -p "$RDIR/images"
: > "$METRICS"

echo ">>> rewriting adapter_config.json so load_adapters covers the spectrum picks (§7)"
python "$FRESH_SPEC_DIR/adapter_config_fix.py" --adapter-dir "$ADAPTER" --spectrum-layers "$LAYERS" \
  | tee "$RDIR/adapter_config_rewrite.txt"

echo ">>> asserting every adapter key lands in the loaded model (strict=False will not tell you)"
PYTHONPATH="$FRESH_SPEC_DIR:${PYTHONPATH:-}" python - "$ADAPTER" <<'PY' | tee "$RDIR/key_assertion.txt"
import sys
from mlx_lm.utils import load
from adapter_config_fix import assert_adapter_keys_present
adapter = sys.argv[1]
model, _ = load("models/qwen-q4", adapter_path=adapter)
n = assert_adapter_keys_present(model, adapter + "/adapters.safetensors")
print(f"OK: all {n} adapter keys present in the loaded model")
PY

TMPADP="model-experiments/04-cpt-sft/sft_cptspectrum_probe/adapters/sft-spectrum-ckpt-eval"
CKPT_DIR="$ADAPTER/checkpoints"
for CK in "$CKPT_DIR"/*_adapters.safetensors; do
  [ -e "$CK" ] || continue
  STEP="$(basename "$CK" | grep -oE '^[0-9]+' | sed 's/^0*//')"; STEP="${STEP:-0}"
  rm -rf "$TMPADP"; mkdir -p "$TMPADP"
  cp "$CK" "$TMPADP/adapters.safetensors"
  cp "$ADAPTER/adapter_config.json" "$TMPADP/adapter_config.json"   # already rewritten
  echo ">>> checkpoint $STEP (subset=$SUBSET)"
  JAC_EVAL_MODE=mlx JAC_EVAL_MODEL=models/qwen-q4 JAC_EVAL_ADAPTER="$TMPADP" \
    JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_LIMIT="$SUBSET" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$STEP" \
    jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac 2>/dev/null | tail -5
done
rm -rf "$TMPADP"

echo ">>> final cptspectrum SFT checkpoint -- FULL holdout"
TOTAL_ITERS="$(grep -E '^iters:' "$SPEC_DIR/configs/sft_spectrum.yaml" | grep -oE '[0-9]+' | head -1)"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL=models/qwen-q4 JAC_EVAL_ADAPTER="$ADAPTER" \
  JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$TOTAL_ITERS" \
  jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac | tee "$RDIR/final.txt"

echo "=== functional eval sweep done: $METRICS ==="
echo "Next: three-way z-tests vs sft-on-cptv2 (72.6%, stock) and sft-on-cptv2-spectrum (70.5%, union/freeze)."
