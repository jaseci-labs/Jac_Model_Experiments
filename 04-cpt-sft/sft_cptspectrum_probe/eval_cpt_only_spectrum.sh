#!/usr/bin/env bash
# CPT-ONLY functional eval of the Arm 5 CPT-on-Spectrum snapshot.
#
# Scope change (2026-08-13): before spending the SFT/DPO budget, answer whether
# CPT-on-Spectrum-layers beats stock CPT-v2 (and the no-CPT base) AT THE CPT
# STAGE ITSELF.  This is five-arms-overview.md §5.4 step 6 / cpt-spectrum-plan.md
# §7.4(c), which name the eval but ship no script -- the mechanism is the SFT
# harness pointed at a BARE CPT adapter, exactly as the two baselines were made:
#
#   sft_cptv2_probe/eval_sft_sweep.sh:17-21   ->  CPT-v2 adapter, step 0  -> 404/855
#   sft_fresh_probe/eval_sft_spectrum.sh      ->  ADAPTER="", step 0      ->  90/855
#
# Same harness (sft_cptv2_probe/jacgen/eval_functional.jac, unmodified), same
# byte-identical holdout (md5 dd416f9707bb4a861c1424dbe2c8052b), so all three
# numbers are directly comparable and the baselines are QUOTED, not re-measured.
#
# THE ONE ADDITION over the baselines' invocation: the §7 adapter_config rewrite
# plus the key assertion.  Spectrum's picks include blocks 0/22/23/27/30, which
# are OUTSIDE the trailing-16 slice load_adapters rebuilds from the adapter's own
# `num_layers: 16`.  Without the rewrite, load_weights(strict=False) drops them
# silently and this would score a partially-loaded model -- the same failure
# class as the mlx_lm.fuse bug.  n MUST come back 256.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

PROBE="model-experiments/04-cpt-sft/sft_cptspectrum_probe"
FRESH_SPEC="model-experiments/04-cpt-sft/sft_fresh_probe/spectrum"
ADAPTER="model-experiments/03-cpt-only/adapters/cpt-v2-spectrum-final"
LAYERS="model-experiments/03-cpt-only/cpt_train/spectrum/configs/spectrum_layers.json"
HOLDOUT="$PROBE/dataset/sft/valid.jsonl"
RDIR="model-experiments/03-cpt-only/results/cpt-v2-spectrum/cpt_only_eval"
METRICS="$RDIR/metrics_functional.jsonl"

mkdir -p "$RDIR"
for f in "$ADAPTER/adapters.safetensors" "$LAYERS" "$HOLDOUT" "$FRESH_SPEC/adapter_config_fix.py"; do
  [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
done

if pgrep -f "jac start" >/dev/null 2>&1 || pgrep -f "mlx_lm" >/dev/null 2>&1; then
  echo "!!! another jac/mlx_lm process is running -- refusing to compete for GPU memory."
  pgrep -fl "jac start|mlx_lm" || true; exit 1
fi

echo ">>> sanity: snapshot adapter is real, finite, non-zero, on the picks"
python "$PROBE/arm5_checks.py" adapter "$ADAPTER/adapters.safetensors" --min-bytes 1000000000

echo ">>> §7 rewrite so load_adapters covers the non-trailing picks"
python "$FRESH_SPEC/adapter_config_fix.py" --adapter-dir "$ADAPTER" --spectrum-layers "$LAYERS" \
  | tee "$RDIR/adapter_config_rewrite.txt"

echo ">>> asserting all 256 adapter keys land in the loaded model"
PYTHONPATH="$FRESH_SPEC:${PYTHONPATH:-}" python - "$ADAPTER" <<'PY' | tee "$RDIR/key_assertion.txt"
import sys
from mlx_lm.utils import load
from adapter_config_fix import assert_adapter_keys_present
adapter = sys.argv[1]
model, _ = load("models/qwen-q4", adapter_path=adapter)
n = assert_adapter_keys_present(model, adapter + "/adapters.safetensors")
assert n == 256, f"adapter holds {n} keys, expected 256 -- would be scoring a partially-loaded model"
print(f"OK: all {n} adapter keys present -- {adapter}")
PY

echo ">>> CPT-only functional eval, FULL holdout (~35 min)"
: > "$METRICS"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL=models/qwen-q4 JAC_EVAL_ADAPTER="$ADAPTER" \
  JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP=0 \
  jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac | tee "$RDIR/cpt_only.txt"

echo "=== CPT-only eval done: $METRICS ==="
python "$PROBE/arm5_checks.py" metrics "$METRICS" --min-full-rows 1
