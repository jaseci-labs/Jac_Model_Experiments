#!/usr/bin/env bash
# Full-holdout eval for the cptv2 arm's DPO-spectrum run.
#
# Structurally the fresh arm's eval_dpo_spectrum.sh with the same arm-specific
# correction eval_sft_spectrum.sh carries:
#
#   THE §7 REWRITE MUST USE THE **UNION**, NOT THE PICKS.
#   After the §8.3 step-5 merges, this arm's DPO adapters cover
#   `picks | {32..47}` (336 keys over 21 blocks). Rewriting num_layers from the
#   picks alone would leave any frozen CPT-v2 block below min(picks) outside the
#   trailing slice load_adapters rebuilds, and load_weights(strict=False) would
#   drop it without a word. The key assertion below is what actually catches
#   that -- the rewrite is the fix, the assertion is the proof.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

SPEC_DIR="model-experiments/04-cpt-sft/sft_cptv2_probe/spectrum"
FRESH_SPEC_DIR="model-experiments/04-cpt-sft/sft_fresh_probe/spectrum"
LAYERS="$SPEC_DIR/configs/spectrum_layers.json"
BASE_MODEL="models/qwen-q4"
DPO_ADAPTER="model-experiments/04-cpt-sft/sft_cptv2_probe/adapters/dpo-on-sft-cptv2-spectrum"
BEST_ADAPTER="model-experiments/04-cpt-sft/sft_cptv2_probe/adapters/dpo-on-sft-cptv2-spectrum-best"
HOLDOUT="model-experiments/04-cpt-sft/sft_cptv2_probe/dataset/sft/valid.jsonl"
RDIR="model-experiments/04-cpt-sft/sft_cptv2_probe/results/dpo-spectrum"
METRICS="$RDIR/metrics_functional.jsonl"
UNION="$RDIR/union_layers.json"

[ -f "$LAYERS" ] || { echo "MISSING: $LAYERS"; exit 1; }
[ -f "$DPO_ADAPTER/adapter_config.json" ] || { echo "MISSING: $DPO_ADAPTER/adapter_config.json (train first)"; exit 1; }
if [ ! -f "$RDIR/.merge.done" ]; then
  echo "!!! $RDIR/.merge.done is absent -- run_dpo_spectrum.sh's §8.3 step-5 merge on the"
  echo "    final adapter has not run. The file on disk holds ONLY the trainable picks, so"
  echo "    the CPT-v2 delta is missing and scoring it now would report a wrong number."
  exit 1
fi

if [ ! -f "$UNION" ]; then
  echo ">>> deriving the union picks | {32..47}"
  PYTHONPATH="$SPEC_DIR:$FRESH_SPEC_DIR:${PYTHONPATH:-}" python - "$LAYERS" "$UNION" <<'PY' | tee "$RDIR/union.txt"
import json, sys
from cptv2_spectrum_lora_layers import assert_cpt_v2_shape, union_layers, frozen_layers
picks = json.loads(open(sys.argv[1]).read())["layers"]
base = assert_cpt_v2_shape()
u, fz = union_layers(picks, base), frozen_layers(picks, base)
json.dump({"layers": u, "count": len(u), "picks": picks,
           "cpt_v2_layers": base, "frozen_layers": fz}, open(sys.argv[2], "w"), indent=2)
print(f"union({len(u)})={u}")
PY
fi

FINAL_STEP="$(cat "$RDIR/.dpo_progress_steps" 2>/dev/null || echo 250)"
BEST_STEP="$(cat "$RDIR/.best_step" 2>/dev/null || echo "$FINAL_STEP")"
LAST_EVAL_STEP=$(( FINAL_STEP + 1000000 ))
BEST_EVAL_STEP=$(( BEST_STEP + 500000 ))

for A in "$DPO_ADAPTER" "$BEST_ADAPTER"; do
  [ -f "$A/adapters.safetensors" ] || continue
  echo ">>> rewriting $A/adapter_config.json so load_adapters covers the UNION (§7 + §8.3)"
  python "$FRESH_SPEC_DIR/adapter_config_fix.py" --adapter-dir "$A" --spectrum-layers "$UNION" \
    | tee -a "$RDIR/adapter_config_rewrite.txt"

  echo ">>> asserting every adapter key lands in the loaded model, and that the frozen"
  echo "    CPT-v2 blocks are still bit-identical to the source (strict=False will not tell you)"
  PYTHONPATH="$SPEC_DIR:$FRESH_SPEC_DIR:${PYTHONPATH:-}" python - "$A" "$UNION" "$LAYERS" \
    <<'PY' | tee -a "$RDIR/key_assertion.txt"
import json, sys
from mlx_lm.utils import load
from adapter_config_fix import assert_adapter_keys_present
from cptv2_dpo_spectrum_train import assert_frozen_blocks_match_cpt_v2

adapter, union_path, layers_path = sys.argv[1], sys.argv[2], sys.argv[3]
union = json.loads(open(union_path).read())["layers"]
picks = json.loads(open(layers_path).read())["layers"]

info = assert_frozen_blocks_match_cpt_v2(adapter + "/adapters.safetensors", picks)
print(f"frozen blocks {info['frozen_layers']}: all {info['n_frozen_verified']} "
      f"keys bit-identical to the CPT-v2 source")

model, _ = load("models/qwen-q4", adapter_path=adapter)
n = assert_adapter_keys_present(model, adapter + "/adapters.safetensors")
expected = 16 * len(union)
assert n == expected, (f"adapter holds {n} keys, expected {expected} "
                       f"(16/block x {len(union)} union blocks) -- the step-5 merge "
                       f"did not produce union coverage")
print(f"OK: all {n} adapter keys present in the loaded model, covering {len(union)} blocks -- {adapter}")
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
echo "Next: paired McNemar vs this arm's SFT-spectrum result and vs dpo-v4-nofuse"
echo "(613/855, 71.7%, step 40) -- the trailing-16 DPO incumbent."
