#!/usr/bin/env bash
# Sequential per-checkpoint functional eval for the cptv2 spectrum arm.
#
# Same harness as every other number in RESULTS.md: jacgen/eval_functional.jac,
# env-var driven, unmodified. Structurally the fresh arm's eval_sft_spectrum.sh
# with one arm-specific correction:
#
#   THE §7 REWRITE MUST USE THE **UNION**, NOT THE PICKS.
#   After run_sft_spectrum.sh's §8.3 step-5 merge, this arm's adapters cover
#   `picks | {32..47}`, not just the picks. Rewriting num_layers from the picks
#   alone would leave any frozen CPT-v2 block below min(picks) outside the
#   trailing slice load_adapters rebuilds, and load_weights(strict=False) would
#   drop it without a word. The assertion below is what actually catches that --
#   the rewrite is the fix, the assertion is the proof.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

SPEC_DIR="model-experiments/04-cpt-sft/sft_cptv2_probe/spectrum"
FRESH_SPEC_DIR="model-experiments/04-cpt-sft/sft_fresh_probe/spectrum"
LAYERS="$SPEC_DIR/configs/spectrum_layers.json"
ADAPTER="model-experiments/04-cpt-sft/sft_cptv2_probe/adapters/sft-on-cptv2-spectrum"
RDIR="model-experiments/04-cpt-sft/sft_cptv2_probe/results/sft-spectrum"
HOLDOUT="model-experiments/04-cpt-sft/sft_cptv2_probe/dataset/sft/valid.jsonl"
METRICS="$RDIR/metrics_functional.jsonl"
UNION="$RDIR/union_layers.json"
SUBSET="${SUBSET:-100}"

[ -f "$LAYERS" ] || { echo "MISSING: $LAYERS (run the SNR scan + layer_select.py first)"; exit 1; }
[ -f "$ADAPTER/adapter_config.json" ] || { echo "MISSING: $ADAPTER/adapter_config.json (train first)"; exit 1; }
if [ ! -f "$RDIR/.merge.done" ]; then
  echo "!!! $RDIR/.merge.done is absent -- run_sft_spectrum.sh's §8.3 step-5 merge has"
  echo "    not run. The adapters on disk hold ONLY the trainable picks, so the CPT-v2"
  echo "    delta is missing and scoring them now would report a wrong number."
  exit 1
fi

mkdir -p "$RDIR/images"
: > "$METRICS"

echo ">>> deriving the union picks | {32..47} -- what the merged adapters actually cover"
PYTHONPATH="$SPEC_DIR:$FRESH_SPEC_DIR:${PYTHONPATH:-}" python - "$LAYERS" "$UNION" <<'PY' | tee "$RDIR/union.txt"
import json, sys
from cptv2_spectrum_lora_layers import assert_cpt_v2_shape, union_layers
picks = json.loads(open(sys.argv[1]).read())["layers"]
base = assert_cpt_v2_shape()
u = union_layers(picks, base)
json.dump({"layers": u, "count": len(u), "picks": picks, "cpt_v2_layers": base}, open(sys.argv[2], "w"), indent=2)
print(f"picks({len(picks)})={picks}")
print(f"cpt-v2({len(base)})={base}")
print(f"union({len(u)})={u}")
PY

echo ">>> rewriting adapter_config.json so load_adapters covers the UNION (§7 + §8.3)"
python "$FRESH_SPEC_DIR/adapter_config_fix.py" --adapter-dir "$ADAPTER" --spectrum-layers "$UNION" \
  | tee "$RDIR/adapter_config_rewrite.txt"

echo ">>> asserting every adapter key lands in the loaded model (strict=False will not tell you)"
PYTHONPATH="$FRESH_SPEC_DIR:${PYTHONPATH:-}" python - "$ADAPTER" "$UNION" <<'PY' | tee "$RDIR/key_assertion.txt"
import json, sys
from mlx_lm.utils import load
from adapter_config_fix import assert_adapter_keys_present
adapter, union_path = sys.argv[1], sys.argv[2]
union = json.loads(open(union_path).read())["layers"]
model, _ = load("models/qwen-q4", adapter_path=adapter)
n = assert_adapter_keys_present(model, adapter + "/adapters.safetensors")
expected = 16 * len(union)
assert n == expected, f"adapter holds {n} keys, expected {expected} (16/block x {len(union)} union blocks) -- the step-5 merge did not produce union coverage"
print(f"OK: all {n} adapter keys present in the loaded model, covering {len(union)} blocks")
PY

echo ">>> base (plain Qwen, no CPT, no SFT) -- FULL holdout"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL=models/qwen-q4 JAC_EVAL_ADAPTER="" \
  JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP=0 \
  jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac | tee "$RDIR/base.txt"

TMPADP="model-experiments/04-cpt-sft/sft_cptv2_probe/adapters/sft-spectrum-ckpt-eval"
CKPT_DIR="$ADAPTER/checkpoints"
for CK in "$CKPT_DIR"/*_adapters.safetensors; do
  [ -e "$CK" ] || continue
  # run_sft_spectrum.sh's watchdog archives each checkpoint under its TRUE global
  # step name and step 5 merged the frozen CPT-v2 blocks into each of them, so
  # the filename IS the real step and the file IS the full composition.
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

echo ">>> final cptv2 spectrum SFT checkpoint -- FULL holdout"
TOTAL_ITERS="$(grep -E '^iters:' "$SPEC_DIR/configs/sft_spectrum.yaml" | grep -oE '[0-9]+' | head -1)"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL=models/qwen-q4 JAC_EVAL_ADAPTER="$ADAPTER" \
  JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$TOTAL_ITERS" \
  jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac | tee "$RDIR/final.txt"

echo "=== functional eval sweep done: $METRICS ==="
echo "Next: paired McNemar vs sft-on-cptv2 (621/855, 72.6%), then the four-way"
echo "table in spectrum-workflow.md Phase 8."
