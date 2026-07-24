#!/usr/bin/env bash
# Sequential per-checkpoint functional eval for the fresh-base probe. Reuses
# sft_cptv2_probe/jacgen/eval_functional.jac unmodified (env-var driven).
set -euo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

ADAPTER="model-experiments/04-cpt-sft/sft_fresh_probe/adapters/sft-on-fresh"
RDIR="model-experiments/04-cpt-sft/sft_fresh_probe/results/sft"
HOLDOUT="model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl"
METRICS="$RDIR/metrics_functional.jsonl"
SUBSET="${SUBSET:-100}"

mkdir -p "$RDIR/images"
: > "$METRICS"

echo ">>> base (plain Qwen, no CPT, no SFT) -- FULL holdout"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL=models/qwen-q4 JAC_EVAL_ADAPTER="" \
  JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP=0 \
  jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac | tee "$RDIR/base.txt"

TMPADP="model-experiments/04-cpt-sft/sft_fresh_probe/adapters/sft-ckpt-eval"
CKPT_DIR="$ADAPTER/checkpoints"
for CK in "$CKPT_DIR"/*_adapters.safetensors; do
  [ -e "$CK" ] || continue
  # Unlike sft_cptv2_probe's eval_sft_sweep.sh (which needed a manual "+820"
  # true-global-step correction because it swept mlx_lm.lora's own
  # LOCALLY-numbered checkpoint files), run_sft.sh's watchdog loop already
  # copies each segment's checkpoint into checkpoints/ under its TRUE global
  # step name -- so the filename IS the real step, no offset needed here.
  STEP="$(basename "$CK" | grep -oE '^[0-9]+' | sed 's/^0*//')"; STEP="${STEP:-0}"
  rm -rf "$TMPADP"; mkdir -p "$TMPADP"
  cp "$CK" "$TMPADP/adapters.safetensors"
  [ -f "$ADAPTER/adapter_config.json" ] && cp "$ADAPTER/adapter_config.json" "$TMPADP/adapter_config.json"
  echo ">>> checkpoint $STEP (subset=$SUBSET)"
  JAC_EVAL_MODE=mlx JAC_EVAL_MODEL=models/qwen-q4 JAC_EVAL_ADAPTER="$TMPADP" \
    JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_LIMIT="$SUBSET" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$STEP" \
    jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac 2>/dev/null | tail -5
done
rm -rf "$TMPADP"

echo ">>> final SFT checkpoint -- FULL holdout"
TOTAL_ITERS="$(grep -E '^iters:' model-experiments/04-cpt-sft/sft_fresh_probe/configs/sft.yaml | grep -oE '[0-9]+' | head -1)"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL=models/qwen-q4 JAC_EVAL_ADAPTER="$ADAPTER" \
  JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$TOTAL_ITERS" \
  jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac | tee "$RDIR/final.txt"

echo "=== functional eval sweep done: $METRICS ==="
