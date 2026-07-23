#!/usr/bin/env bash
# Sequential per-checkpoint functional eval -- ONE model in RAM at a time,
# same convention as run_probe.sh step 6 (no concurrent training+eval).
set -euo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

ADAPTER="model-experiments/04-cpt-sft/sft_cptv2_probe/adapters/sft-on-cptv2"
RDIR="model-experiments/04-cpt-sft/sft_cptv2_probe/results/sft"
HOLDOUT="model-experiments/04-cpt-sft/sft_cptv2_probe/dataset/sft/valid.jsonl"
METRICS="$RDIR/metrics_functional.jsonl"
SUBSET="${SUBSET:-100}"   # rows/checkpoint for the interim learning curve (full 1428 only at base/final)

mkdir -p "$RDIR/images"
: > "$METRICS"

echo ">>> base (CPT-v2, no SFT) -- FULL holdout"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL=models/qwen-q4 \
  JAC_EVAL_ADAPTER=model-experiments/03-cpt-only/adapters/cpt-v2 \
  JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP=0 \
  jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac | tee "$RDIR/base.txt"

TMPADP="model-experiments/04-cpt-sft/sft_cptv2_probe/adapters/sft-ckpt-eval"
for CK in "$ADAPTER"/*_adapters.safetensors; do
  [ -e "$CK" ] || continue
  BN="$(basename "$CK")"
  [ "$BN" = "adapters.safetensors" ] && continue   # final handled separately, below, at full holdout
  STEP="$(basename "$CK" | grep -oE '^[0-9]+' | sed 's/^0*//')"; STEP="${STEP:-0}"
  # STEP CORRECTION: training crashed (NaN, since fixed) and was resumed from a
  # true-global-step-820 checkpoint using a FRESH `mlx_lm.lora` invocation
  # (--iters 7380) whose own internal checkpoint counter restarts at 1 -- so
  # every filename's numeric prefix is a LOCAL (per-invocation) step, not the
  # true global step. True global step = local step + 820. (The genuine
  # true-step-820 checkpoint from the pre-crash run was silently overwritten
  # by the resumed run's own first save, which reused the same local-step-820
  # filename -- confirmed via file mtimes; accepted as a harmless one-point gap.)
  STEP=$((STEP + 820))
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
TOTAL_ITERS="$(grep -E '^iters:' model-experiments/04-cpt-sft/sft_cptv2_probe/configs/sft.yaml | grep -oE '[0-9]+' | head -1)"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL=models/qwen-q4 JAC_EVAL_ADAPTER="$ADAPTER" \
  JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$TOTAL_ITERS" \
  jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac | tee "$RDIR/final.txt"

echo "=== functional eval sweep done: $METRICS ==="
