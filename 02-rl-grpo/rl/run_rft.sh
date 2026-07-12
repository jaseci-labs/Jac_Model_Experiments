#!/usr/bin/env bash
# RFT warm-start — bootstrap a cold (non-jac-trained) base before GRPO.
#
# Cold bases emit mostly non-compiling Jac -> GRPO reward is ~all-zero -> stall.
# RFT: sample from the base, keep completions that PASS the jac reward, SFT
# (LoRA) on them, fuse -> a warmed base GRPO can climb from. Skip this for the
# already-jac-trained base (jac-qwen3coder); run it for the fresh ones.
#
# Usage:  RL_BASE=models/qwen-q4 ./02-rl-grpo/rl/run_rft.sh qwen3coder
#   then: RL_BASE=models/qwen3coder-rft-q4 ./02-rl-grpo/rl/run_grpo.sh qwen3coder
# Env: RFT_SAMPLES(8) RFT_TEMP(1.0) RFT_PASS(0.9; lower to 0.6 if 0 pass)
#      RFT_ITERS(150) FORCE_SAMPLE(0 -> reuse existing rft data)
set -euo pipefail

if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu env CAFFEINATED=1 "$0" "$@"
fi
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(cd "$SELF_DIR/../.." && pwd)"   # repo root
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

NAME="${1:?short name, e.g. qwen3coder}"
RL_BASE="${RL_BASE:?set RL_BASE to the cold MLX base, e.g. models/qwen-q4}"
RFT_ITERS="${RFT_ITERS:-150}"

need() { command -v "$1" >/dev/null 2>&1 || { echo "MISSING: $1 ($2)"; exit 1; }; }
need jac "pip install jaclang"
for s in lora fuse; do need "mlx_lm.$s" "pip install mlx-lm"; done
[ -f "$RL_BASE/config.json" ] || { echo "MISSING base model: $RL_BASE"; exit 1; }
[ -f 02-rl-grpo/dataset/rl/train.jsonl ] || { echo "MISSING 02-rl-grpo/dataset/rl/train.jsonl — run: jac run 02-rl-grpo/rl/build_tasks.jac && jac run 02-rl-grpo/rl/build_rl_splits.jac"; exit 1; }

RFT_ADAPTER="02-rl-grpo/adapters/${NAME}-rft"
WARM="models/${NAME}-rft-q4"
RDIR="results/${NAME}/rft"; mkdir -p "$RDIR"

# --- 1. sample passing completions from the base (the rejection-sampling step) ---
if [ "${FORCE_SAMPLE:-0}" != "1" ] && [ -s 02-rl-grpo/dataset/rl/rft/train.jsonl ]; then
  echo ">>> reuse existing 02-rl-grpo/dataset/rl/rft (FORCE_SAMPLE=1 to resample)"
else
  echo ">>> RFT sampling from $RL_BASE (samples=$RFT_SAMPLES temp=${RFT_TEMP:-1.0} pass>=${RFT_PASS:-0.9})"
  JAC_EVAL_MODEL="$RL_BASE" jac run 02-rl-grpo/rl/rft_sample.jac | tee "$RDIR/sample.log"
fi
[ -s 02-rl-grpo/dataset/rl/rft/train.jsonl ] || { echo "!!! no passing samples — base too cold. Retry: RFT_PASS=0.6 FORCE_SAMPLE=1 RFT_SAMPLES=16 ./02-rl-grpo/rl/run_rft.sh $NAME"; exit 1; }

# --- 2. LoRA SFT on the kept completions ---
if [ -f "$RFT_ADAPTER/adapters.safetensors" ]; then
  echo ">>> RFT adapter exists ($RFT_ADAPTER) — skipping SFT (rm it to retrain)"
else
  # mlx_lm.lora needs >= batch_size examples in BOTH train AND valid; low ladder
  # rungs have a tiny gold valid (build_sft_gold carves 1) and crash validation at
  # the default batch 2. Clamp batch to the smaller of train/valid so any rung trains.
  NTRAIN=$(grep -cve '^[[:space:]]*$' 02-rl-grpo/dataset/rl/rft/train.jsonl)
  NVALID=$(grep -cve '^[[:space:]]*$' 02-rl-grpo/dataset/rl/rft/valid.jsonl)
  NMIN=$NTRAIN; [ "$NVALID" -lt "$NMIN" ] && NMIN=$NVALID
  BATCH="${RFT_BATCH:-2}"
  [ "$NMIN" -lt "$BATCH" ] && { BATCH="$NMIN"; echo ">>> small rung: batch -> $BATCH (train=$NTRAIN valid=$NVALID)"; }
  echo ">>> RFT SFT: ${RFT_ITERS} iters on 02-rl-grpo/dataset/rl/rft (layers=${RFT_LAYERS:-16} batch=$BATCH seq=${RFT_MAXSEQ:-2048})"
  # --grad-checkpoint + lean overrides so a DENSE base (Qwen3.6-27B, all-active)
  # fits 48GB. MoE bases are fine on defaults; dense needs RFT_LAYERS=8 etc.
  mlx_lm.lora --config 01-sft-dpo/sft_dpo/configs/lora.yaml --model "$RL_BASE" \
    --data 02-rl-grpo/dataset/rl/rft --adapter-path "$RFT_ADAPTER" --iters "$RFT_ITERS" \
    --num-layers "${RFT_LAYERS:-16}" --batch-size "$BATCH" \
    --max-seq-length "${RFT_MAXSEQ:-2048}" --grad-checkpoint \
    2>&1 | tee "$RDIR/train.log"
  [ -f "$RFT_ADAPTER/adapters.safetensors" ] || { echo "!!! RFT SFT produced no adapter — see $RDIR/train.log"; exit 1; }
fi

# --- 3. fuse -> warmed base for GRPO ---
if [ -f "$WARM/config.json" ]; then echo ">>> reuse $WARM"; else
  echo ">>> fusing RFT adapter -> $WARM"
  mlx_lm.fuse --model "$RL_BASE" --adapter-path "$RFT_ADAPTER" --save-path "$WARM"
fi

echo "=== RFT warm-start done -> $WARM ==="
echo "  next:  RL_BASE=$WARM ./02-rl-grpo/rl/run_grpo.sh $NAME"
