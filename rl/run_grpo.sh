#!/usr/bin/env bash
# GRPO stage — reinforce IDIOMATIC, COMPILER-CORRECT Jac with a verifiable reward.
#
# Reward (rl/reward.py:jac_behavioral): splice each sampled completion into its
# this_is_jac task template, `jac run` it, score compiles + runs + output-match +
# idiom. No learned reward model — the Jac compiler/behavior IS the reward.
#
# LoRA-GRPO: reference-model-path is left unset, so the reference is the frozen
# base and only ONE weight set sits in RAM -> fits 48GB (same trick as run_dpo.sh).
#
# Usage:  RL_BASE=<mlx-model-path> ./rl/run_grpo.sh <short-name>
#   e.g.  RL_BASE=models/qwen-q4 ./rl/run_grpo.sh qwen3
# Env: GRPO_ITERS(200) GRPO_LR(1e-6) GRPO_BETA(0.04) GROUP_SIZE(4)
#      MAX_COMPLETION(256) MAX_SEQ(1280) GRPO_TEMP(1.0) GRPO_LAYERS(8)
# Defaults are tuned to FIT a 30B-A3B q4 on a 48GB box (peak ~38GB, no OOM).
# group6/comp512 OOMs Metal at ~iter 2. Bodies are short so comp256 is ample.
set -euo pipefail

if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu env CAFFEINATED=1 "$0" "$@"
fi
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(cd "$SELF_DIR/.." && pwd)"   # repo root: dataset/ models/ adapters/ results/ resolve here
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

NAME="${1:?short name, e.g. qwen3}"
RL_BASE="${RL_BASE:?set RL_BASE to an MLX model path, e.g. models/qwen-q4}"
GRPO_ITERS="${GRPO_ITERS:-200}"; GRPO_LR="${GRPO_LR:-1e-6}"; GRPO_BETA="${GRPO_BETA:-0.04}"
GROUP_SIZE="${GROUP_SIZE:-4}"; MAX_COMPLETION="${MAX_COMPLETION:-256}"
MAX_SEQ="${MAX_SEQ:-1280}"; GRPO_TEMP="${GRPO_TEMP:-1.0}"; GRPO_LAYERS="${GRPO_LAYERS:-8}"

# --- prereqs ---
need() { command -v "$1" >/dev/null 2>&1 || { echo "MISSING: $1 ($2)"; exit 1; }; }
need jac "pip install jaclang"
python -c "import mlx_lm_lora" 2>/dev/null || { echo "MISSING: mlx-lm-lora (pip install mlx-lm-lora)"; exit 1; }
python -c "import sys; sys.path.insert(0,'rl'); import reward; from mlx_lm_lora.trainer.grpo_reward_functions import get_reward_function; get_reward_function('jac_behavioral')" \
  || { echo "MISSING/BROKEN: rl/reward.py jac_behavioral did not register"; exit 1; }
[ -f "$RL_BASE/config.json" ] || { echo "MISSING base model: $RL_BASE"; exit 1; }
for f in dataset/rl/train.jsonl dataset/rl/valid.jsonl; do
  [ -f "$f" ] || { echo "MISSING $f — run: jac run rl/build_tasks.jac && jac run rl/build_rl_splits.jac"; exit 1; }
done

RDIR="results/${NAME}/grpo"; mkdir -p "$RDIR"
GRPO_ADAPTER="adapters/${NAME}-grpo"
TRAIN_LOG="$RDIR/train.log"

echo ">>> GRPO: base=$RL_BASE  iters=$GRPO_ITERS  group=$GROUP_SIZE  beta=$GRPO_BETA  lr=$GRPO_LR"
echo ">>> adapter -> $GRPO_ADAPTER   log -> $TRAIN_LOG"

python -m mlx_lm_lora.train \
  --model "$RL_BASE" --train --data dataset/rl --train-mode grpo \
  --adapter-path "$GRPO_ADAPTER" --train-type lora --num-layers "$GRPO_LAYERS" --grad-checkpoint \
  --group-size "$GROUP_SIZE" --max-completion-length "$MAX_COMPLETION" --temperature "$GRPO_TEMP" \
  --reward-functions-file rl/reward.py --reward-functions jac_behavioral \
  --batch-size 1 --max-seq-length "$MAX_SEQ" --iters "$GRPO_ITERS" \
  --learning-rate "$GRPO_LR" --beta "$GRPO_BETA" \
  --steps-per-report 5 --steps-per-eval 100000 --val-batches 1 --save-every 50 \
  2>&1 | tee "$TRAIN_LOG"

[ -f "$GRPO_ADAPTER/adapters.safetensors" ] || { echo "!!! GRPO produced no adapter — see $TRAIN_LOG"; exit 1; }
echo "=== GRPO done -> $GRPO_ADAPTER ==="
echo "  eval:  JAC_EVAL_MODEL=$RL_BASE JAC_EVAL_ADAPTER=$GRPO_ADAPTER jac run rl/eval_rl.jac"
