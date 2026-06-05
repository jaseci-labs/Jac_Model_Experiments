#!/usr/bin/env bash
# DPO stage — push the SFT model from Python-shaped toward IDIOMATIC Jac.
#
# Why: the SFT model scores 94% behavioral but ~99% of its correct outputs are
# Python-shaped (idiom_eval.jac: avg transpile-similarity 0.968). DPO on the R3
# de-Python-ification pairs (chosen=idiomatic, rejected=transpile) teaches it to
# prefer idiomatic Jac. Goal: avg_sim DROPS, idiomatic% RISES, behavior holds.
#
# mlx-lm has NO native DPO -> uses the third-party mlx-lm-lora trainer.
# LoRA-DPO: reference-model-path is left unset, so the reference is the (frozen)
# base and only ONE 30B weight set sits in RAM -> fits 48GB (a separate reference
# copy would OOM, same failure as the live-eval gotcha).
#
# Usage: ./run_dpo.sh <short-name>          e.g. ./run_dpo.sh qwen
# Env: DPO_ITERS(200) DPO_LR(1e-6) DPO_BETA(0.1) SUBSET(50, eval tasks)
set -euo pipefail

if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu env CAFFEINATED=1 "$0" "$@"
fi
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -d "$SELF_DIR/.venv/bin" ] && export PATH="$SELF_DIR/.venv/bin:$PATH"

NAME="${1:?short name, e.g. qwen}"
DPO_ITERS="${DPO_ITERS:-200}"; DPO_LR="${DPO_LR:-1e-6}"; DPO_BETA="${DPO_BETA:-0.1}"
SUBSET="${SUBSET:-50}"

# --- prereqs ---
need() { command -v "$1" >/dev/null 2>&1 || { echo "MISSING: $1 ($2)"; exit 1; }; }
need jac "pip install jaclang"
python -c "import mlx_lm_lora" 2>/dev/null || { echo "MISSING: mlx-lm-lora (pip install mlx-lm-lora)"; exit 1; }
SFT_ADAPTER="adapters/${NAME}-probe"
[ -f "$SFT_ADAPTER/adapters.safetensors" ] || { echo "MISSING: $SFT_ADAPTER/adapters.safetensors — run run_probe.sh first"; exit 1; }
for f in dataset/mlx_dpo/train.jsonl dataset/mlx_dpo/valid.jsonl; do
  [ -f "$f" ] || { echo "MISSING $f — run: jac run srccurrent/jacgen/build_dpo_splits.jac"; exit 1; }
done

RDIR="results/${NAME}/dpo"; mkdir -p "$RDIR"
DPO_ADAPTER="adapters/${NAME}-dpo"
SFT_Q4="models/${NAME}-jac-fused-q4"     # DPO trains on the SFT model (Q4 = lighter)
SFT_Q8="models/${NAME}-jac-fused-q8"     # already built by run_probe.sh (fuse stage)
DPO_FUSED="models/${NAME}-jac-dpo-fused-q8"
TRAIN_LOG="$RDIR/train.log"

# --- 1. build the Q4 SFT-fused base to DPO on top of (idempotent) ---
if [ -f "$SFT_Q4/config.json" ]; then echo ">>> reuse $SFT_Q4"; else
  echo ">>> fusing SFT adapter into a Q4 base for DPO"
  mlx_lm.fuse --model "models/${NAME}-q4" --adapter-path "$SFT_ADAPTER" --save-path "$SFT_Q4"
fi

# --- 2. DPO train (LoRA, reference = frozen base; resumes nothing, it's short) ---
if [ -f "$DPO_ADAPTER/adapters.safetensors" ]; then
  echo ">>> DPO adapter exists ($DPO_ADAPTER) — skipping train (rm it to retrain)"
else
  echo ">>> DPO training: ${DPO_ITERS} iters, beta=${DPO_BETA}, lr=${DPO_LR}"
  # --grad-checkpoint: DPO runs 4 forward passes (policy+reference x chosen+rejected);
  # without checkpointing the activation peak OOMs a 30B on 48GB (Metal). Data maxes
  # ~370 tokens so 512 max-seq is safe and caps any spike.
  # DPO on a 30B is at the 48GB Metal ceiling (policy+reference x chosen+rejected,
  # 4 forward graphs, memory grows across iters). Minimize: fewer LoRA layers (8),
  # grad-checkpoint, short max-seq (data maxes ~370), and NO mid-train val
  # (steps-per-eval high) since the val pass is an extra memory spike.
  DPO_LAYERS="${DPO_LAYERS:-8}"
  python -m mlx_lm_lora.train \
    --model "$SFT_Q4" --train --data dataset/mlx_dpo --train-mode dpo \
    --adapter-path "$DPO_ADAPTER" --train-type lora --num-layers "$DPO_LAYERS" --grad-checkpoint \
    --batch-size 1 --max-seq-length 384 --iters "$DPO_ITERS" \
    --learning-rate "$DPO_LR" --beta "$DPO_BETA" --dpo-cpo-loss-type sigmoid \
    --steps-per-report 10 --steps-per-eval 100000 --val-batches 1 --save-every 50 \
    2>&1 | tee "$TRAIN_LOG"
  [ -f "$DPO_ADAPTER/adapters.safetensors" ] || { echo "!!! DPO produced no adapter — see $TRAIN_LOG"; exit 1; }
fi

# --- 3. fuse DPO adapter onto the Q8 SFT model -> final DPO model ---
if [ -f "$DPO_FUSED/config.json" ]; then echo ">>> reuse $DPO_FUSED"; else
  echo ">>> fusing DPO adapter -> $DPO_FUSED"
  mlx_lm.fuse --model "$SFT_Q8" --adapter-path "$DPO_ADAPTER" --save-path "$DPO_FUSED"
fi

# --- 4. eval DPO model: behavior (must hold) + idiom (must improve) ---
echo ">>> DPO eval: behavior"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="$DPO_FUSED" \
  jac run srccurrent/jacgen/eval_probe.jac | tee "$RDIR/finetuned.txt"
echo ">>> DPO eval: idiom"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="$DPO_FUSED" JAC_IDIOM_OUT="$RDIR/idiom-metrics.jsonl" \
  jac run srccurrent/jacgen/idiom_eval.jac | tee "$RDIR/idiom.txt"

echo "=== DPO done ==="
echo "  model:    $DPO_FUSED"
echo "  behavior: $RDIR/finetuned.txt   (compare to results/${NAME}/finetuned.txt)"
echo "  idiom:    $RDIR/idiom.txt        (compare avg_sim to the SFT idiom: results/${NAME}/idiom-finetuned.txt)"
echo "  WIN = behavior holds AND avg transpile-similarity drops (more idiomatic, less transpile-shaped)."
