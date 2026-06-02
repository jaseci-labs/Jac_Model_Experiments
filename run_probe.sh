#!/usr/bin/env bash
# Conversion-probe runner (SCAFFOLD — not executed by the data-prep work).
# Prerequisites the data-prep does NOT do for you:
#   pip install mlx-lm
#   ~50-60GB disk per model
#
# Usage:
#   ./run_probe.sh <hf-model-id> <short-name>
#   ./run_probe.sh Qwen/Qwen3-Coder-30B-A3B qwen
#
# Steps: quantize -> base eval -> LoRA SFT -> fuse -> finetuned eval -> compare.
# The eval harness reads JAC_EVAL_MODE / JAC_EVAL_MODEL from the environment, so
# nothing edits source files.
set -euo pipefail

HF_MODEL="${1:?hf model id, e.g. Qwen/Qwen3-Coder-30B-A3B}"
NAME="${2:?short name, e.g. qwen}"
mkdir -p models adapters results

# 1. quantize: Q4 for training, Q8 for eval
mlx_lm.convert --hf-path "$HF_MODEL" --mlx-path "models/${NAME}-q4" -q --q-bits 4
mlx_lm.convert --hf-path "$HF_MODEL" --mlx-path "models/${NAME}-q8" -q --q-bits 8

# 2. base eval (before finetuning)
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="models/${NAME}-q8" \
  jac run srccurrent/jacgen/eval_probe.jac | tee "results/${NAME}-base.txt"

# 3. LoRA SFT  (set `model: models/${NAME}-q4` in configs/lora.yaml first)
mlx_lm.lora --config configs/lora.yaml

# 4. fuse adapter into the Q8 eval model
mlx_lm.fuse --model "models/${NAME}-q8" \
  --adapter-path adapters/conversion-probe \
  --save-path "models/${NAME}-jac-fused-q8"

# 5. finetuned eval
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="models/${NAME}-jac-fused-q8" \
  jac run srccurrent/jacgen/eval_probe.jac | tee "results/${NAME}-finetuned.txt"

echo "=== compare base vs finetuned ==="
echo "  results/${NAME}-base.txt"
echo "  results/${NAME}-finetuned.txt"
