#!/bin/bash
# Post-probe bake-off steps for a STANDARD model (not gptoss).
# Assumes run_probe.sh already produced models/<N>-q8, models/<N>-jac-fused-q8,
# 01-sft-dpo/results/<N>/{base,finetuned}.txt, metrics.jsonl. Adds: function SFT idiom baseline,
# graph holdout (base/SFT/idiom), DPO (function behavior+idiom built in), graph DPO.
set -uo pipefail
cd /Volumes/ExtremePro/JaseciLabs/jac_model_studio
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"
N="$1"
GH=01-sft-dpo/dataset/eval_holdout/graph_conversion.jsonl
Q8="models/${N}-q8"
FUSED="models/${N}-jac-fused-q8"
DF="models/${N}-jac-dpo-fused-q8"
EP=01-sft-dpo/sft_dpo/jacgen/eval_probe.jac
IE=01-sft-dpo/sft_dpo/jacgen/idiom_eval.jac
mkdir -p "01-sft-dpo/results/${N}/dpo"

echo "## [${N}] function SFT idiom baseline"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="$FUSED" JAC_IDIOM_OUT="01-sft-dpo/results/${N}/idiom-metrics.jsonl" \
  jac run "$IE" | tee "01-sft-dpo/results/${N}/idiom-finetuned.txt"

echo "## [${N}] graph base"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="$Q8" JAC_HOLDOUT="$GH" \
  jac run "$EP" | tee "01-sft-dpo/results/${N}/graph-base.txt"
echo "## [${N}] graph SFT"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="$FUSED" JAC_HOLDOUT="$GH" \
  jac run "$EP" | tee "01-sft-dpo/results/${N}/graph-finetuned.txt"
echo "## [${N}] graph SFT idiom"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="$FUSED" JAC_HOLDOUT="$GH" JAC_IDIOM_OUT="01-sft-dpo/results/${N}/graph-idiom-metrics.jsonl" \
  jac run "$IE" | tee "01-sft-dpo/results/${N}/graph-idiom-finetuned.txt"

echo "## [${N}] DPO (function behavior + idiom built in)"
caffeinate -s ./01-sft-dpo/sft_dpo/run_dpo.sh "$N"

echo "## [${N}] graph DPO behavior"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="$DF" JAC_HOLDOUT="$GH" \
  jac run "$EP" | tee "01-sft-dpo/results/${N}/dpo/graph-finetuned.txt"
echo "## [${N}] graph DPO idiom"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="$DF" JAC_HOLDOUT="$GH" JAC_IDIOM_OUT="01-sft-dpo/results/${N}/dpo/graph-idiom-metrics.jsonl" \
  jac run "$IE" | tee "01-sft-dpo/results/${N}/dpo/graph-idiom-finetuned.txt"

echo "## POSTPROBE DONE ${N}"
