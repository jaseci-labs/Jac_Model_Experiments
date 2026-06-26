#!/usr/bin/env bash
# Run the two fresh-base models end-to-end with the fixed reward/splice:
#   convert (qwen36 only) -> gold-SFT warm-start -> base/warm/grpo evals -> GRPO.
# jac-qwen3coder is done separately (it's the jac-trained base). Reuses the
# model-agnostic gold warm-start set in dataset/rl/rft (build it first).
#
# Usage: ./rl/run_remaining.sh            (runs both, sequential — single GPU)
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"
export CAFFEINATED=1

# one eval -> file. args: model adapter holdout outfile
ev() {
  local m="$1" a="$2" h="$3" o="$4"
  if [ -n "$a" ]; then
    JAC_EVAL_MODEL="$m" JAC_EVAL_ADAPTER="$a" JAC_RL_HOLDOUT="$h" jac run rl/eval_rl.jac 2>&1 | tail -8 > "$o"
  else
    JAC_EVAL_MODEL="$m" JAC_RL_HOLDOUT="$h" jac run rl/eval_rl.jac 2>&1 | tail -8 > "$o"
  fi
}

# full pipeline for one model. args: NAME BASE_Q4 GROUP_SIZE
run_model() {
  local NAME="$1" BASE="$2" GS="$3"
  local R="results/${NAME}/final"; mkdir -p "$R"
  echo "############ $NAME  (base=$BASE)  $(date) ############"

  echo ">>> [$NAME] base evals"
  ev "$BASE" "" dataset/rl/holdout.jsonl "$R/base-holdout.txt"
  ev "$BASE" "" dataset/rl/train.jsonl   "$R/base-train.txt"

  echo ">>> [$NAME] gold-SFT warm-start"
  RL_BASE="$BASE" RFT_ITERS=200 ./rl/run_rft.sh "$NAME" || { echo "!! $NAME warm-start failed"; return 1; }
  local WARM="models/${NAME}-rft-q4"
  ev "$WARM" "" dataset/rl/holdout.jsonl "$R/warm-holdout.txt"
  ev "$WARM" "" dataset/rl/train.jsonl   "$R/warm-train.txt"

  echo ">>> [$NAME] GRPO (LR 1e-5, 300it, temp 1.0, group $GS)"
  rm -rf "adapters/${NAME}-grpo"
  RL_BASE="$WARM" GRPO_ITERS=300 GRPO_LR=1e-5 GRPO_TEMP=1.0 GROUP_SIZE="$GS" \
    MAX_COMPLETION=256 MAX_SEQ=1280 GRPO_LAYERS=8 ./rl/run_grpo.sh "$NAME" || { echo "!! $NAME grpo failed"; return 1; }
  ev "$WARM" "adapters/${NAME}-grpo" dataset/rl/holdout.jsonl "$R/grpo-holdout.txt"
  ev "$WARM" "adapters/${NAME}-grpo" dataset/rl/train.jsonl   "$R/grpo-train.txt"
  echo ">>> [$NAME] DONE $(date)"
}

# ensure gold warm-start set exists (model-agnostic)
[ -s dataset/rl/rft/train.jsonl ] || jac run rl/build_sft_gold.jac

# --- qwen3coder: fresh Qwen3-Coder-30B-A3B (base already local) ---
run_model qwen3coder models/qwen-q4 4

# --- qwen36: Qwen3.6-27B dense (convert from HF cache first) ---
if [ ! -f models/qwen36-q4/config.json ]; then
  echo ">>> converting Qwen/Qwen3.6-27B -> models/qwen36-q4 $(date)"
  mlx_lm.convert --hf-path Qwen/Qwen3.6-27B --mlx-path models/qwen36-q4 -q --q-bits 4
fi
run_model qwen36 models/qwen36-q4 3   # dense -> smaller group

echo "############ ALL REMAINING DONE $(date) ############"
