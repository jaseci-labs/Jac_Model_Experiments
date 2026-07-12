#!/usr/bin/env bash
# STaR / expert-iteration loop — iterate the warm-start (the thing that works).
#
# Round 0 seeds an SFT set with the GOLD bodies. Each round: SFT from the ORIGINAL
# base on the accumulated set, eval (pass@1 + pass@k), then sample from the new
# model and KEEP the completions that pass exactly -> add them to the set. Repeat.
# Re-SFTing from base (not the previous round) avoids compounding drift.
#
# Usage:  RL_BASE=models/qwen-q4 ./02-rl-grpo/rl/run_star.sh qwen3coder
# Env: STAR_ROUNDS(2) STAR_SAMPLES(6) STAR_PASS(0.9 exact) STAR_TEMP(1.0)
#      RFT_ITERS(200) EVAL_K(4)  + RFT_LAYERS/RFT_BATCH/RFT_MAXSEQ for dense bases
set -uo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"
export CAFFEINATED=1

NAME="${1:?short name}"
RL_BASE="${RL_BASE:?set RL_BASE to the q4 base}"
ROUNDS="${STAR_ROUNDS:-2}"; SAMPLES="${STAR_SAMPLES:-6}"; PASS="${STAR_PASS:-0.9}"
TEMP="${STAR_TEMP:-1.0}"; EVAL_K="${EVAL_K:-4}"
STAR="results/${NAME}/star"; mkdir -p "$STAR"
SFTSET="02-rl-grpo/dataset/rl/star/${NAME}"; mkdir -p "$SFTSET"

ev() {  # model -> appends to a file. args: model outfile
  JAC_EVAL_MODEL="$1" JAC_RL_HOLDOUT=02-rl-grpo/dataset/rl/holdout.jsonl JAC_EVAL_K="$EVAL_K" JAC_EVAL_TEMP="$TEMP" \
    jac run 02-rl-grpo/rl/eval_rl.jac 2>&1 | tail -9 > "$2"
}

# seed the accumulated SFT set with the gold bodies
[ -s 02-rl-grpo/dataset/rl/rft/train.jsonl ] || jac run 02-rl-grpo/rl/build_sft_gold.jac >/dev/null 2>&1
jac run 02-rl-grpo/rl/build_sft_gold.jac >/dev/null 2>&1   # gold from current train split
cp 02-rl-grpo/dataset/rl/rft/train.jsonl "$SFTSET/sft.jsonl"
cp 02-rl-grpo/dataset/rl/rft/valid.jsonl "$SFTSET/valid.jsonl"

for r in $(seq 0 "$ROUNDS"); do
  echo "############ $NAME STaR round $r  $(date)  (set=$(wc -l < "$SFTSET/sft.jsonl") ex) ############"
  # SFT from the ORIGINAL base on the accumulated set
  cp "$SFTSET/sft.jsonl" 02-rl-grpo/dataset/rl/rft/train.jsonl
  cp "$SFTSET/valid.jsonl" 02-rl-grpo/dataset/rl/rft/valid.jsonl
  RNAME="${NAME}-star${r}"
  rm -rf "02-rl-grpo/adapters/${RNAME}-rft" "models/${RNAME}-rft-q4"
  RL_BASE="$RL_BASE" RFT_ITERS="${RFT_ITERS:-200}" ./02-rl-grpo/rl/run_rft.sh "$RNAME" || { echo "!! round $r SFT failed"; break; }
  MODEL="models/${RNAME}-rft-q4"
  ev "$MODEL" "$STAR/round${r}.txt"
  echo ">>> round $r eval:"; grep -E 'pass@1:|pass@' "$STAR/round${r}.txt"

  # sample from the new model, keep exact-passing completions, grow the set
  if [ "$r" -lt "$ROUNDS" ]; then
    rm -rf "$SFTSET/samples${r}"
    RFT_OUT="$SFTSET/samples${r}" RFT_SRC=02-rl-grpo/dataset/rl/train.jsonl RFT_SAMPLES="$SAMPLES" \
      RFT_PASS="$PASS" RFT_TEMP="$TEMP" JAC_EVAL_MODEL="$MODEL" jac run 02-rl-grpo/rl/rft_sample.jac \
      2>&1 | tail -4
    if [ -s "$SFTSET/samples${r}/train.jsonl" ]; then
      cat "$SFTSET/samples${r}/train.jsonl" >> "$SFTSET/sft.jsonl"
      # exact-dedup the accumulated set
      sort -u "$SFTSET/sft.jsonl" -o "$SFTSET/sft.jsonl"
      echo ">>> added $(wc -l < "$SFTSET/samples${r}/train.jsonl") new samples"
    else
      echo ">>> round $r produced no new passing samples"
    fi
  fi
done
echo "############ $NAME STaR done $(date) ############"
echo "per-round holdout pass:"; for r in $(seq 0 "$ROUNDS"); do echo "  round $r:"; grep -E 'pass@1:|pass@' "$STAR/round${r}.txt" 2>/dev/null; done
