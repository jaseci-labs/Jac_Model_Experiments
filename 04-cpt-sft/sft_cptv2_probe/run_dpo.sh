#!/usr/bin/env bash
# DPO-on-SFT-on-CPT-v2 runner. rank16/layers16 (NOT mlx_lm_lora's rank8/layers8
# default) for consistency with every other stage in this probe -- there is no
# technical requirement to match here (this is a FRESH adapter on a FUSED
# model, not a --resume-adapter-file stack), it's a deliberate choice.
set -euo pipefail
if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu env CAFFEINATED=1 "$0" "$@"
fi
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

need() { command -v "$1" >/dev/null 2>&1 || { echo "MISSING: $1"; exit 1; }; }
need mlx_lm.fuse "pip install mlx-lm"
python3 -c "import mlx_lm_lora" || { echo "MISSING: mlx-lm-lora (pip install mlx-lm-lora)"; exit 1; }

SFT_ADAPTER="model-experiments/04-cpt-sft/sft_cptv2_probe/adapters/sft-on-cptv2"
SFT_FUSED="models/sft-cptv2-fused-q4"
DPO_ADAPTER="model-experiments/04-cpt-sft/sft_cptv2_probe/adapters/dpo-on-sft"
RDIR="model-experiments/04-cpt-sft/sft_cptv2_probe/results/dpo"
DPO_ITERS="${DPO_ITERS:-250}"
DPO_LR="${DPO_LR:-1e-6}"
DPO_BETA="${DPO_BETA:-0.1}"
DPO_MAXLEN="${DPO_MAXLEN:-1024}"   # dataset/dpo/{train,valid}.jsonl pre-filtered to this bound.
                                    # NOT matched to SFT's 3072: DPO here loads TWO full model
                                    # copies (policy + reference -- mlx_lm_lora always loads a
                                    # fresh `load(args.model)` for the reference when
                                    # --reference-model-path is unset), so memory headroom is
                                    # much tighter than SFT's single-copy training. Empirically
                                    # (Task 6): max_seq_length 3072 OOM'd mid-training at iter 1
                                    # (a real, seq-length-driven crash); 1024 and 512 both
                                    # completed all 8 dry-run iters cleanly with sane losses
                                    # (~0.67-0.75, no NaN/Inf) -- 1024 chosen for better tail
                                    # coverage of dataset_stats.json's dpo p99=2497 (chat-
                                    # template real stats) while staying inside the verified-safe
                                    # range. Truncation here is safe (no NaN risk): this trainer
                                    # masks the WHOLE truncated sequence, not a per-token prompt
                                    # mask, so long examples just lose tail signal, unlike the
                                    # SFT mask_prompt bug this filtering step guards against.
                                    #
                                    # A SEPARATE, unrelated bug was also fixed in configs/
                                    # dpo_lora.yaml (fuse: false): mlx_lm_lora defaults to
                                    # fuse:true, which after training unconditionally dequantizes
                                    # the full 30.5B-param model to fp16 (~61GB) and OOMs no
                                    # matter what max_seq_length is -- this is what actually
                                    # crashed the first three dry-run attempts (all AFTER
                                    # training itself finished 8/8 iters cleanly), not the
                                    # training loop.

mkdir -p "$RDIR"
[ -f "$SFT_ADAPTER/adapters.safetensors" ] || { echo "!!! SFT adapter missing, finish Task 3 first"; exit 1; }

echo ">>> fuse SFT adapter -> $SFT_FUSED"
if [ -f "$SFT_FUSED/config.json" ]; then echo "  reuse $SFT_FUSED"; else
  mlx_lm.fuse --model models/qwen-q4 --adapter-path "$SFT_ADAPTER" --save-path "$SFT_FUSED"
fi

echo ">>> DPO dry-run (8 iters) -- bail check"
python -m mlx_lm_lora.train --model "$SFT_FUSED" --train --data model-experiments/04-cpt-sft/sft_cptv2_probe/dataset/dpo \
  --train-mode dpo --config model-experiments/04-cpt-sft/sft_cptv2_probe/configs/dpo_lora.yaml \
  --adapter-path model-experiments/04-cpt-sft/sft_cptv2_probe/adapters/dpo-dry \
  --train-type lora --num-layers 16 --grad-checkpoint --batch-size 1 --max-seq-length "$DPO_MAXLEN" \
  --iters 8 --learning-rate "$DPO_LR" --beta "$DPO_BETA" --dpo-cpo-loss-type sigmoid \
  --steps-per-report 2 --steps-per-eval 100000 --val-batches 1 --save-every 100 2>&1 | tail -25
echo ">>> dry-run done -- Ctrl-C within 8s to abort"; sleep 8

echo ">>> DPO training ($DPO_ITERS iters)"
: > "$RDIR/train.log"
python -m mlx_lm_lora.train --model "$SFT_FUSED" --train --data model-experiments/04-cpt-sft/sft_cptv2_probe/dataset/dpo \
  --train-mode dpo --config model-experiments/04-cpt-sft/sft_cptv2_probe/configs/dpo_lora.yaml \
  --adapter-path "$DPO_ADAPTER" --train-type lora --num-layers 16 --grad-checkpoint \
  --batch-size 1 --max-seq-length "$DPO_MAXLEN" --iters "$DPO_ITERS" \
  --learning-rate "$DPO_LR" --beta "$DPO_BETA" --dpo-cpo-loss-type sigmoid \
  --steps-per-report 10 --steps-per-eval 50 --val-batches 1 --save-every 50 \
  > "$RDIR/train.log" 2>&1
echo "=== DPO training done: $RDIR/train.log ==="
