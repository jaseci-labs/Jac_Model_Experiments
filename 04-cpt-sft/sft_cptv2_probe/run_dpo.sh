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
DPO_MAXLEN="${DPO_MAXLEN:-512}"     # dataset/dpo/{train,valid}.jsonl pre-filtered to 1024 (Task 6);
                                    # re-filtered to 512 is NOT required -- longer examples just
                                    # truncate safely (see note below), no NaN risk either way.
                                    # NOT matched to SFT's 3072: DPO here loads TWO full model
                                    # copies (policy + reference -- mlx_lm_lora always loads a
                                    # fresh `load(args.model)` for the reference when
                                    # --reference-model-path is unset), so memory headroom is
                                    # much tighter than SFT's single-copy training.
                                    #
                                    # 1024 was Task 6's original choice (passed an 8-iter
                                    # dry-run cleanly) but OOM'd the real 250-iter run at iter 10
                                    # (Task 8, first attempt) -- peak_mem was still CLIMBING at
                                    # iter 8 of the dry-run (38.7->39.1->39.8->39.9GB), which an
                                    # 8-iter window can't distinguish from "stable." Root cause:
                                    # mlx_lm_lora's DPO trainer keeps a persistent, reused
                                    # KV-cache (`make_prompt_cache`, dpo_trainer.py:291/293) for
                                    # both policy and reference models, grown once to the longest
                                    # sequence seen so far and never shrunk -- `mx.clear_cache()`
                                    # (dpo_trainer.py:106) clears the general allocator, not this
                                    # cache object's already-allocated buffer. So memory climbs
                                    # early in training then plateaus, not unbounded -- but the
                                    # plateau height scales with max_seq_length, and 1024's
                                    # plateau exceeds the ~48GB ceiling before settling.
                                    #
                                    # Verified 512 properly this time: a real 40-ITER dry-run
                                    # (not just 8) shows peak_mem climbs once (39.639->39.685GB
                                    # by iter 15) then stays FLAT through iter 40 -- a genuine
                                    # plateau with ~8GB safe margin, not a short window getting
                                    # lucky. Truncation here is safe (no NaN risk): this trainer
                                    # masks the WHOLE truncated sequence, not a per-token prompt
                                    # mask, so long examples just lose tail signal, unlike the
                                    # SFT mask_prompt bug the dataset filter guards against.
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

DRY_DONE_MARK="$RDIR/.dry.done"
if [ ! -f "$DRY_DONE_MARK" ]; then
  echo ">>> DPO dry-run (8 iters) -- bail check"
  python -m mlx_lm_lora.train --model "$SFT_FUSED" --train --data model-experiments/04-cpt-sft/sft_cptv2_probe/dataset/dpo \
    --train-mode dpo --config model-experiments/04-cpt-sft/sft_cptv2_probe/configs/dpo_lora.yaml \
    --adapter-path model-experiments/04-cpt-sft/sft_cptv2_probe/adapters/dpo-dry \
    --train-type lora --num-layers 16 --grad-checkpoint --batch-size 1 --max-seq-length "$DPO_MAXLEN" \
    --iters 8 --learning-rate "$DPO_LR" --beta "$DPO_BETA" --dpo-cpo-loss-type sigmoid \
    --steps-per-report 2 --steps-per-eval 100000 --val-batches 1 --save-every 100 2>&1 | tail -25
  touch "$DRY_DONE_MARK"
fi

# Persisted-state gate (same pattern as run_sft.sh's fix): only waives once a
# real DPO checkpoint already exists, so it re-triggers correctly on every
# invocation, not just the process that happened to run the dry-run --
# session-local flags reset every invocation, which is exactly the bug this
# script had (auto-continued into the real 250-iter run unattended, unlike
# run_sft.sh which was fixed for this after the same failure mode).
if [ ! -f "$DPO_ADAPTER/adapters.safetensors" ] && [ "${CONFIRM_FULL_RUN:-}" != "1" ]; then
  echo "Dry-run complete. Re-run with CONFIRM_FULL_RUN=1 to start the real DPO training run."
  exit 0
fi

# Resumable, SEGMENTED training loop. Real crashes hit twice at DIFFERENT
# max_seq_length values (iter 10 at 1024, iter 51 at 512) -- and a 3rd crash
# after resuming from checkpoint 50 landed again before reaching 100, i.e.
# within roughly the same ~50-iters-since-process-start window both times,
# starting from DIFFERENT points in the dataset shuffle. That pattern points
# at PROCESS-LIFETIME-driven cache growth (dpo_trainer.py's persistent,
# never-shrunk policy+reference KV-cache, dpo_trainer.py:291/293) rather than
# purely "which long example happens to appear" -- a fixed seq_length or a
# short dry-run can't rule this out. Real fix: cap each invocation to a small
# SEGMENT_ITERS regardless of how much overall remains, forcing a fresh
# process restart (which resets the cache to empty) well before the observed
# ~50-iter danger zone, instead of requesting the full remaining count in one
# continuous process.
#
# IMPORTANT: mlx_lm_lora's checkpoint filenames use a per-invocation LOCAL
# iteration counter too (the same bug class already documented for
# mlx_lm.lora's SFT checkpoints) -- a freshly-resumed 20-iter segment saves
# "0000020_adapters.safetensors", which a numeric sort ranks BELOW an older
# "0000050" file despite being chronologically newer real progress. Confirmed
# live: this caused the first version of this loop to silently re-resume from
# the same stale checkpoint 5 times in a row, discarding each segment's real
# progress and never advancing (a genuine infinite loop that LOOKED like
# repeated retries but was actually a bookkeeping bug, not a memory crash --
# each individual segment was completing successfully). Fixed by tracking
# progress in our own explicit counter file instead of trying to infer it
# from the library's unreliable checkpoint filenames, and always resuming
# from mlx_lm_lora's "adapters.safetensors" (its own always-current-state
# pointer file, confirmed via mtime to update after every segment,
# regardless of that segment's numbered-checkpoint filename).
SEGMENT_ITERS="${DPO_SEGMENT_ITERS:-20}"
PROGRESS_FILE="$RDIR/.dpo_progress_steps"
[ -f "$PROGRESS_FILE" ] || echo 0 > "$PROGRESS_FILE"
: > "$RDIR/train.log"   # fresh log for this invocation's segments (each appended below)
CONSECUTIVE_FAILS=0
while true; do
  DONE_STEPS="$(cat "$PROGRESS_FILE")"
  REMAIN=$(( DPO_ITERS - DONE_STEPS ))
  if [ "$REMAIN" -le 0 ]; then break; fi
  SEG_ITERS=$(( REMAIN < SEGMENT_ITERS ? REMAIN : SEGMENT_ITERS ))
  RESUME_FLAG=()
  if [ "$DONE_STEPS" -gt 0 ] && [ -f "$DPO_ADAPTER/adapters.safetensors" ]; then
    RESUME_FLAG=(--resume-adapter-file "$DPO_ADAPTER/adapters.safetensors")
  fi
  echo ">>> DPO training segment: ${SEG_ITERS} iters this process (${DONE_STEPS}/${DPO_ITERS} done, ${REMAIN} remaining overall)" | tee -a "$RDIR/train.log"
  RC=0
  python -m mlx_lm_lora.train --model "$SFT_FUSED" --train --data model-experiments/04-cpt-sft/sft_cptv2_probe/dataset/dpo \
    --train-mode dpo --config model-experiments/04-cpt-sft/sft_cptv2_probe/configs/dpo_lora.yaml \
    --adapter-path "$DPO_ADAPTER" --train-type lora --num-layers 16 --grad-checkpoint \
    --batch-size 1 --max-seq-length "$DPO_MAXLEN" --iters "$SEG_ITERS" "${RESUME_FLAG[@]}" \
    --learning-rate "$DPO_LR" --beta "$DPO_BETA" --dpo-cpo-loss-type sigmoid \
    --steps-per-report 10 --steps-per-eval "$SEG_ITERS" --val-batches 1 --save-every "$SEG_ITERS" \
    >> "$RDIR/train.log" 2>&1 || RC=$?
  if [ "$RC" -ne 0 ]; then
    CONSECUTIVE_FAILS=$(( CONSECUTIVE_FAILS + 1 ))
    if [ "$CONSECUTIVE_FAILS" -ge 5 ]; then
      echo "!!! segment crashed ${CONSECUTIVE_FAILS}x in a row at the same ${DONE_STEPS}/${DPO_ITERS} point -- giving up." | tee -a "$RDIR/train.log"
      tail -20 "$RDIR/train.log"; exit 1
    fi
    echo "!!! segment crashed (exit $RC, attempt ${CONSECUTIVE_FAILS}/5) -- retrying same segment from last good state" | tee -a "$RDIR/train.log"
    continue
  fi
  CONSECUTIVE_FAILS=0
  echo $(( DONE_STEPS + SEG_ITERS )) > "$PROGRESS_FILE"
done
echo "=== DPO training done: $RDIR/train.log ==="
