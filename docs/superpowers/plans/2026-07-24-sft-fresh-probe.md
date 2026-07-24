# SFT+DPO on Fresh (non-CPT) Qwen — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** train base Qwen (`models/qwen-q4`, no CPT-v2 adapter) with the exact same SFT recipe, dataset split, holdout, and DPO-v1 config already used for the CPT-v2 arm (`sft_cptv2_probe/`), so the two arms differ in exactly one variable — whether CPT-v2 sits underneath — and their functional-eval numbers are directly diffable.

**Architecture:** new sibling directory `model-experiments/04-cpt-sft/sft_fresh_probe/`, structurally mirroring `sft_cptv2_probe/` (configs/adapters/dataset/results/jacgen). Dataset splits are copied byte-for-byte from the CPT-v2 probe (not regenerated) so both arms share the identical 1428-row holdout. The only substantive change from the CPT-v2 probe's scripts is removing `resume_adapter_file` from `sft.yaml` and adjusting paths. A final cross-arm task reads both probes' `metrics_functional.jsonl` files and produces the actual "did CPT help" comparison.

**Tech Stack:** `mlx_lm.lora` (SFT), `mlx_lm_lora` (DPO), Jac (`jac run`) for eval/plotting scripts, bash runners with dry-run-first + `CONFIRM_FULL_RUN` gates (established pattern in this repo).

## Global Constraints

- Base model: `models/qwen-q4` (already on disk, no CPT-v2 adapter, no SFT). Do not touch `models/qwen-cpt-v2-fused-q4` or `adapters/cpt-v2/` — this arm never loads them.
- Dataset splits MUST be byte-identical copies of `sft_cptv2_probe/dataset/{sft,dpo}/{train,valid}.jsonl` — verify by MD5, never regenerate the 85/15 split, or the holdout comparison is invalid.
- LoRA hyperparams (rank16, layers16, scale2.0, dropout0.05), iters (SFT 8200 / DPO 250), seed 42, `max_seq_length` (SFT 3072 / DPO 512), and `lr_schedule` must stay byte-identical to `sft_cptv2_probe`'s configs — CPT-v2-adapter-presence is the only variable.
- DPO scope: v1 config only (β=0.1, lr=1e-6, 250 iters, segment 20) — no v2 rerun (user decision, DPO collapse was dataset-driven not CPT-driven).
- Same git branch (`sft-cptv2-probe`) — no new branch.
- Every real (non-dry-run) training invocation stays gated behind `CONFIRM_FULL_RUN=1` — never auto-proceed from dry-run to the multi-hour real run.
- Reuse existing generic (env-var-driven, no hardcoded path) scripts as-is: `jacgen/eval_functional.jac`, `jacgen/plot_functional.jac`, `jacgen/plot_dpo.jac`, and `01-sft-dpo/sft_dpo/jacgen/plot_metrics.jac`. Only copy files that have hardcoded paths inside them (`make_dashboard.jac`) or that need CPT-specific line removal (`sft.yaml`, `run_sft.sh`).
- **No task dies unattended.** Both `run_sft.sh` and `run_dpo.sh` are watchdog-supervised: segmented into small chunks, actively polled (log growth, not the child process's self-report), auto-restarted on crash, auto-killed-and-restarted on stall (no log growth for `*_STALL_SECS`). This directly targets a documented failure mode in this repo (`[[project-04-cpt-sft-design]]`: dispatched agents/processes reporting plausible "still running" status while making zero real progress).
- **OOM auto-recovery ladder** (both scripts): on a crash whose log contains an OOM signature, shrink `SEGMENT_ITERS` (halve, floor 5-50 depending on script) for the first 2 recoveries; DPO additionally drops `max_seq_length` 512→384 as a documented last resort on a 3rd OOM. Give up (exit 1, do not spin forever) if OOM persists past the ladder. Both scripts preflight-check for a competing resident-model process (`jac start` / another `mlx_lm` process) before starting — the documented dual-model-load OOM gotcha from the CPT work.
- **DPO done right, not repeating the cptv2 arm's two flagged mistakes**: (1) `val_batches` raised 1→10 (`FULL-RESULTS.md` explicitly flagged n=1 validation reads as noise, unusable for checkpoint selection); (2) snapshots are evaluated **during** training (inline subset functional eval after every 20-iter segment), not only after the full run completes — if functional pass rate collapses relative to this arm's own SFT baseline for 2 consecutive snapshots, the run **stops early** instead of blindly burning the remaining budget on an already-collapsed policy (the cptv2 arm's v2 report found collapse had already happened by iteration 15/250 — this arm can now detect that in-flight). The best-scoring snapshot is tracked and kept (`adapters/dpo-on-sft-best/`) regardless of where training stops.

---

### Task 1: Scaffold directory + copy dataset splits

**Files:**
- Create dirs: `model-experiments/04-cpt-sft/sft_fresh_probe/{configs,dataset/sft,dataset/dpo,adapters,results,jacgen}/`
- Create (copy): `model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/train.jsonl`
- Create (copy): `model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl`
- Create (copy): `model-experiments/04-cpt-sft/sft_fresh_probe/dataset/dpo/train.jsonl`
- Create (copy): `model-experiments/04-cpt-sft/sft_fresh_probe/dataset/dpo/valid.jsonl`

**Interfaces:**
- Produces: the frozen dataset splits every later task's `--data`/`JAC_HOLDOUT` path points at.

- [ ] **Step 1: Create the directory tree**

```bash
mkdir -p model-experiments/04-cpt-sft/sft_fresh_probe/{configs,dataset/sft,dataset/dpo,adapters,results,jacgen}
```

- [ ] **Step 2: Copy the dataset splits verbatim**

```bash
cp model-experiments/04-cpt-sft/sft_cptv2_probe/dataset/sft/train.jsonl model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/train.jsonl
cp model-experiments/04-cpt-sft/sft_cptv2_probe/dataset/sft/valid.jsonl model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl
cp model-experiments/04-cpt-sft/sft_cptv2_probe/dataset/dpo/train.jsonl model-experiments/04-cpt-sft/sft_fresh_probe/dataset/dpo/train.jsonl
cp model-experiments/04-cpt-sft/sft_cptv2_probe/dataset/dpo/valid.jsonl model-experiments/04-cpt-sft/sft_fresh_probe/dataset/dpo/valid.jsonl
```

- [ ] **Step 3: Verify MD5s match the source exactly (this IS the test for this task)**

Run:
```bash
md5 model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/train.jsonl model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl model-experiments/04-cpt-sft/sft_fresh_probe/dataset/dpo/train.jsonl model-experiments/04-cpt-sft/sft_fresh_probe/dataset/dpo/valid.jsonl
```
Expected (macOS `md5`; use `md5sum` on Linux and compare the hash column only):
```
MD5 (.../sft_fresh_probe/dataset/sft/train.jsonl) = a85b9ed2cec29686abb6491c7a7065eb
MD5 (.../sft_fresh_probe/dataset/sft/valid.jsonl) = dd416f9707bb4a861c1424dbe2c8052b
MD5 (.../sft_fresh_probe/dataset/dpo/train.jsonl) = 3d10a1a3e96dabc16a06ae283fc1ae49
MD5 (.../sft_fresh_probe/dataset/dpo/valid.jsonl) = f65c9041f81f16827391b8d740fd2343
```
If any hash differs, delete and re-copy — do not proceed with a mismatched holdout.

- [ ] **Step 4: Commit**

```bash
git add model-experiments/04-cpt-sft/sft_fresh_probe/
git commit -m "feat: scaffold fresh-base probe, freeze byte-identical dataset split from cptv2 probe"
```

---

### Task 2: SFT config + runner script

**Files:**
- Create: `model-experiments/04-cpt-sft/sft_fresh_probe/configs/sft.yaml`
- Create: `model-experiments/04-cpt-sft/sft_fresh_probe/run_sft.sh`

**Interfaces:**
- Consumes: dataset splits from Task 1 (`dataset/sft/{train,valid}.jsonl`).
- Produces: `adapters/sft-on-fresh/adapters.safetensors` (current pointer, always resumable), `adapters/sft-on-fresh/checkpoints/NNNNNNN_adapters.safetensors` (true-global-step-named per-segment snapshots), `results/sft/{train.log,.sft_progress_steps}` — used by Task 5 (eval sweep) and Task 6 (DPO).

- [ ] **Step 1: Write `configs/sft.yaml`** — identical to `sft_cptv2_probe/configs/sft.yaml` with `resume_adapter_file` removed and `data`/`adapter_path` repointed:

```yaml
model: "models/qwen-q4"
train: true
data: "model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft"
fine_tune_type: lora
num_layers: 16
lora_parameters:
  rank: 16
  scale: 2.0
  dropout: 0.05
batch_size: 1
iters: 8200
learning_rate: 2.0e-5
lr_schedule:
  name: cosine_decay
  warmup: 820
  arguments: [2.0e-5, 8200, 1.0e-6]
max_seq_length: 3072
adapter_path: "model-experiments/04-cpt-sft/sft_fresh_probe/adapters/sft-on-fresh"
save_every: 820
steps_per_eval: 500
steps_per_report: 50
val_batches: 8
seed: 42
mask_prompt: true
grad_checkpoint: true
```

- [ ] **Step 2: Write `run_sft.sh`** — watchdog-supervised version. Unlike `sft_cptv2_probe/run_sft.sh` (one long-lived `mlx_lm.lora` process for the whole remaining iter count), this segments training into `save_every`-sized chunks and actively polls each segment for log growth, auto-restarting on crash and auto-killing+restarting on stall, with a bounded OOM shrink ladder. Per your ask ("keep a watchdog so no task dies", "if it OOMs, have a fix"):

```bash
#!/usr/bin/env bash
# SFT-on-fresh-Qwen probe runner, watchdog-supervised. Unlike
# sft_cptv2_probe/run_sft.sh (one long-lived process for the whole remaining
# iter count -- that arm's NaN-crash-and-resume had to be caught and resumed
# by hand), this segments training into save_every-sized chunks and actively
# polls each segment: log growth (not the child's self-report) proves real
# progress, a stalled segment gets killed and restarted, a crashed segment
# gets retried, and a detected OOM shrinks the segment size (then, if that's
# not enough, would need a human -- this script does not silently degrade
# past a documented floor).
set -euo pipefail

if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu env CAFFEINATED=1 "$0" "$@"
fi

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(cd "$SELF_DIR/../../.." && pwd)"   # repo root
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

CFG="model-experiments/04-cpt-sft/sft_fresh_probe/configs/sft.yaml"
ADAPTER="model-experiments/04-cpt-sft/sft_fresh_probe/adapters/sft-on-fresh"
CKPT_DIR="$ADAPTER/checkpoints"    # true-global-step-named snapshots, see the loop comment below for why
RDIR="model-experiments/04-cpt-sft/sft_fresh_probe/results/sft"
TRAIN_LOG="$RDIR/train.log"
DRY_ITERS="${DRY_ITERS:-30}"
EVAL_EVERY="${EVAL_EVERY:-60}"
SEGMENT_ITERS="${SFT_SEGMENT_ITERS:-820}"    # = save_every, so every segment boundary is a real checkpoint
STALL_SECS="${SFT_STALL_SECS:-900}"          # 15min with no new log line => treat as hung

need() { command -v "$1" >/dev/null 2>&1 || { echo "MISSING: $1"; exit 1; }; }
need jac "pip install jaclang"; need mlx_lm.lora "pip install mlx-lm"
for f in model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/train.jsonl \
         model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl "$CFG"; do
  [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }
done

# Preflight: the CPT-v1/v2 dual-model-load scripts OOM'd on this same 48GB
# machine when the studio dev server (its own resident model) was left
# running -- catch that proactively instead of discovering it via an OOM.
if pgrep -f "jac start" >/dev/null 2>&1 || pgrep -f "mlx_lm" >/dev/null 2>&1; then
  echo "!!! another jac/mlx_lm process is already running -- stop it first (a competing"
  echo "!!! resident model will OOM a 48GB machine regardless of this script's own settings)."
  pgrep -fl "jac start|mlx_lm" || true
  exit 1
fi

mkdir -p "$RDIR" "$ADAPTER" "$CKPT_DIR"
done_mark() { touch "$RDIR/.$1.done"; }
is_done() { [ -f "$RDIR/.$1.done" ]; }
ADAPTER_FILE="$ADAPTER/adapters.safetensors"
PROGRESS_FILE="$RDIR/.sft_progress_steps"

# --- dry-run (only on a truly fresh start) ---
if [ ! -f "$ADAPTER_FILE" ] && ! is_done dry && [ "${SKIP_DRY:-0}" != "1" ]; then
  echo ">>> dry-run (${DRY_ITERS} iters) -- bail check"
  mlx_lm.lora --config "$CFG" --iters "$DRY_ITERS" \
    --adapter-path "model-experiments/04-cpt-sft/sft_fresh_probe/adapters/dry" 2>&1 | tail -25
  echo ">>> dry-run complete."
  done_mark dry
fi

# --- gate: only proceed to real training if explicitly confirmed ---
if [ ! -f "$ADAPTER_FILE" ] && is_done dry && [ "${CONFIRM_FULL_RUN:-}" != "1" ]; then
  echo "Dry-run complete. Re-run with CONFIRM_FULL_RUN=1 to start the real multi-hour training run."
  exit 0
fi

TOTAL_ITERS="$(grep -E '^iters:' "$CFG" | grep -oE '[0-9]+' | head -1)"
TOTAL_ITERS="${TOTAL_ITERS:-8200}"
[ -f "$PROGRESS_FILE" ] || echo 0 > "$PROGRESS_FILE"

if is_done train && [ -f "$ADAPTER_FILE" ]; then
  echo ">>> training: already complete"
  exit 0
fi

# Watchdog-supervised segmented loop. Progress is tracked via an EXPLICIT
# counter file, NOT by parsing mlx_lm.lora's own checkpoint filenames.
# VERIFIED against the installed package (.venv/lib/.../mlx_lm/tuner/
# trainer.py:374-375): checkpoints are named "{it:07d}_adapters.safetensors"
# where `it` is the LOCAL per-invocation loop counter, not a global step --
# since every segment here calls `--iters "$SEG" --save-every "$SEG"`
# (forcing exactly one numbered save per segment, always at local iter
# $SEG), a naive "parse the latest numbered file" resume would read the
# SAME filename every segment and never detect real progress. This is the
# same bug CLASS already hit and fixed in this repo's DPO runner (a
# checkpoint-bookkeeping bug that silently re-resumed from the same stale
# checkpoint 5 times) -- fixed here the same way DPO fixed it: an explicit
# progress file, and resuming from the POINTER file (adapters.safetensors,
# confirmed elsewhere in this repo via mtime to always be the real current
# state), never from a numbered file.
consecutive_fails=0
oom_shrinks=0
while true; do
  DONE_STEPS="$(cat "$PROGRESS_FILE")"
  REMAIN=$(( TOTAL_ITERS - DONE_STEPS ))
  if [ "$REMAIN" -le 0 ] && [ -f "$ADAPTER_FILE" ]; then
    done_mark train
    echo "=== SFT training done ($DONE_STEPS/$TOTAL_ITERS). Next: functional eval sweep ==="
    break
  fi
  SEG=$(( REMAIN < SEGMENT_ITERS ? REMAIN : SEGMENT_ITERS ))
  echo ">>> SFT segment: ${SEG} iters (${DONE_STEPS}/${TOTAL_ITERS} done, ${REMAIN} remaining)" | tee -a "$TRAIN_LOG"

  : > "$RDIR/.segment.log"
  RESUME_FLAGS=()
  if [ "$DONE_STEPS" -gt 0 ] && [ -f "$ADAPTER_FILE" ]; then
    RESUME_FLAGS=(--resume-adapter-file "$ADAPTER_FILE")
  fi
  # --save-every "$SEG" (CLI override, not the config's static 820) guarantees
  # exactly one numbered checkpoint per segment even if the OOM ladder below
  # later shrinks SEGMENT_ITERS to something that no longer divides evenly.
  mlx_lm.lora --config "$CFG" --adapter-path "$ADAPTER" --iters "$SEG" --save-every "$SEG" \
    "${RESUME_FLAGS[@]}" >> "$RDIR/.segment.log" 2>&1 &
  SEG_PID=$!

  # Poll: refresh the live graphs every EVAL_EVERY; kill+treat-as-failed if
  # no new log line for STALL_SECS -- this is the "no task dies unattended"
  # mechanism, driven by real file growth, not the child's exit status alone.
  last_growth=$(date +%s); last_size=0
  while kill -0 "$SEG_PID" 2>/dev/null; do
    sleep "$EVAL_EVERY"
    cur_size="$(wc -l < "$RDIR/.segment.log" 2>/dev/null || echo 0)"; now="$(date +%s)"
    if [ "$cur_size" -gt "$last_size" ]; then last_size="$cur_size"; last_growth="$now"; fi
    if [ $(( now - last_growth )) -ge "$STALL_SECS" ]; then
      echo "!!! stalled: no log growth for ${STALL_SECS}s -- killing PID $SEG_PID and treating as a failed segment" | tee -a "$TRAIN_LOG"
      kill -9 "$SEG_PID" 2>/dev/null || true
      break
    fi
    JAC_TRAIN_LOG="$RDIR/.segment.log" JAC_METRICS="/dev/null" JAC_PLOT_DIR="$RDIR" \
      jac run model-experiments/01-sft-dpo/sft_dpo/jacgen/plot_metrics.jac >/dev/null 2>&1 || true
  done
  RC=0; wait "$SEG_PID" 2>/dev/null || RC=$?
  cat "$RDIR/.segment.log" >> "$TRAIN_LOG"

  if [ "$RC" -ne 0 ]; then
    consecutive_fails=$(( consecutive_fails + 1 ))
    echo "!!! segment failed (attempt ${consecutive_fails})" | tee -a "$TRAIN_LOG"
    if grep -qEi "out of memory|OutOfMemory|kIOGPUCommandBuffer|MTL::.*(OOM|Insufficient)" "$RDIR/.segment.log"; then
      echo "!!! OOM signature detected in segment log" | tee -a "$TRAIN_LOG"
      if [ "$oom_shrinks" -lt 2 ]; then
        oom_shrinks=$(( oom_shrinks + 1 ))
        SEGMENT_ITERS=$(( SEGMENT_ITERS / 2 )); [ "$SEGMENT_ITERS" -lt 50 ] && SEGMENT_ITERS=50
        echo "!!! OOM-recovery ${oom_shrinks}/2: shrinking SEGMENT_ITERS to ${SEGMENT_ITERS} and retrying" | tee -a "$TRAIN_LOG"
      else
        echo "!!! OOM persisted through the shrink ladder (2/2 already applied) -- giving up. Check for a competing process (the preflight check above should have caught one that started BEFORE this run; a new one may have started since)." | tee -a "$TRAIN_LOG"
        exit 1
      fi
    fi
    if [ "$consecutive_fails" -ge 5 ]; then
      echo "!!! segment failed 5x in a row at the same ${DONE_STEPS}/${TOTAL_ITERS} point -- giving up." | tee -a "$TRAIN_LOG"
      tail -20 "$TRAIN_LOG"; exit 1
    fi
    continue
  fi
  consecutive_fails=0

  NEW_DONE=$(( DONE_STEPS + SEG ))
  echo "$NEW_DONE" > "$PROGRESS_FILE"
  # Snapshot this segment's checkpoint under its TRUE global step. The
  # numbered file mlx_lm.lora just wrote is named by LOCAL iteration count
  # (see the comment above the loop) -- copy it immediately under an
  # unambiguous global-step name so Task 5's sweep can trust filenames
  # directly, no "+N" offset correction needed afterward.
  LOCAL_NAME="$(printf '%07d' "$SEG")_adapters.safetensors"
  if [ -f "$ADAPTER/$LOCAL_NAME" ]; then
    cp "$ADAPTER/$LOCAL_NAME" "$CKPT_DIR/$(printf '%07d' "$NEW_DONE")_adapters.safetensors"
    echo "  checkpoint snapshot: $CKPT_DIR/$(printf '%07d' "$NEW_DONE")_adapters.safetensors" | tee -a "$TRAIN_LOG"
  fi
done
```

- [ ] **Step 3: Make it executable**

```bash
chmod +x model-experiments/04-cpt-sft/sft_fresh_probe/run_sft.sh
```

- [ ] **Step 4: Commit**

```bash
git add model-experiments/04-cpt-sft/sft_fresh_probe/configs/sft.yaml model-experiments/04-cpt-sft/sft_fresh_probe/run_sft.sh
git commit -m "feat: SFT config + runner for fresh-base probe (no CPT-v2 resume)"
```

---

### Task 3: SFT dry-run sanity check

**Files:** none new — exercises Task 2's script.

**Interfaces:**
- Consumes: `run_sft.sh`, `configs/sft.yaml` from Task 2.
- Produces: `adapters/dry/` (throwaway), `results/sft/.dry.done` marker — gates Task 4.

- [ ] **Step 1: Run the dry-run (30 iters, no CONFIRM_FULL_RUN)**

```bash
cd /Volumes/ExtremePro/JaseciLabs/jac_model_studio
bash model-experiments/04-cpt-sft/sft_fresh_probe/run_sft.sh
```
Expected: first the preflight check passes silently (no competing `jac`/`mlx_lm` process), then prints `>>> dry-run (30 iters) -- bail check`, runs 30 iters of `mlx_lm.lora` with a sane decreasing loss (no `nan`/`inf`), ends with `Dry-run complete. Re-run with CONFIRM_FULL_RUN=1 to start the real multi-hour training run.` and exits 0. Peak memory should stay well under 48GB (SFT alone peaked ~mid-30s GB in the CPT-v2 arm at the same config). If the preflight check fires instead (`!!! another jac/mlx_lm process is already running`), stop and close whatever it found (most likely the studio dev server) before retrying.

- [ ] **Step 2: If it OOMs or produces NaN loss, stop and report before proceeding** — do not adjust hyperparameters silently; `max_seq_length`/`batch_size`/`grad_checkpoint` are already the values proven safe on this exact base model in `sft_cptv2_probe`, so a failure here signals something environmental (e.g. another process holding memory) rather than a config bug. (The real training run's watchdog, Task 4, has its own automated OOM ladder — this dry-run failing is a different, earlier signal worth a human look before committing to the multi-hour run.)

- [ ] **Step 3: No commit** — dry-run artifacts (`adapters/dry/`) are throwaway; leave `.gitignore` (already covers `adapters/`, verify below) to keep them out of git.

```bash
grep -n "adapters/" .gitignore model-experiments/04-cpt-sft/.gitignore 2>/dev/null || echo "CHECK: adapters/ dirs may need a .gitignore entry"
```

---

### Task 4: Full SFT training run

**Files:** none new — exercises Task 2's script to completion.

**Interfaces:**
- Consumes: dry-run-passed state from Task 3.
- Produces: `adapters/sft-on-fresh/adapters.safetensors` (final), `results/sft/train.log`, `results/sft/{train_loss,val_loss,learning_rate,tokens_per_sec,iters_per_sec,trained_tokens,peak_mem}.png` — required by Task 6 (DPO fuse) and Task 5 (eval sweep).

- [ ] **Step 1: Kick off the real run**

```bash
cd /Volumes/ExtremePro/JaseciLabs/jac_model_studio
CONFIRM_FULL_RUN=1 bash model-experiments/04-cpt-sft/sft_fresh_probe/run_sft.sh
```
This is a multi-hour run (the CPT-v2 arm's identical-size run took multiple sessions — 8200 iters, batch_size 1, seq_len 3072). The watchdog loop inside `run_sft.sh` now supervises this itself: it segments into 820-iter chunks (each a real checkpoint), polls for log growth, kills+restarts a stalled segment, retries a crashed one, and shrinks the segment size automatically on a detected OOM — you do not need to babysit it or manually re-run on a crash. Still run it in a dedicated terminal/background session since it's genuinely multi-hour; if the whole process is killed externally (e.g. laptop sleep with `caffeinate` somehow bypassed), re-running the same command resumes cleanly from the latest checkpoint either way.

- [ ] **Step 2: Verify completion**

```bash
tail -30 model-experiments/04-cpt-sft/sft_fresh_probe/results/sft/train.log
ls model-experiments/04-cpt-sft/sft_fresh_probe/adapters/sft-on-fresh/adapters.safetensors
grep -c "OOM-recovery\|stalled:" model-experiments/04-cpt-sft/sft_fresh_probe/results/sft/train.log || true
```
Expected: log ends `=== SFT training done (8200/8200). Next: functional eval sweep ===`; `adapters.safetensors` exists and is non-empty; train loss well below its starting value with no `nan` anywhere in the log. If the OOM/stall grep found hits, that's fine (the watchdog is designed to recover from exactly that) — note in the commit message how many recoveries happened. Checkpoint step numbers in `adapters/sft-on-fresh/checkpoints/*_adapters.safetensors` are already true global steps by construction (each segment is copied there under its real cumulative step, see the script's comment) — no offset correction needed in Task 5 regardless of how many OOM/stall recoveries happened.

```bash
ls model-experiments/04-cpt-sft/sft_fresh_probe/adapters/sft-on-fresh/checkpoints/
cat model-experiments/04-cpt-sft/sft_fresh_probe/results/sft/.sft_progress_steps
```
Expected: `.sft_progress_steps` reads `8200`; the `checkpoints/` dir has one file per completed segment (normally 10, at true steps 820,1640,...,8200 — fewer/irregularly-spaced if the OOM ladder shrank `SEGMENT_ITERS` partway through, which is fine).

- [ ] **Step 3: Commit the log + graphs (not the multi-GB adapter weights, already gitignored)**

```bash
git add model-experiments/04-cpt-sft/sft_fresh_probe/results/sft/train.log model-experiments/04-cpt-sft/sft_fresh_probe/results/sft/images/
git commit -m "feat: fresh-base SFT training run complete, 8200 iters"
```

---

### Task 5: SFT functional eval sweep

**Files:**
- Create: `model-experiments/04-cpt-sft/sft_fresh_probe/eval_sft_sweep.sh`

**Interfaces:**
- Consumes: `adapters/sft-on-fresh/*_adapters.safetensors` (Task 4), reuses `sft_cptv2_probe/jacgen/eval_functional.jac` as-is (env-var driven, no copy needed).
- Produces: `results/sft/metrics_functional.jsonl`, `results/sft/{base,final}.txt`, `results/sft/images/functional_pass_rate.png` — required by Task 9's cross-arm comparison.

- [ ] **Step 1: Write `eval_sft_sweep.sh`** — copy of `sft_cptv2_probe/eval_sft_sweep.sh` with paths repointed and the base row now evaluating **plain `models/qwen-q4` with no adapter** (the true fresh-base number, not CPT-v2's 47.3%). Note: this reuses the *original* `eval_functional.jac` at its `sft_cptv2_probe` path — it's generic and env-var-driven, no need to duplicate it.

```bash
#!/usr/bin/env bash
# Sequential per-checkpoint functional eval for the fresh-base probe. Reuses
# sft_cptv2_probe/jacgen/eval_functional.jac unmodified (env-var driven).
set -euo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

ADAPTER="model-experiments/04-cpt-sft/sft_fresh_probe/adapters/sft-on-fresh"
RDIR="model-experiments/04-cpt-sft/sft_fresh_probe/results/sft"
HOLDOUT="model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl"
METRICS="$RDIR/metrics_functional.jsonl"
SUBSET="${SUBSET:-100}"

mkdir -p "$RDIR/images"
: > "$METRICS"

echo ">>> base (plain Qwen, no CPT, no SFT) -- FULL holdout"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL=models/qwen-q4 JAC_EVAL_ADAPTER="" \
  JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP=0 \
  jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac | tee "$RDIR/base.txt"

TMPADP="model-experiments/04-cpt-sft/sft_fresh_probe/adapters/sft-ckpt-eval"
CKPT_DIR="$ADAPTER/checkpoints"
for CK in "$CKPT_DIR"/*_adapters.safetensors; do
  [ -e "$CK" ] || continue
  # Unlike sft_cptv2_probe's eval_sft_sweep.sh (which needed a manual "+820"
  # true-global-step correction because it swept mlx_lm.lora's own
  # LOCALLY-numbered checkpoint files), run_sft.sh's watchdog loop already
  # copies each segment's checkpoint into checkpoints/ under its TRUE global
  # step name -- so the filename IS the real step, no offset needed here.
  STEP="$(basename "$CK" | grep -oE '^[0-9]+' | sed 's/^0*//')"; STEP="${STEP:-0}"
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
TOTAL_ITERS="$(grep -E '^iters:' model-experiments/04-cpt-sft/sft_fresh_probe/configs/sft.yaml | grep -oE '[0-9]+' | head -1)"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL=models/qwen-q4 JAC_EVAL_ADAPTER="$ADAPTER" \
  JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$TOTAL_ITERS" \
  jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac | tee "$RDIR/final.txt"

echo "=== functional eval sweep done: $METRICS ==="
```

- [ ] **Step 2: Run it**

```bash
chmod +x model-experiments/04-cpt-sft/sft_fresh_probe/eval_sft_sweep.sh
bash model-experiments/04-cpt-sft/sft_fresh_probe/eval_sft_sweep.sh
```
Expected: prints a base `runs %` row (plain Qwen, expect this to land somewhere near or below CPT-v2 base's 47.3% — pure base has never seen Jac at all, so a lower number than CPT-v2-base here would itself be informative), then per-checkpoint subset rows, then a final full-holdout row. Ends `=== functional eval sweep done: .../metrics_functional.jsonl ===`.

- [ ] **Step 3: Generate the pass-rate-over-training graph**

```bash
JAC_FUNC_METRICS=model-experiments/04-cpt-sft/sft_fresh_probe/results/sft/metrics_functional.jsonl \
  JAC_PLOT_DIR=model-experiments/04-cpt-sft/sft_fresh_probe/results/sft/images \
  jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/plot_functional.jac
```
Expected: `wrote .../sft_fresh_probe/results/sft/images/functional_pass_rate.png`.

- [ ] **Step 4: Commit**

```bash
git add model-experiments/04-cpt-sft/sft_fresh_probe/eval_sft_sweep.sh model-experiments/04-cpt-sft/sft_fresh_probe/results/sft/
git commit -m "feat: fresh-base SFT functional eval sweep (base vs final, 855-row holdout)"
```

---

### Task 6: DPO config + runner script (v1 config only)

**Files:**
- Create: `model-experiments/04-cpt-sft/sft_fresh_probe/configs/dpo_lora.yaml`
- Create: `model-experiments/04-cpt-sft/sft_fresh_probe/run_dpo.sh`

**Interfaces:**
- Consumes: `adapters/sft-on-fresh/adapters.safetensors` (Task 4), `results/sft/metrics_functional.jsonl` (Task 5, for the collapse-gate baseline).
- Produces: `models/sft-fresh-fused-q4/` (fused SFT model), `adapters/dpo-on-sft/adapters.safetensors` (last checkpoint), `adapters/dpo-on-sft-best/` (best-scoring snapshot), `results/dpo/{train.log,snapshots/,metrics_functional.jsonl,.best_step,EARLY_STOP.md if triggered}`.

- [ ] **Step 1: Write `configs/dpo_lora.yaml`** — byte-identical to `sft_cptv2_probe/configs/dpo_lora.yaml` (no repo-relative paths inside it, copied for self-containment):

```yaml
lora_parameters:
  rank: 16
  scale: 2.0
  dropout: 0.05
fuse: false   # mlx_lm_lora's CONFIG_DEFAULTS default this to true, which after
              # training unconditionally calls save_pretrained_merged(...,
              # de_quantize=True) -- i.e. dequantizes the full 30.5B-param
              # model to fp16 (~61GB) and writes it into --adapter-path.
              # That's not what we want (we want adapter-only checkpoints,
              # see run_dpo.sh's DPO_ADAPTER/RDIR), and it OOMs unconditionally
              # on a 48GB machine regardless of max_seq_length/batch_size --
              # confirmed empirically in the CPT-v2 arm: actual DPO training
              # itself completed cleanly at max_seq_length 512, and only this
              # post-training auto-fuse step crashed with Metal
              # kIOGPUCommandBufferCallbackErrorOutOfMemory.
```

- [ ] **Step 2: Write `run_dpo.sh`** — v1 hyperparams (β=0.1, lr=1e-6, 250 iters) per your scope decision, but with the operational fixes the cptv2 arm's own `FULL-RESULTS.md` explicitly flagged as missing ("do DPO as it should be, don't repeat the same mistakes"): `val_batches` raised 1→10 (n=1 validation reads during training were documented as noise, unusable for checkpoint selection), per-segment snapshotting, an **inline functional subset-eval after every snapshot** (not only at the end), an **early-stop gate** if pass rate collapses for 2 consecutive snapshots relative to this arm's own SFT baseline, a best-snapshot tracker, an OOM shrink ladder, and the same preflight/stall-watchdog pattern as `run_sft.sh`:

```bash
#!/usr/bin/env bash
# DPO-on-SFT-on-fresh-base runner, v1 hyperparams (beta=0.1, lr=1e-6, 250
# iters) per scope decision -- the cptv2 arm's v2 follow-up (stronger
# regularization) still collapsed to the same functional floor, so this arm
# doesn't repeat that rerun. What IS new here, fixing two mistakes the
# cptv2 arm's own report flagged:
#   - val_batches raised 1->10 (FULL-RESULTS.md: n=1 validation reads during
#     training are noise, not a trustworthy checkpoint-selection signal)
#   - functional subset-eval runs INLINE after every snapshot, with an
#     early-stop gate -- v2's real finding was that collapse had already
#     happened by the earliest checkpoint tested (iter 15/250); this arm
#     detects that in-flight instead of discovering it only after a full
#     blind 250-iter run
# Plus watchdog/OOM-ladder/preflight, matching run_sft.sh.
set -euo pipefail
if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu env CAFFEINATED=1 "$0" "$@"
fi
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

need() { command -v "$1" >/dev/null 2>&1 || { echo "MISSING: $1"; exit 1; }; }
need mlx_lm.fuse "pip install mlx-lm"; need jac "pip install jaclang"
python3 -c "import mlx_lm_lora" || { echo "MISSING: mlx-lm-lora (pip install mlx-lm-lora)"; exit 1; }

if pgrep -f "jac start" >/dev/null 2>&1 || pgrep -f "mlx_lm" >/dev/null 2>&1; then
  echo "!!! another jac/mlx_lm process is already running -- stop it first (a competing resident model will OOM a 48GB machine)."
  pgrep -fl "jac start|mlx_lm" || true
  exit 1
fi

SFT_ADAPTER="model-experiments/04-cpt-sft/sft_fresh_probe/adapters/sft-on-fresh"
SFT_FUSED="models/sft-fresh-fused-q4"
DPO_ADAPTER="model-experiments/04-cpt-sft/sft_fresh_probe/adapters/dpo-on-sft"
BEST_ADAPTER="model-experiments/04-cpt-sft/sft_fresh_probe/adapters/dpo-on-sft-best"
RDIR="model-experiments/04-cpt-sft/sft_fresh_probe/results/dpo"
SNAP_DIR="$RDIR/snapshots"
DPO_ITERS="${DPO_ITERS:-250}"
DPO_LR="${DPO_LR:-1e-6}"
DPO_BETA="${DPO_BETA:-0.1}"
DPO_MAXLEN="${DPO_MAXLEN:-512}"
DPO_VAL_BATCHES="${DPO_VAL_BATCHES:-10}"
SEGMENT_ITERS="${DPO_SEGMENT_ITERS:-20}"
STALL_SECS="${DPO_STALL_SECS:-900}"
EVAL_SUBSET="${DPO_EVAL_SUBSET:-100}"
COLLAPSE_ABS_FLOOR="${DPO_COLLAPSE_ABS_FLOOR:-30}"   # absolute runs_pct fallback threshold
HOLDOUT="model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl"
DPO_METRICS="$RDIR/metrics_functional.jsonl"

mkdir -p "$RDIR" "$SNAP_DIR"
[ -f "$SFT_ADAPTER/adapters.safetensors" ] || { echo "!!! SFT adapter missing, finish Task 4 first"; exit 1; }
[ -f "$DPO_METRICS" ] || : > "$DPO_METRICS"

echo ">>> fuse SFT adapter -> $SFT_FUSED"
if [ -f "$SFT_FUSED/config.json" ]; then echo "  reuse $SFT_FUSED"; else
  mlx_lm.fuse --model models/qwen-q4 --adapter-path "$SFT_ADAPTER" --save-path "$SFT_FUSED"
fi

# SFT baseline for the collapse gate -- read from Task 5's real result, not
# hardcoded, so this arm's gate always compares against ITS OWN SFT number.
SFT_FINAL_PCT="$(python3 -c "
import json
best = 0.0
try:
    with open('model-experiments/04-cpt-sft/sft_fresh_probe/results/sft/metrics_functional.jsonl') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{'): continue
            r = json.loads(line)
            if r.get('category') == '__overall__':
                best = max(best, float(r.get('runs_pct', 0)))
except FileNotFoundError:
    pass
print(best)
")"
COLLAPSE_REL_THRESHOLD="$(python3 -c "print(round(float('${SFT_FINAL_PCT}') * 0.5, 1))")"
echo ">>> SFT baseline: ${SFT_FINAL_PCT}% -- collapse gate fires at 2 consecutive snapshots below max(${COLLAPSE_ABS_FLOOR}%, ${COLLAPSE_REL_THRESHOLD}%)"

DRY_DONE_MARK="$RDIR/.dry.done"
if [ ! -f "$DRY_DONE_MARK" ]; then
  echo ">>> DPO dry-run (8 iters) -- bail check"
  python -m mlx_lm_lora.train --model "$SFT_FUSED" --train --data model-experiments/04-cpt-sft/sft_fresh_probe/dataset/dpo \
    --train-mode dpo --config model-experiments/04-cpt-sft/sft_fresh_probe/configs/dpo_lora.yaml \
    --adapter-path model-experiments/04-cpt-sft/sft_fresh_probe/adapters/dpo-dry \
    --train-type lora --num-layers 16 --grad-checkpoint --batch-size 1 --max-seq-length "$DPO_MAXLEN" \
    --iters 8 --learning-rate "$DPO_LR" --beta "$DPO_BETA" --dpo-cpo-loss-type sigmoid \
    --steps-per-report 2 --steps-per-eval 100000 --val-batches "$DPO_VAL_BATCHES" --save-every 100 2>&1 | tail -25
  touch "$DRY_DONE_MARK"
fi

if [ ! -f "$DPO_ADAPTER/adapters.safetensors" ] && [ "${CONFIRM_FULL_RUN:-}" != "1" ]; then
  echo "Dry-run complete. Re-run with CONFIRM_FULL_RUN=1 to start the real DPO training run."
  exit 0
fi

PROGRESS_FILE="$RDIR/.dpo_progress_steps"
[ -f "$PROGRESS_FILE" ] || echo 0 > "$PROGRESS_FILE"
[ -f "$RDIR/train.log" ] || : > "$RDIR/train.log"
consecutive_fails=0
oom_shrinks=0
low_streak=0
best_pct=-1.0
while true; do
  DONE_STEPS="$(cat "$PROGRESS_FILE")"
  REMAIN=$(( DPO_ITERS - DONE_STEPS ))
  if [ "$REMAIN" -le 0 ]; then echo "=== DPO reached full ${DPO_ITERS} iters ===" | tee -a "$RDIR/train.log"; break; fi
  SEG_ITERS=$(( REMAIN < SEGMENT_ITERS ? REMAIN : SEGMENT_ITERS ))
  echo ">>> DPO segment: ${SEG_ITERS} iters (${DONE_STEPS}/${DPO_ITERS} done)" | tee -a "$RDIR/train.log"

  : > "$RDIR/.segment.log"
  RESUME_FLAGS=()
  if [ "$DONE_STEPS" -gt 0 ] && [ -f "$DPO_ADAPTER/adapters.safetensors" ]; then
    RESUME_FLAGS=(--resume-adapter-file "$DPO_ADAPTER/adapters.safetensors")
  fi
  python -m mlx_lm_lora.train --model "$SFT_FUSED" --train --data model-experiments/04-cpt-sft/sft_fresh_probe/dataset/dpo \
    --train-mode dpo --config model-experiments/04-cpt-sft/sft_fresh_probe/configs/dpo_lora.yaml \
    --adapter-path "$DPO_ADAPTER" --train-type lora --num-layers 16 --grad-checkpoint \
    --batch-size 1 --max-seq-length "$DPO_MAXLEN" --iters "$SEG_ITERS" "${RESUME_FLAGS[@]}" \
    --learning-rate "$DPO_LR" --beta "$DPO_BETA" --dpo-cpo-loss-type sigmoid \
    --steps-per-report 5 --steps-per-eval "$SEG_ITERS" --val-batches "$DPO_VAL_BATCHES" --save-every "$SEG_ITERS" \
    > "$RDIR/.segment.log" 2>&1 &
  SEG_PID=$!

  last_growth=$(date +%s); last_size=0
  while kill -0 "$SEG_PID" 2>/dev/null; do
    sleep 30
    cur_size="$(wc -l < "$RDIR/.segment.log" 2>/dev/null || echo 0)"; now="$(date +%s)"
    if [ "$cur_size" -gt "$last_size" ]; then last_size="$cur_size"; last_growth="$now"; fi
    if [ $(( now - last_growth )) -ge "$STALL_SECS" ]; then
      echo "!!! stalled: no log growth for ${STALL_SECS}s -- killing PID $SEG_PID" | tee -a "$RDIR/train.log"
      kill -9 "$SEG_PID" 2>/dev/null || true
      break
    fi
  done
  RC=0; wait "$SEG_PID" 2>/dev/null || RC=$?
  cat "$RDIR/.segment.log" >> "$RDIR/train.log"

  if [ "$RC" -ne 0 ]; then
    consecutive_fails=$(( consecutive_fails + 1 ))
    if grep -qEi "out of memory|OutOfMemory|kIOGPUCommandBuffer|MTL::.*(OOM|Insufficient)" "$RDIR/.segment.log"; then
      echo "!!! OOM signature detected" | tee -a "$RDIR/train.log"
      if [ "$oom_shrinks" -lt 3 ]; then
        oom_shrinks=$(( oom_shrinks + 1 ))
        if [ "$oom_shrinks" -le 2 ]; then
          SEGMENT_ITERS=$(( SEGMENT_ITERS / 2 )); [ "$SEGMENT_ITERS" -lt 5 ] && SEGMENT_ITERS=5
          echo "!!! OOM-recovery ${oom_shrinks}/3: shrinking DPO_SEGMENT_ITERS to ${SEGMENT_ITERS}" | tee -a "$RDIR/train.log"
        else
          DPO_MAXLEN=384
          echo "!!! OOM-recovery 3/3: dropping DPO_MAXLEN to ${DPO_MAXLEN} (last resort -- flag this deviation in the final report)" | tee -a "$RDIR/train.log"
        fi
      else
        echo "!!! OOM persisted through the full shrink ladder -- giving up." | tee -a "$RDIR/train.log"
        tail -20 "$RDIR/train.log"; exit 1
      fi
    fi
    if [ "$consecutive_fails" -ge 5 ]; then
      echo "!!! segment failed 5x in a row -- giving up." | tee -a "$RDIR/train.log"
      tail -20 "$RDIR/train.log"; exit 1
    fi
    continue
  fi
  consecutive_fails=0

  NEW_DONE=$(( DONE_STEPS + SEG_ITERS ))
  echo "$NEW_DONE" > "$PROGRESS_FILE"
  STEP_TAG="$(printf '%04d' "$NEW_DONE")"
  SNAP="$SNAP_DIR/step_${STEP_TAG}"
  mkdir -p "$SNAP"
  cp "$DPO_ADAPTER/adapters.safetensors" "$SNAP/adapters.safetensors"
  cp "$DPO_ADAPTER/adapter_config.json" "$SNAP/adapter_config.json"
  echo "  snapshot saved: $SNAP" | tee -a "$RDIR/train.log"

  echo ">>> subset functional eval on snapshot step ${NEW_DONE} (n=${EVAL_SUBSET})"
  JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="$SFT_FUSED" JAC_EVAL_ADAPTER="$SNAP" \
    JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_LIMIT="$EVAL_SUBSET" JAC_EVAL_METRICS_OUT="$DPO_METRICS" JAC_EVAL_STEP="$NEW_DONE" \
    jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac 2>&1 | tail -5 | tee -a "$RDIR/train.log"
  STEP_PCT="$(python3 -c "
import json
p = 0.0
try:
    with open('$DPO_METRICS') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{'): continue
            r = json.loads(line)
            if r.get('category') == '__overall__' and r.get('step') == $NEW_DONE:
                p = float(r.get('runs_pct', 0))
except Exception:
    pass
print(p)
")"
  echo "  step ${NEW_DONE} subset runs_pct: ${STEP_PCT}%" | tee -a "$RDIR/train.log"

  IS_BEST="$(python3 -c "print(1 if ${STEP_PCT} > ${best_pct} else 0)")"
  if [ "$IS_BEST" = "1" ]; then
    best_pct="$STEP_PCT"
    rm -rf "$BEST_ADAPTER"; mkdir -p "$BEST_ADAPTER"
    cp "$SNAP/adapters.safetensors" "$BEST_ADAPTER/adapters.safetensors"
    cp "$SNAP/adapter_config.json" "$BEST_ADAPTER/adapter_config.json"
    echo "$NEW_DONE" > "$RDIR/.best_step"
    echo "  new best: step ${NEW_DONE} (${STEP_PCT}%)" | tee -a "$RDIR/train.log"
  fi

  BELOW_GATE="$(python3 -c "print(1 if ${STEP_PCT} < max(${COLLAPSE_ABS_FLOOR}, ${COLLAPSE_REL_THRESHOLD}) else 0)")"
  if [ "$BELOW_GATE" = "1" ]; then low_streak=$(( low_streak + 1 )); else low_streak=0; fi
  if [ "$low_streak" -ge 2 ]; then
    GATE_VAL="$(python3 -c "print(max(${COLLAPSE_ABS_FLOOR}, ${COLLAPSE_REL_THRESHOLD}))")"
    {
      echo "# DPO early-stopped at step ${NEW_DONE}/${DPO_ITERS}"
      echo ""
      echo "Two consecutive snapshots scored below the collapse gate (${GATE_VAL}%, ="
      echo "max(${COLLAPSE_ABS_FLOOR}% absolute floor, 50% of this arm's SFT baseline ${SFT_FINAL_PCT}%))."
      echo "Best snapshot seen: step $(cat "$RDIR/.best_step" 2>/dev/null || echo "$NEW_DONE") at ${best_pct}%."
      echo "Stopped early instead of burning the remaining $(( DPO_ITERS - NEW_DONE )) iters on an already-collapsed policy."
    } > "$RDIR/EARLY_STOP.md"
    touch "$RDIR/.early_stopped"
    echo "!!! EARLY STOP: 2 consecutive snapshots below collapse gate. See $RDIR/EARLY_STOP.md" | tee -a "$RDIR/train.log"
    break
  fi
done
echo "=== DPO training loop done: $RDIR/train.log (best snapshot: step $(cat "$RDIR/.best_step" 2>/dev/null || echo N/A) at ${best_pct}%) ==="
```

- [ ] **Step 3: Make it executable, commit**

```bash
chmod +x model-experiments/04-cpt-sft/sft_fresh_probe/run_dpo.sh
git add model-experiments/04-cpt-sft/sft_fresh_probe/configs/dpo_lora.yaml model-experiments/04-cpt-sft/sft_fresh_probe/run_dpo.sh
git commit -m "feat: DPO v1 config + runner for fresh-base probe"
```

---

### Task 7: DPO dry-run sanity check

**Files:** none new — exercises Task 6's script.

- [ ] **Step 1: Run the dry-run**

```bash
cd /Volumes/ExtremePro/JaseciLabs/jac_model_studio
bash model-experiments/04-cpt-sft/sft_fresh_probe/run_dpo.sh
```
Expected: preflight check passes silently, prints the SFT-baseline/collapse-gate line (`>>> SFT baseline: X% -- collapse gate fires at 2 consecutive snapshots below max(...)`, reading Task 5's real number), fuses `adapters/sft-on-fresh` into `models/sft-fresh-fused-q4` (skips if it already exists), runs 8 dry-run DPO iters with `--val-batches 10` and sane loss (~0.6-0.8 range, no NaN), prints `Dry-run complete. Re-run with CONFIRM_FULL_RUN=1 to start the real DPO training run.`

- [ ] **Step 2: If the fuse step OOMs**, verify `configs/dpo_lora.yaml` really has `fuse: false` (Task 6 Step 1) — this exact failure mode is documented and already guarded against, so a recurrence means the config wasn't written correctly, not a new bug.

- [ ] **Step 3: No commit** (dry-run artifacts only).

---

### Task 8: Full DPO training run

**Files:** none new — exercises Task 6's script to completion.

**Interfaces:**
- Produces: `adapters/dpo-on-sft/adapters.safetensors` (last checkpoint), `adapters/dpo-on-sft-best/` (best snapshot), `results/dpo/{train.log,snapshots/,metrics_functional.jsonl}` — required by Task 9.

- [ ] **Step 1: Kick off the real run**

```bash
cd /Volumes/ExtremePro/JaseciLabs/jac_model_studio
CONFIRM_FULL_RUN=1 bash model-experiments/04-cpt-sft/sft_fresh_probe/run_dpo.sh
```
Segmented (20 iters/process), watchdog-supervised (stall detection + OOM shrink ladder, same mechanism as Task 4), and resumable — if a segment crashes, just re-run the same command; it picks up from `results/dpo/.dpo_progress_steps`. Every segment now also runs an inline subset functional eval and may trigger the early-stop gate — **do not treat an early exit as a failure** without checking for `results/dpo/EARLY_STOP.md` first; that's the collapse gate working as designed, not a crash.

- [ ] **Step 2: Verify completion**

```bash
cat model-experiments/04-cpt-sft/sft_fresh_probe/results/dpo/.dpo_progress_steps
ls model-experiments/04-cpt-sft/sft_fresh_probe/adapters/dpo-on-sft/adapters.safetensors
ls model-experiments/04-cpt-sft/sft_fresh_probe/adapters/dpo-on-sft-best/adapters.safetensors
cat model-experiments/04-cpt-sft/sft_fresh_probe/results/dpo/EARLY_STOP.md 2>/dev/null || echo "no early stop -- ran to full budget"
```
Expected: either progress file reads `250` with no `EARLY_STOP.md` (ran to completion), or progress reads less than `250` WITH an `EARLY_STOP.md` explaining why (collapse gate fired) — both are valid, informative outcomes for this experiment, not failures. Either way, both `adapters/dpo-on-sft/` (last) and `adapters/dpo-on-sft-best/` (best-scoring snapshot) should exist.

- [ ] **Step 3: Generate DPO training graphs**

```bash
JAC_DPO_LOG=model-experiments/04-cpt-sft/sft_fresh_probe/results/dpo/train.log \
  JAC_PLOT_DIR=model-experiments/04-cpt-sft/sft_fresh_probe/results/dpo/images \
  jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/plot_dpo.jac
```
Expected: `wrote .../dpo_loss.png`, `.../dpo_accuracy.png`, `.../dpo_margin.png`, `.../dpo_rewards.png`.

- [ ] **Step 4: Commit** (not the multi-GB snapshot adapter weights — gitignored; the metrics/logs are small and worth keeping)

```bash
git add model-experiments/04-cpt-sft/sft_fresh_probe/results/dpo/train.log model-experiments/04-cpt-sft/sft_fresh_probe/results/dpo/images/ model-experiments/04-cpt-sft/sft_fresh_probe/results/dpo/.dpo_progress_steps model-experiments/04-cpt-sft/sft_fresh_probe/results/dpo/.best_step model-experiments/04-cpt-sft/sft_fresh_probe/results/dpo/metrics_functional.jsonl
git add model-experiments/04-cpt-sft/sft_fresh_probe/results/dpo/EARLY_STOP.md 2>/dev/null || true
git commit -m "feat: fresh-base DPO v1 training run complete (see .dpo_progress_steps for actual iters -- may be <250 if the collapse gate fired)"
```

---

### Task 9: DPO final functional eval

**Files:**
- Create: `model-experiments/04-cpt-sft/sft_fresh_probe/eval_dpo.sh`

**Interfaces:**
- Consumes: `adapters/dpo-on-sft/adapters.safetensors` (last) and `adapters/dpo-on-sft-best/` (best) from Task 8, `models/sft-fresh-fused-q4` (Task 6).
- Produces: `results/dpo/{final_last.txt,final_best.txt}`, appends full-holdout rows to `results/dpo/metrics_functional.jsonl` (which already has Task 8's per-snapshot subset rows) — required by Task 10.

- [ ] **Step 1: Write `eval_dpo.sh`** — evaluates BOTH the last checkpoint and the best-scoring snapshot (if different) on the FULL 855-row holdout, since Task 8's snapshotting exists precisely because the last checkpoint isn't necessarily the most informative one (the cptv2 arm's own v2 finding: collapse can already be worst-case by the earliest checkpoint). Step numbers are offset to sort after all subset-eval rows so `make_dashboard.jac`'s "max step = final" logic always picks the full-holdout "last" row as the canonical DPO result, never a partial subset row:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

SFT_FUSED="models/sft-fresh-fused-q4"
DPO_ADAPTER="model-experiments/04-cpt-sft/sft_fresh_probe/adapters/dpo-on-sft"
BEST_ADAPTER="model-experiments/04-cpt-sft/sft_fresh_probe/adapters/dpo-on-sft-best"
HOLDOUT="model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl"
RDIR="model-experiments/04-cpt-sft/sft_fresh_probe/results/dpo"
METRICS="$RDIR/metrics_functional.jsonl"   # keep -- already has Task 8's per-snapshot subset rows

FINAL_STEP="$(cat "$RDIR/.dpo_progress_steps" 2>/dev/null || echo 250)"
BEST_STEP="$(cat "$RDIR/.best_step" 2>/dev/null || echo "$FINAL_STEP")"
LAST_EVAL_STEP=$(( FINAL_STEP + 1000000 ))   # sorts after every subset row -> "the" dpo_final for the dashboard
BEST_EVAL_STEP=$(( BEST_STEP + 500000 ))     # sorts after subset rows but before LAST_EVAL_STEP -- diagnostic only

echo ">>> FULL holdout eval: last checkpoint (step $FINAL_STEP)"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="$SFT_FUSED" JAC_EVAL_ADAPTER="$DPO_ADAPTER" \
  JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$LAST_EVAL_STEP" \
  jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac | tee "$RDIR/final_last.txt"

if [ -f "$BEST_ADAPTER/adapters.safetensors" ] && [ "$BEST_STEP" != "$FINAL_STEP" ]; then
  echo ">>> FULL holdout eval: best snapshot (step $BEST_STEP)"
  JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="$SFT_FUSED" JAC_EVAL_ADAPTER="$BEST_ADAPTER" \
    JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$BEST_EVAL_STEP" \
    jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac | tee "$RDIR/final_best.txt"
else
  echo ">>> best snapshot == last checkpoint (step $BEST_STEP), skipping duplicate full eval"
fi
```

- [ ] **Step 2: Run it**

```bash
chmod +x model-experiments/04-cpt-sft/sft_fresh_probe/eval_dpo.sh
bash model-experiments/04-cpt-sft/sft_fresh_probe/eval_dpo.sh
```
Expected: one or two overall `runs %` rows on the full 855-row holdout. No specific pass-rate is "expected" — the CPT-v2 arm's own DPO-v1 result was a collapse to 12.3%, so this arm's number is a genuine open question (does the collapse also hit a non-CPT base, or was CPT-v2 specifically implicated). If Task 8's collapse gate fired, `final_last.txt`'s step already IS the best-seen point (the gate stops as soon as 2 consecutive snapshots underperform, so "last" and "best" are close in that scenario, not equal).

- [ ] **Step 3: Commit**

```bash
git add model-experiments/04-cpt-sft/sft_fresh_probe/eval_dpo.sh model-experiments/04-cpt-sft/sft_fresh_probe/results/dpo/final_last.txt model-experiments/04-cpt-sft/sft_fresh_probe/results/dpo/final_best.txt model-experiments/04-cpt-sft/sft_fresh_probe/results/dpo/metrics_functional.jsonl
git commit -m "feat: fresh-base DPO-final functional eval (last + best snapshot) on full SFT holdout"
```

---

### Task 10: Per-probe 3-way dashboard (base / +SFT / +SFT+DPO)

**Files:**
- Create: `model-experiments/04-cpt-sft/sft_fresh_probe/jacgen/make_dashboard.jac`

**Interfaces:**
- Consumes: `results/sft/metrics_functional.jsonl` (Task 5), `results/dpo/metrics_functional.jsonl` (Task 9).
- Produces: `results/images/comparison_overall.png`, `results/images/comparison_by_category.png`.

- [ ] **Step 1: Write the dashboard script** — copy of `sft_cptv2_probe/jacgen/make_dashboard.jac` (hardcoded paths, so this one file must be copied not reused) with the three `glob` paths repointed and labels changed from "CPT-v2 base" to "base (no CPT)":

```jac
"""Final 3-way comparison: base (no CPT) vs +SFT vs +SFT+DPO, fresh probe.

Mirrors sft_cptv2_probe/jacgen/make_dashboard.jac exactly, repointed at this
probe's own metrics files.
"""

import json;
import os;
import matplotlib;
import from matplotlib { pyplot }
import from pathlib { Path }

glob SFT_METRICS: str = "model-experiments/04-cpt-sft/sft_fresh_probe/results/sft/metrics_functional.jsonl";
glob DPO_METRICS: str = "model-experiments/04-cpt-sft/sft_fresh_probe/results/dpo/metrics_functional.jsonl";
glob OUT_DIR: str = "model-experiments/04-cpt-sft/sft_fresh_probe/results/images";


def load_rows(path: str) -> list[dict] {
    rows: list[dict] = [];
    p = Path(path);
    if not p.exists() { return rows; }
    for line in p.read_text().split("\n") {
        if line.strip().startswith("{") { rows.append(dict(json.loads(line))); }
    }
    return rows;
}


def category_breakdown(rows: list[dict], step: int, shape: dict) -> dict {
    out: dict = {};
    for r in rows {
        if r["category"] == "__overall__" { continue; }
        if r["step"] != step { continue; }
        key: tuple = (r["category"], r["gate_class"]);
        if key in shape and shape[key] != r["total"] { continue; }
        cat: str = str(r["category"]);
        if cat not in out { out[cat] = {"total": 0, "runs": 0}; }
        out[cat]["total"] += int(r["total"]);
        out[cat]["runs"] += int(r["runs"]);
    }
    return out;
}


def base_shape(rows: list[dict], step: int) -> dict {
    shape: dict = {};
    for r in rows {
        if r["category"] == "__overall__" { continue; }
        if r["step"] != step { continue; }
        shape[(r["category"], r["gate_class"])] = int(r["total"]);
    }
    return shape;
}


with entry {
    matplotlib.use("Agg");
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True);

    sft_rows: list[dict] = load_rows(SFT_METRICS);
    dpo_rows: list[dict] = load_rows(DPO_METRICS);

    base_overall: dict = {};
    sft_final_overall: dict = {};
    max_sft_step: int = 0;
    for r in sft_rows {
        if r["category"] != "__overall__" { continue; }
        if r["step"] == 0 { base_overall = r; }
        if r["step"] >= max_sft_step { max_sft_step = r["step"]; sft_final_overall = r; }
    }
    dpo_final_overall: dict = {};
    max_dpo_step: int = -1;
    for r in dpo_rows {
        if r["category"] != "__overall__" { continue; }
        if r["step"] >= max_dpo_step { max_dpo_step = r["step"]; dpo_final_overall = r; }
    }

    labels: list[str] = ["base (no CPT)", "+SFT"];
    values: list[float] = [float(base_overall.get("runs_pct", 0)), float(sft_final_overall.get("runs_pct", 0))];
    if len(dpo_final_overall) > 0 {
        labels.append("+SFT+DPO");
        values.append(float(dpo_final_overall.get("runs_pct", 0)));
    } else {
        print("NOTE: no DPO metrics_functional.jsonl found -- run Task 9 first.");
    }

    pyplot.figure(figsize=(7, 5));  # jac:ignore[E1051]
    pyplot.bar(labels, values, color=["#888888", "#4C72B0", "#C44E52"][:len(labels)]);  # jac:ignore[E1051]
    pyplot.title("Functional pass rate (runs %) -- fresh-base probe");  # jac:ignore[E1051]
    pyplot.ylabel("runs %");  # jac:ignore[E1051]
    pyplot.ylim(0, 100);  # jac:ignore[E1051]
    for (i, v) in enumerate(values) { pyplot.text(i, v + 1, str(round(v, 1)) + "%", ha="center"); }  # jac:ignore[E1051]
    pyplot.grid(True, axis="y");  # jac:ignore[E1051]
    pyplot.savefig(OUT_DIR + "/comparison_overall.png");  # jac:ignore[E1051]
    pyplot.close();  # jac:ignore[E1051]
    print("wrote " + OUT_DIR + "/comparison_overall.png");
    print(json.dumps({"base": base_overall, "sft_final": sft_final_overall, "dpo_final": dpo_final_overall}, indent=2));

    shape: dict = base_shape(sft_rows, 0);
    base_cats: dict = category_breakdown(sft_rows, 0, shape);
    sft_cats: dict = category_breakdown(sft_rows, max_sft_step, shape);
    dpo_cats: dict = category_breakdown(dpo_rows, max_dpo_step, shape);

    cat_names: list[str] = sorted(list(base_cats.keys()));
    base_pcts: list[float] = [];
    sft_pcts: list[float] = [];
    dpo_pcts: list[float] = [];
    for cat in cat_names {
        b: dict = base_cats.get(cat, {"total": 0, "runs": 0});
        s: dict = sft_cats.get(cat, {"total": 0, "runs": 0});
        d: dict = dpo_cats.get(cat, {"total": 0, "runs": 0});
        base_pcts.append(round(100.0 * b["runs"] / b["total"], 1) if b["total"] > 0 else 0.0);
        sft_pcts.append(round(100.0 * s["runs"] / s["total"], 1) if s["total"] > 0 else 0.0);
        dpo_pcts.append(round(100.0 * d["runs"] / d["total"], 1) if d["total"] > 0 else 0.0);
    }

    x: list[int] = list(range(len(cat_names)));
    width: float = 0.25;
    pyplot.figure(figsize=(11, 6));  # jac:ignore[E1051]
    pyplot.bar([i - width for i in x], base_pcts, width, label="base (no CPT)", color="#888888");  # jac:ignore[E1051]
    pyplot.bar(x, sft_pcts, width, label="+SFT", color="#4C72B0");  # jac:ignore[E1051]
    pyplot.bar([i + width for i in x], dpo_pcts, width, label="+SFT+DPO", color="#C44E52");  # jac:ignore[E1051]
    pyplot.xticks(x, cat_names, rotation=20);  # jac:ignore[E1051]
    pyplot.ylabel("runs %");  # jac:ignore[E1051]
    pyplot.ylim(0, 100);  # jac:ignore[E1051]
    pyplot.title("Functional pass rate by category -- fresh-base probe");  # jac:ignore[E1051]
    pyplot.legend();  # jac:ignore[E1051]
    pyplot.grid(True, axis="y");  # jac:ignore[E1051]
    pyplot.tight_layout();  # jac:ignore[E1051]
    pyplot.savefig(OUT_DIR + "/comparison_by_category.png");  # jac:ignore[E1051]
    pyplot.close();  # jac:ignore[E1051]
    print("wrote " + OUT_DIR + "/comparison_by_category.png");
    print(json.dumps({"categories": cat_names, "base": base_pcts, "sft": sft_pcts, "dpo": dpo_pcts}, indent=2));
}
```

- [ ] **Step 2: Run it**

```bash
jac run model-experiments/04-cpt-sft/sft_fresh_probe/jacgen/make_dashboard.jac
```
Expected: two `wrote ...png` lines, plus JSON dumps of the underlying numbers to stdout.

- [ ] **Step 3: Commit**

```bash
git add model-experiments/04-cpt-sft/sft_fresh_probe/jacgen/make_dashboard.jac model-experiments/04-cpt-sft/sft_fresh_probe/results/images/
git commit -m "feat: fresh-base probe 3-way dashboard (base/+SFT/+SFT+DPO)"
```

---

### Task 11: Cross-arm "did CPT help" comparison + final report

**Files:**
- Create: `model-experiments/04-cpt-sft/jacgen/compare_arms.jac`
- Create: `model-experiments/04-cpt-sft/docs/reports/2026-07-cpt-vs-fresh-comparison.md`

**Interfaces:**
- Consumes: `sft_cptv2_probe/results/{sft,dpo}/metrics_functional.jsonl` (already committed), `sft_fresh_probe/results/{sft,dpo}/metrics_functional.jsonl` (Tasks 5 + 9).
- Produces: `model-experiments/04-cpt-sft/results/images/cpt_vs_fresh_overall.png`, `cpt_vs_fresh_by_category.png`, the final markdown readout — this is the actual deliverable answering "did CPT help."

- [ ] **Step 1: Write `compare_arms.jac`** — 6-bar chart (base/+SFT/+SFT+DPO × 2 arms) plus per-category grouped comparison, computing CPT deltas explicitly:

```jac
"""Cross-arm comparison: does CPT-v2 help once SFT/DPO trains on top of it?
Reads both probes' metrics_functional.jsonl (both already share the identical
1428-row holdout -- verified in sft_fresh_probe's Task 1 via MD5) and computes
the CPT delta at each stage.
"""

import json;
import os;
import matplotlib;
import from matplotlib { pyplot }
import from pathlib { Path }

glob CPTV2_SFT_METRICS: str = "model-experiments/04-cpt-sft/sft_cptv2_probe/results/sft/metrics_functional.jsonl";
glob CPTV2_DPO_METRICS: str = "model-experiments/04-cpt-sft/sft_cptv2_probe/results/dpo/metrics_functional.jsonl";
glob FRESH_SFT_METRICS: str = "model-experiments/04-cpt-sft/sft_fresh_probe/results/sft/metrics_functional.jsonl";
glob FRESH_DPO_METRICS: str = "model-experiments/04-cpt-sft/sft_fresh_probe/results/dpo/metrics_functional.jsonl";
glob OUT_DIR: str = "model-experiments/04-cpt-sft/results/images";


def load_rows(path: str) -> list[dict] {
    rows: list[dict] = [];
    p = Path(path);
    if not p.exists() { return rows; }
    for line in p.read_text().split("\n") {
        if line.strip().startswith("{") { rows.append(dict(json.loads(line))); }
    }
    return rows;
}


def overall_stages(rows: list[dict]) -> dict {
    base: dict = {}; final: dict = {}; max_step: int = 0;
    for r in rows {
        if r["category"] != "__overall__" { continue; }
        if r["step"] == 0 { base = r; }
        if r["step"] >= max_step { max_step = r["step"]; final = r; }
    }
    return {"base": base, "final": final};
}


def dpo_overall(rows: list[dict]) -> dict {
    out: dict = {}; max_step: int = -1;
    for r in rows {
        if r["category"] != "__overall__" { continue; }
        if r["step"] >= max_step { max_step = r["step"]; out = r; }
    }
    return out;
}


with entry {
    matplotlib.use("Agg");
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True);

    cptv2_sft: dict = overall_stages(load_rows(CPTV2_SFT_METRICS));
    fresh_sft: dict = overall_stages(load_rows(FRESH_SFT_METRICS));
    cptv2_dpo: dict = dpo_overall(load_rows(CPTV2_DPO_METRICS));
    fresh_dpo: dict = dpo_overall(load_rows(FRESH_DPO_METRICS));

    cptv2_base_pct: float = float(cptv2_sft["base"].get("runs_pct", 0));
    fresh_base_pct: float = float(fresh_sft["base"].get("runs_pct", 0));
    cptv2_sft_pct: float = float(cptv2_sft["final"].get("runs_pct", 0));
    fresh_sft_pct: float = float(fresh_sft["final"].get("runs_pct", 0));
    cptv2_dpo_pct: float = float(cptv2_dpo.get("runs_pct", 0));
    fresh_dpo_pct: float = float(fresh_dpo.get("runs_pct", 0));

    labels: list[str] = ["base", "+SFT", "+SFT+DPO"];
    fresh_vals: list[float] = [fresh_base_pct, fresh_sft_pct, fresh_dpo_pct];
    cptv2_vals: list[float] = [cptv2_base_pct, cptv2_sft_pct, cptv2_dpo_pct];

    x: list[int] = list(range(len(labels)));
    width: float = 0.35;
    pyplot.figure(figsize=(9, 6));  # jac:ignore[E1051]
    pyplot.bar([i - width / 2 for i in x], fresh_vals, width, label="fresh base (no CPT)", color="#4C72B0");  # jac:ignore[E1051]
    pyplot.bar([i + width / 2 for i in x], cptv2_vals, width, label="CPT-v2 base", color="#C44E52");  # jac:ignore[E1051]
    for (i, v) in enumerate(fresh_vals) { pyplot.text(i - width / 2, v + 1, str(round(v, 1)) + "%", ha="center"); }  # jac:ignore[E1051]
    for (i, v) in enumerate(cptv2_vals) { pyplot.text(i + width / 2, v + 1, str(round(v, 1)) + "%", ha="center"); }  # jac:ignore[E1051]
    pyplot.xticks(x, labels);  # jac:ignore[E1051]
    pyplot.ylabel("runs %");  # jac:ignore[E1051]
    pyplot.ylim(0, 100);  # jac:ignore[E1051]
    pyplot.title("Did CPT-v2 help? fresh-base vs CPT-v2-base, same SFT/DPO recipe+data");  # jac:ignore[E1051]
    pyplot.legend();  # jac:ignore[E1051]
    pyplot.grid(True, axis="y");  # jac:ignore[E1051]
    pyplot.savefig(OUT_DIR + "/cpt_vs_fresh_overall.png");  # jac:ignore[E1051]
    pyplot.close();  # jac:ignore[E1051]
    print("wrote " + OUT_DIR + "/cpt_vs_fresh_overall.png");

    deltas: dict = {
        "base_delta_cptv2_minus_fresh": round(cptv2_base_pct - fresh_base_pct, 2),
        "sft_delta_cptv2_minus_fresh": round(cptv2_sft_pct - fresh_sft_pct, 2),
        "dpo_delta_cptv2_minus_fresh": round(cptv2_dpo_pct - fresh_dpo_pct, 2),
        "fresh_base_pct": fresh_base_pct, "cptv2_base_pct": cptv2_base_pct,
        "fresh_sft_pct": fresh_sft_pct, "cptv2_sft_pct": cptv2_sft_pct,
        "fresh_dpo_pct": fresh_dpo_pct, "cptv2_dpo_pct": cptv2_dpo_pct,
    };
    print(json.dumps(deltas, indent=2));
    Path(OUT_DIR + "/../cpt_vs_fresh_deltas.json").write_text(json.dumps(deltas, indent=2));
}
```

- [ ] **Step 2: Run it**

```bash
mkdir -p model-experiments/04-cpt-sft/results/images
jac run model-experiments/04-cpt-sft/jacgen/compare_arms.jac
```
Expected: `wrote .../cpt_vs_fresh_overall.png` plus a JSON delta printout. Read the delta values before writing the report — they determine the report's verdict, don't pre-write a conclusion.

- [ ] **Step 3: Write the final report** at `model-experiments/04-cpt-sft/docs/reports/2026-07-cpt-vs-fresh-comparison.md`, structured like `sft_cptv2_probe/results/FULL-RESULTS.md`: headline table (both arms × 3 stages), the `cpt_vs_fresh_overall.png` image, the computed deltas from Step 2's JSON, and an explicit verdict sentence — "CPT-v2 provides a real X-point functional advantage once SFT lands on top of it" or "CPT-v2's contribution is statistically indistinguishable from a fresh base once SFT lands on top of it," whichever the numbers actually show. Do not write this file until Step 2's real numbers exist.

- [ ] **Step 4: Commit**

```bash
git add model-experiments/04-cpt-sft/jacgen/compare_arms.jac model-experiments/04-cpt-sft/results/images/ model-experiments/04-cpt-sft/results/cpt_vs_fresh_deltas.json model-experiments/04-cpt-sft/docs/reports/2026-07-cpt-vs-fresh-comparison.md
git commit -m "docs: cross-arm comparison -- does CPT-v2 help once SFT/DPO trains on top of it"
```

---

## Self-Review Notes

- **Spec coverage:** every design element (scaffold, SFT config w/ resume_adapter_file removed, dry-run gate, full run, eval sweep w/ true base row, DPO v1 only, DPO dry-run/full-run/eval, per-probe dashboard, cross-arm comparison) has a task. Branch decision (same branch) required no task — already true.
- **Real bug caught pre-dispatch and fixed:** the first draft of `run_sft.sh`'s segmented watchdog loop inferred resume progress by parsing the latest `*_adapters.safetensors` filename — verified against the installed `mlx_lm` source (`tuner/trainer.py:374-375`) that this filename uses a LOCAL per-invocation iteration counter, so every fixed-size segment would have saved an identically-named file (`0000820_adapters.safetensors`), silently breaking both progress-tracking and Task 5's interim-checkpoint sweep. Same bug class this repo's DPO runner already hit and fixed (a checkpoint-bookkeeping bug that silently re-resumed from a stale checkpoint 5 times, per `.superpowers/sdd/progress.md`). Fixed by adopting DPO's proven pattern: an explicit `.sft_progress_steps` counter file, resuming only from the pointer file (`adapters.safetensors`), and copying each segment's checkpoint into `checkpoints/` under its true global step immediately after the segment completes.
- **Watchdog / OOM / DPO-rigor additions (this revision):** both `run_sft.sh` and `run_dpo.sh` are segmented + actively polled (log growth, not child self-report) + auto-restarted on crash + auto-killed-and-restarted on stall + OOM-shrink-laddered before giving up — directly answers "keep a watchdog so no task dies" and "if it OOMs, have a fix." DPO additionally fixes the two concrete mistakes `sft_cptv2_probe/results/FULL-RESULTS.md` flagged in its own "honest methodological limitation" section: `val_batches` 1→10, and snapshot-time functional eval (not post-hoc only) with an early-stop-on-collapse gate + best-snapshot tracking — answers "do DPO as it should be, don't repeat the same mistakes." None of this touches the hyperparameters themselves (β/lr/iters stay v1, per your scope decision) — it's instrumentation and failure-recovery, not a third DPO config.
- **Placeholder scan:** no TBD/TODO; the one open unknown (whether resumed-crash step correction is needed in Task 5) is explicitly flagged as a conditional check, not a silent gap. The early-stop scenario (Task 8/9) is explicitly documented as a valid, non-failure outcome, not glossed over.
- **Type/path consistency:** `sft-on-fresh` / `dpo-on-sft` / `dpo-on-sft-best` / `sft-fresh-fused-q4` names used consistently across Tasks 2, 4-10; `metrics_functional.jsonl` schema (`category`, `step`, `runs_pct`, `total`, `runs`, `gate_class`) matches `eval_functional.jac`'s actual output field names (verified by reading `sft_cptv2_probe/jacgen/{plot_functional,make_dashboard}.jac`, which already consume that exact schema). The `LAST_EVAL_STEP`/`BEST_EVAL_STEP` offset scheme in Task 9 (`+1000000`/`+500000`) is a deliberate, documented device so `make_dashboard.jac`'s existing unmodified "max step wins" logic still picks the right row — not a magic number.
- **Scope:** DPO-v2 hyperparameter rerun and multi-seed noise-floor repeats explicitly excluded per your answers — not silently dropped, called out in Global Constraints. The new watchdog/snapshot/early-stop machinery is operational rigor, not a scope expansion into a new DPO config.
