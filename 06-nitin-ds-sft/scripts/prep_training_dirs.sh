#!/usr/bin/env bash
# ============================================================================
# prep_training_dirs.sh -- turn the single-file release JSONLs that
# scripts/collect.jac emits into the train/valid DIRECTORY shape mlx_lm_lora
# actually reads.
#
# WHY THIS EXISTS. Both arms' configs point `data:` at a DIRECTORY, not a file:
#
#     stock_probe/configs/sft.yaml            data: ".../06-nitin-ds-sft/dataset/sft"
#     spectrum_probe/spectrum/configs/sft_spectrum.yaml   (same)
#     run_dpo_nofuse.sh / run_dpo_spectrum.sh  --data ".../06-nitin-ds-sft/dataset/dpo"
#
# mlx_lm / mlx_lm_lora resolve that directory to `train.jsonl` + `valid.jsonl`
# inside it (workflow.md Stage 2, "mlx_lm_lora is pointed at the *directory*").
# collect.jac writes single release files instead:
#
#     dataset/sft_train.jsonl   ->  dataset/sft/{train,valid}.jsonl
#     dataset/dpo_train.jsonl   ->  dataset/dpo/{train,valid}.jsonl
#
# SPLIT RATIO: 85/15, seeded (seed 42) -- matching 04-cpt-sft's fresh arm
# exactly, which is the recipe this phase replicates:
#     sft/train.jsonl 8100 : sft/valid.jsonl 1428  = 85.0 / 15.0
#     dpo/train.jsonl  654 : dpo/valid.jsonl  115  = 85.0 / 15.0
# The valid splits are mlx_lm_lora's TRAINING-TIME validation splits, not
# scored eval sets (spec.md §4.2 / CONTEXT_BRIEF.md §4.5 item 4). Every
# reported number comes from the eval scripts against holdout (a) or (b).
#
# The Nitin holdout (dataset/nitin_holdout.jsonl) is carved upstream in Stage 0
# and is NOT touched here -- this script only splits what collect.jac released.
#
# USAGE (run from anywhere; resolves the repo root itself):
#     model-experiments/06-nitin-ds-sft/scripts/prep_training_dirs.sh
#     FORCE=1 ... # overwrite existing non-empty splits
#     SPLIT_SEED=42 TRAIN_FRAC=0.85 ...  # both overridable
# ----------------------------------------------------------------------------
set -euo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"   # repo root
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

EXP="model-experiments/06-nitin-ds-sft"
SPLIT_SEED="${SPLIT_SEED:-42}"
TRAIN_FRAC="${TRAIN_FRAC:-0.85}"
FORCE="${FORCE:-0}"

split_one() {   # $1 = release file, $2 = output dir, $3 = required-field list (comma sep)
  local release="$1" outdir="$2" required="$3"
  if [ ! -f "$release" ]; then
    echo "SKIP: $release does not exist yet (Stage 1/2 collect has not run)."
    return 0
  fi
  if [ -s "$outdir/train.jsonl" ] && [ "$FORCE" != "1" ]; then
    echo "SKIP: $outdir/train.jsonl already exists and is non-empty."
    echo "      Re-run with FORCE=1 to overwrite. (Refusing to clobber a split a"
    echo "      training run may already be reading -- this project's data is single-copy.)"
    return 0
  fi
  mkdir -p "$outdir"
  python3 - "$release" "$outdir" "$SPLIT_SEED" "$TRAIN_FRAC" "$required" <<'PY'
import json, random, sys
from pathlib import Path

release, outdir, seed, frac, required = sys.argv[1:6]
seed, frac = int(seed), float(frac)
required = [f for f in required.split(",") if f]
outdir = Path(outdir)

rows, bad = [], 0
with open(release) as f:
    for ln, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            print(f"  !! line {ln}: not valid JSON -- dropped")
            continue
        missing = [k for k in required if not r.get(k)]
        if missing:
            bad += 1
            print(f"  !! line {ln}: missing/empty {missing} -- dropped")
            continue
        rows.append(line)

if not rows:
    print(f"  !! {release}: zero usable rows -- refusing to write an empty split")
    sys.exit(1)

# Deterministic: sort first so the shuffle does not inherit file order, then
# shuffle with an explicit seed. Same input file => same split, every time.
rows.sort()
random.Random(seed).shuffle(rows)
n_train = int(round(len(rows) * frac))
n_train = max(1, min(n_train, len(rows) - 1))   # never produce an empty side
train, valid = rows[:n_train], rows[n_train:]

(outdir / "train.jsonl").write_text("\n".join(train) + "\n")
(outdir / "valid.jsonl").write_text("\n".join(valid) + "\n")
pct = 100.0 * len(train) / len(rows)
print(f"  {release}: {len(rows)} usable rows ({bad} dropped)")
print(f"  -> {outdir}/train.jsonl  {len(train)} ({pct:.1f}%)")
print(f"  -> {outdir}/valid.jsonl  {len(valid)} ({100.0 - pct:.1f}%)  [seed={seed}]")
PY
}

echo ">>> SFT release -> $EXP/dataset/sft/"
split_one "$EXP/dataset/sft_train.jsonl" "$EXP/dataset/sft" "messages"

echo ">>> DPO release -> $EXP/dataset/dpo/"
split_one "$EXP/dataset/dpo_train.jsonl" "$EXP/dataset/dpo" "prompt,chosen,rejected"

echo
echo "=== current training-dir state ==="
for f in "$EXP/dataset/sft/train.jsonl" "$EXP/dataset/sft/valid.jsonl" \
         "$EXP/dataset/dpo/train.jsonl" "$EXP/dataset/dpo/valid.jsonl"; do
  if [ -f "$f" ]; then printf '  %8s  %s\n' "$(wc -l < "$f" | tr -d ' ')" "$f"
  else printf '  %8s  %s\n' "MISSING" "$f"; fi
done
