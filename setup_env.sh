#!/usr/bin/env bash
# One-time environment setup (no anaconda). Creates/updates the repo-root .venv
# on your system python3, installs the training/eval toolchain, and sanity-checks
# the pipeline scripts ($EXP/{scaffold,train,eval}/). Run from anywhere.
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"   # repo root (this script lives there)

BASE="models/qwen-q4"
EXP="experiments/08-nitin-new2-ds"    # the experiment whose scripts are sanity-checked

[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install --upgrade pip >/dev/null
# pinned: spectrum_lora_layers.jac hash-checks the mlx-lm source (0.31.3)
.venv/bin/pip install jaclang==0.16.1 mlx==0.31.2 mlx-lm==0.31.3 mlx-lm-lora==2.1.0 matplotlib numpy

echo "--- verify ---"
.venv/bin/jac --version >/dev/null 2>&1 && echo "jac: ok" || echo "jac: MISSING"
.venv/bin/python -c "import mlx_lm, mlx_lm_lora" 2>/dev/null && echo "mlx_lm + mlx_lm_lora: ok" || echo "mlx_lm/mlx_lm_lora: MISSING"
[ -f "$BASE/config.json" ] && echo "base model $BASE: ok" || echo "base model $BASE: MISSING (Qwen3-Coder-30B-A3B 4-bit)"
fail=0
for f in "$EXP"/scaffold/*.sh "$EXP"/train/*.sh "$EXP"/train/*/*.sh "$EXP"/eval/*.sh; do bash -n "$f" || { echo "bash -n FAILED: $f"; fail=1; }; done
for f in "$EXP"/eval/eval_functional.jac "$EXP"/scaffold/{nan_guard.jac,prep_training_dirs.sh} \
         "$EXP"/train/spectrum/{spectrum_lora_layers,adapter_config_fix}.jac "$EXP"/train/spectrum/spectrum_layers.json; do
  [ -f "$f" ] || { echo "MISSING: $f"; fail=1; }
done
.venv/bin/jac check "$EXP"/scaffold/nan_guard.jac "$EXP"/train/spectrum/*.jac >/dev/null 2>&1 \
  && echo "jac check (core scripts): ok" || { echo "jac check: FAILED"; fail=1; }
[ "$fail" = 0 ] && echo "pipeline sanity: ok" || echo "pipeline sanity: FAILED"

echo
echo "next (see PLAYBOOK.md; each script targets the experiment it sits in --"
echo "      new experiment: cp -r $EXP/{scaffold,train,eval} experiments/<new>/):"
echo "  source .venv/bin/activate"
echo "  $EXP/scaffold/prep_training_dirs.sh   # split + NaN guard"
echo "  $EXP/train/run_sft_spectrum.sh        # spectrum SFT"
echo "  $EXP/eval/eval_sft_spectrum.sh        # functional eval"
echo
echo "JMS (chat + train + data + evals; sibling repo ../Jac_Model_Studio):"
echo "  ../Jac_Model_Studio/start.sh          # API :8001 + UI :8000"
