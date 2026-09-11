#!/usr/bin/env bash
# One-time environment setup (no anaconda). Creates/updates the repo-root .venv
# on your system python3, installs the training/eval toolchain, and sanity-checks
# 08-nitin-new2-ds. Run from anywhere.
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"   # repo root (this script lives there)

EXP="08-nitin-new2-ds"
BASE="models/qwen-q4"

[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install --upgrade pip >/dev/null
# pinned: spectrum_lora_layers.jac hash-checks the mlx-lm source (0.31.3)
.venv/bin/pip install jaclang==0.16.1 mlx==0.31.2 mlx-lm==0.31.3 mlx-lm-lora==2.1.0 matplotlib numpy

echo "--- verify ---"
.venv/bin/jac --version >/dev/null 2>&1 && echo "jac: ok" || echo "jac: MISSING"
.venv/bin/python -c "import mlx_lm, mlx_lm_lora" 2>/dev/null && echo "mlx_lm + mlx_lm_lora: ok" || echo "mlx_lm/mlx_lm_lora: MISSING"
[ -f "$BASE/config.json" ] && echo "base model $BASE: ok" || echo "base model $BASE: MISSING (Qwen3-Coder-30B-A3B 4-bit)"
fail=0
for f in "$EXP"/scripts/*.sh "$EXP"/*_probe/*.sh; do bash -n "$f" || { echo "bash -n FAILED: $f"; fail=1; }; done
for f in "$EXP"/scripts/{eval_functional,nan_guard,prep_training_dirs}.* \
         "$EXP"/spectrum_probe/spectrum/{spectrum_lora_layers,adapter_config_fix}.jac; do
  [ -f "$f" ] || { echo "MISSING: $f"; fail=1; }
done
.venv/bin/jac check "$EXP"/scripts/nan_guard.jac "$EXP"/spectrum_probe/spectrum/*.jac >/dev/null 2>&1 \
  && echo "jac check (08 core scripts): ok" || { echo "jac check: FAILED"; fail=1; }
[ "$fail" = 0 ] && echo "08 sanity: ok" || echo "08 sanity: FAILED"

echo
echo "next (see $EXP/docs/workflow.md):"
echo "  source .venv/bin/activate"
echo "  $EXP/scripts/prep_training_dirs.sh              # split + NaN guard"
echo "  $EXP/spectrum_probe/run_sft_spectrum.sh         # spectrum SFT"
echo "  $EXP/spectrum_probe/eval_sft_spectrum.sh        # functional eval"
echo
echo "JMS (chat + train + data + evals):"
echo "  ./jms/start.sh                              # API :8001 + UI :8000"
