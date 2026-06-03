#!/usr/bin/env bash
# One-time environment setup for the probe (run in your target Python env).
# Anaconda was intentionally removed; install the toolchain wherever you
# standardize. NOT run by the data-prep work.
set -euo pipefail

pip install jaclang mlx-lm matplotlib

echo "--- verify ---"
jac --version || echo "jac missing"
mlx_lm.lora --help >/dev/null 2>&1 && echo "mlx-lm ok" || echo "mlx-lm missing"
python -c "import matplotlib; print('matplotlib', matplotlib.__version__)" 2>/dev/null || echo "matplotlib missing (ASCII dashboard still works)"
echo "then: ./run_probe.sh <hf-model-id> <short-name>"
