#!/usr/bin/env bash
# One-time environment setup (no anaconda). Creates a project venv on your
# system python3 and installs the toolchain. NOT run by the data-prep work.
set -euo pipefail

python3 -m venv .venv
.venv/bin/pip install --upgrade pip >/dev/null
.venv/bin/pip install jaclang mlx-lm matplotlib

echo "--- verify ---"
.venv/bin/jac --version >/dev/null 2>&1 && echo "jac: ok" || echo "jac: MISSING"
.venv/bin/jac check -p srccurrent/jacgen/*.jac >/dev/null 2>&1 \
  && echo "syntax check: ok" || echo "syntax check: FAILED"
echo
echo "next:"
echo "  source .venv/bin/activate     # puts jac + mlx_lm on PATH"
echo "  ./check.sh                     # syntax sweep + behavioral note"
echo "  ./run_probe.sh <hf-model> <name>"
