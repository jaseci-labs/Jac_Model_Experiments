#!/usr/bin/env bash
# CPT-on-Spectrum launcher -- cpt-spectrum-plan.md §3, gate 4.
#
# Gates 0-3 (pytest, --verify-layers, 2-iter dry leg, end-to-end CF-check) were
# already run manually and passed clean on the real 30B model before this
# script exists: 65+5 tests green, VERIFY: PASS, checkpoint post-condition OK
# (256 keys, correct blocks, no all-zero tensors, peak mem 30.058 GB -- safe
# margin below this machine's GPU wired-memory ceiling), CF-check 16/16 with
# cf_check_0000002.json landing in results/cpt-v2-spectrum/json/ and CPT-v2's
# own results/cpt-v2/json/ verified untouched (still 20 files).
#
# This script's job is just: preflight (no competing jac/mlx_lm process, which
# would OOM a 48GB machine regardless of anything else), the CONFIRM_FULL_RUN=1
# gate (this is a ~26-30h commitment, never launch it by accident), then hand
# off to run_epoch_loop_spectrum.py, which is resumable by construction --
# killing and relaunching this script loses at most the in-flight leg.
set -euo pipefail
if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu env CAFFEINATED=1 "$0" "$@"
fi

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(cd "$SELF_DIR/../../../.." && pwd)"   # repo root
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

if pgrep -f "jac start" >/dev/null 2>&1 || pgrep -f "mlx_lm" >/dev/null 2>&1; then
  echo "!!! another jac/mlx_lm process is already running -- stop it first (a competing resident model will OOM a 48GB machine)."
  pgrep -fl "jac start|mlx_lm" || true
  exit 1
fi

ADAPTER_DIR="model-experiments/03-cpt-only/adapters/cpt-v2-spectrum"
if [ -f "$ADAPTER_DIR/adapters.safetensors" ] && [ "${CONFIRM_FULL_RUN:-}" != "1" ]; then
  echo "Gates 0-3 already passed. Re-run with CONFIRM_FULL_RUN=1 to start/resume the real ~26-30h CPT run."
  echo "(An adapter already exists at $ADAPTER_DIR -- this would RESUME from its checkpoints, not restart.)"
  exit 0
fi
if [ ! -f "$ADAPTER_DIR/adapters.safetensors" ] && [ "${CONFIRM_FULL_RUN:-}" != "1" ]; then
  echo "Gates 0-3 already passed (manually, before this script existed). Re-run with CONFIRM_FULL_RUN=1 to start the real ~26-30h CPT run."
  exit 0
fi

echo ">>> starting/resuming the CPT-on-Spectrum epoch loop (12 legs, floor 6 / ceiling 12)"
exec "$PWD/.venv/bin/python3" "$SELF_DIR/run_epoch_loop_spectrum.py"
