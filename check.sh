#!/usr/bin/env bash
# Validate the jac modules — two gates:
#  1. TYPE  — full `jac check` (all 20 modules clean). Dynamic Python-interop
#     (json/subprocess/regex/matplotlib return Any) is handled by str()/list()/
#     dict()/int() casts at the boundary; a few genuinely-untypeable stdlib calls
#     (inspect.signature, signal, matplotlib stubs) carry `# jac:ignore[...]`.
#  2. BEHAVIOR — `jac run`. Every dataset example is re-validated by running it
#     (seed_conversion.jac reports N/N). The real gate.
set -euo pipefail
[ -d "$PWD/.venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"   # subprocess `jac` resolves
JAC="${JAC:-.venv/bin/jac}"
[ -x "$JAC" ] || JAC="jac"   # fall back to PATH (e.g. after `source .venv/bin/activate`)

echo "=== type check (jac check) ==="
"$JAC" check srccurrent/jacgen/*.jac

echo "=== behavior (jac run: re-validate the conversion dataset) ==="
"$JAC" run srccurrent/jacgen/seed_conversion.jac 2>/dev/null | tail -1

echo "OK"
