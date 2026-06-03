#!/usr/bin/env bash
# Validate the jac modules.
#
# Two gates, by design:
#  1. SYNTAX  — `jac check -p` (parse-only). The full jac TYPE checker is
#     over-strict on dynamic Python-interop (json/subprocess/matplotlib all
#     return Any, which it refuses to assign to typed vars) and is NOT this
#     project's gate. It also flags code that runs correctly (see eval_probe's
#     jac-check-vs-jac-run lesson). So syntax is the static gate.
#  2. BEHAVIOR — `jac run`. Every dataset example is re-validated by running it
#     (seed_conversion.jac reports N/N). That is the real gate.
set -euo pipefail
[ -d "$PWD/.venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"   # subprocess `jac` resolves
JAC="${JAC:-.venv/bin/jac}"
[ -x "$JAC" ] || JAC="jac"   # fall back to PATH (e.g. after `source .venv/bin/activate`)

echo "=== syntax (jac check -p) ==="
"$JAC" check -p srccurrent/jacgen/*.jac

echo "=== behavior (jac run: re-validate the conversion dataset) ==="
"$JAC" run srccurrent/jacgen/seed_conversion.jac 2>/dev/null | tail -1

echo "OK"
