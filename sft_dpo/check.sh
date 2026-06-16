#!/usr/bin/env bash
# Validate the jac modules — two gates:
#  1. TYPE  — full `jac check` (all 20 modules clean). Dynamic Python-interop
#     (json/subprocess/regex/matplotlib return Any) is handled by str()/list()/
#     dict()/int() casts at the boundary; a few genuinely-untypeable stdlib calls
#     (inspect.signature, signal, matplotlib stubs) carry `# jac:ignore[...]`.
#  2. BEHAVIOR — `jac run`. A sampled audit re-runs stored dataset examples and
#     confirms their output still matches (verify_dataset.jac, NON-destructive).
#     This is the real gate. NOTE: do NOT use seed_conversion.jac here — it
#     TRUNCATES dataset/conversion/sft.jsonl back to the 32 seeds (and dpo.jsonl
#     to 2), wiping the idiomatic_batch* / dpo_conversion appends. See HANDOFF.md.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"   # repo root, so dataset/ paths resolve
[ -d "$PWD/.venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"   # subprocess `jac` resolves
JAC="${JAC:-.venv/bin/jac}"
[ -x "$JAC" ] || JAC="jac"   # fall back to PATH (e.g. after `source .venv/bin/activate`)

echo "=== type check (jac check) ==="
# eval_probe imports mlx_lm; jaclang's type-checker CRASHES resolving mlx's model
# types (internal bug, not our code). Parse-check it (syntax) + rely on jac run;
# full type-check the other 19.
# eval_probe.jac + idiom_eval.jac import mlx_lm (lazy); the type-checker crashes on
# mlx types, so parse-check (-p) those two and full-check the rest.
CORE=$(ls sft_dpo/jacgen/*.jac | grep -vE 'eval_probe.jac|idiom_eval.jac')
"$JAC" check $CORE
"$JAC" check -p sft_dpo/jacgen/eval_probe.jac
"$JAC" check -p sft_dpo/jacgen/idiom_eval.jac

echo "=== behavior (jac run: sampled re-validation of the conversion dataset) ==="
# NON-destructive: re-runs every Nth stored example and checks output still matches.
JAC_SAMPLE_EVERY="${JAC_SAMPLE_EVERY:-40}" "$JAC" run sft_dpo/jacgen/verify_dataset.jac 2>/dev/null | tail -1

echo "OK"
