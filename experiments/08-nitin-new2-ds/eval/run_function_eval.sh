#!/usr/bin/env bash
# jac-data-gen function eval v1 (Nitin's sealed test cases, evals/function/v1):
#   reference  grade the reference solutions (ceiling; flags broken tasks)
#   base       generate + grade the untrained base model
#   adapter    generate + grade this experiment's adapter
# Generation: gen_function_eval.jac on the PUBLIC prompts (greedy, 1 sample/task).
# Grading: jac-data-gen's own scripts/eval/eval_jac.py on the PRIVATE file
# (jac check, then jac test with the hidden test blocks), unmodified.
#
#   experiments/08-nitin-new2-ds/eval/run_function_eval.sh [reference] [base] [adapter]   (default: all)
#   SPLIT=test|dev (default test)   JAC_EVAL_LIMIT=8 (smoke run, separate out dir)
#   JDG=path/to/jac-data-gen (default: sibling of this repo)   WORKERS=8 (grader threads)
set -euo pipefail
EXP_ABS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$EXP_ABS/../.."   # repo root
EXP="experiments/$(basename "$EXP_ABS")"

JDG="${JDG:-../jac-data-gen}"
SPLIT="${SPLIT:-test}"
LIMIT="${JAC_EVAL_LIMIT:-0}"
PUB="$JDG/evals/function/v1/public/$SPLIT.jsonl"
PRIV="$JDG/evals/function/v1/private/$SPLIT.jsonl"
OUT="$EXP/results/function_v1_$SPLIT"
[ "$LIMIT" != 0 ] && OUT="$EXP/tmp/function_v1_${SPLIT}_smoke$LIMIT"
JAC="$PWD/.venv/bin/jac"   # generation + primary grading toolchain (jaclang 0.16.1, what 08 trained against)
# Cross-check toolchain: the eval's own (jac-data-gen records jac 0.36.1), a standalone binary kept
# out of PATH so ~/.local/bin/jac is untouched. Its `jac test` costs ~4 s per test block here
# (0.16.1: well under 1 s), too slow for all tasks, so it grades a fixed every-20th-task subset
# into <stage>/jac-0.36.1-subset/. CROSS_JAC= (empty) skips it.
CROSS_JAC="${CROSS_JAC-$PWD/.jac-grader/jac-0.36.1}"
STAGES="${*:-reference base adapter}"

for f in "$PUB" "$PRIV" "$JDG/scripts/eval/eval_jac.py"; do
  [ -f "$f" ] || { echo "missing $f (clone github.com/chess10kp/jac-data-gen next to this repo)"; exit 1; }
done
[ -z "$CROSS_JAC" ] || [ -x "$CROSS_JAC" ] || { echo "missing $CROSS_JAC; get it with (or set CROSS_JAC= to skip):"
  echo "  mkdir -p .jac-grader && curl -fsSL -o .jac-grader/jac-0.36.1 https://github.com/jaseci-labs/jac/releases/download/v0.36.1/jac-0.36.1-macos-aarch64 && chmod +x .jac-grader/jac-0.36.1"; exit 1; }
ver() { [ -n "$1" ] && "$1" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1; }
mkdir -p "$OUT"
cat > "$OUT/provenance.json" <<EOF
{"jac_data_gen_commit": "$(git -C "$JDG" rev-parse HEAD)", "split": "$SPLIT", "limit": $LIMIT,
 "gen_and_grader_jac_version": "$(ver "$JAC")", "cross_check_jac_version": "$(ver "$CROSS_JAC")",
 "model": "models/qwen-q4", "adapter": "$EXP/adapter", "date": "$(date -u +%FT%TZ)"}
EOF

# Shared GPU lock with Jac Model Studio agents (mkdir is atomic): one model on the 48 GB box at a
# time. Two loaded models push MLX weights into swap, and an MLX process killed in that state can
# hang in exit (ps state ?E) holding its memory until a reboot.
LOCK=/tmp/jac-gpu.lock; OWNER="run_function_eval.sh pid $$"
gpu_lock() {
  until mkdir "$LOCK" 2>/dev/null; do echo "waiting for $LOCK ($(cat "$LOCK/owner" 2>/dev/null))"; sleep 30; done
  echo "$OWNER" > "$LOCK/owner"; trap gpu_unlock EXIT
}
gpu_unlock() { [ "$(cat "$LOCK/owner" 2>/dev/null)" = "$OWNER" ] && rm -rf "$LOCK"; return 0; }

run_grader() {  # $1 = jac bin, $2 = samples, $3 = out dir, $4 = per-stage timeout (s)
  mkdir -p "$3"
  .venv/bin/python "$JDG/scripts/eval/eval_jac.py" --problems "$PRIV" --samples "$2" \
    --out-dir "$3" --jac-bin "$1" --k 1 --workers "${WORKERS:-8}" --timeout "${4:-120}" > "$3/grade.log" 2>&1 \
    || echo "grader exit $? (see $3/grade.log)"
  .venv/bin/python - "$3/summary.json" "$(ver "$1")" <<'PY'
import json, sys
o = json.load(open(sys.argv[1]))["overall"]
print("  jac", sys.argv[2], {k: o.get(k) for k in ("samples", "task_success_rate", "check_rate", "behavior_test_rate", "status_counts", "complete")})
PY
}
grade() {  # $1 = stage dir
  run_grader "$JAC" "$1/samples.jsonl" "$1"
  [ -n "$CROSS_JAC" ] || return 0
  .venv/bin/python - "$PRIV" "$1/samples.jsonl" > "$1/samples_subset.jsonl" <<'PY'
import json, sys   # every 20th task of the private file, selected by id so all stages share the subset
ids = {json.loads(l)["id"] for i, l in enumerate(open(sys.argv[1])) if i % 20 == 0}
sys.stdout.writelines(l for l in open(sys.argv[2]) if json.loads(l)["problem_id"] in ids)
PY
  PYTEST_XDIST_AUTO_NUM_WORKERS=2 WORKERS=4 run_grader "$CROSS_JAC" "$1/samples_subset.jsonl" "$1/jac-0.36.1-subset" 900
}

for stage in $STAGES; do
  D="$OUT/$stage"; mkdir -p "$D"
  echo "=== $stage -> $D"
  case "$stage" in
    reference)
      # completion tasks: the reference body (grader prepends the prefix); translation: the full idiomatic source
      .venv/bin/python - "$PRIV" "$LIMIT" > "$D/samples.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
if int(sys.argv[2]): rows = rows[:int(sys.argv[2])]
for r in rows:
    s = {"problem_id": r["id"], "sample_id": 0, "task": r["task"]}
    if r["task"] == "completion": s["completion"] = r["reference_completion"]
    else: s["jac"] = r["idiomatic_jac"]
    print(json.dumps(s))
PY
      ;;
    base|adapter)
      AD=""; [ "$stage" = adapter ] && AD="$EXP/adapter"
      gpu_lock
      JAC_EVAL_MODEL=models/qwen-q4 JAC_EVAL_ADAPTER="$AD" JAC_FN_PROBLEMS="$PUB" \
        JAC_FN_SAMPLES_OUT="$D/samples.jsonl" JAC_EVAL_LIMIT="$LIMIT" PYTHONUNBUFFERED=1 \
        "$JAC" run "$EXP/eval/gen_function_eval.jac" 2>&1 | tee -a "$D/gen.log" | grep -vE '^\s*$' | tail -3
      gpu_unlock
      ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
  grade "$D"
done
echo "done -> $OUT"
