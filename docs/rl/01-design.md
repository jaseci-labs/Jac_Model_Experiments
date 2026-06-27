# Jac RL — Design

This is the design of record. It covers task formats, the holdout pool, the training ladder, the reward, and the eval. See [00-overview.md](00-overview.md) for goals and [02-results.md](02-results.md) for measured numbers.

## 1. Task design

### Type A — hole-fill (lead track)
A complete, deterministic `jac run`-able `.jac` file with exactly one unit's body wrapped in sentinels:

    # >>>HOLE id="..." instruction="..."
    <body statements>
    # <<<HOLE

The grader replaces the body with the literal `__HOLE__`, splices the model completion in, and runs the file in an isolated cwd. **Pass = byte-exact stdout** vs the ground-truth stdout (captured by running the driver as-authored).

**Mandatory `unwrap_unit`:** models emit the whole enclosing unit (`can walk_day with Day entry { <body> }`), not the bare body. Before splicing, unwrap a single enclosing unit to its inner block. Skipping this nests `can {...can {...}...}` and nothing runs — this faked the entire first weekend run.

Every Type A task ships a gold `refbodies/<id>.txt` sidecar (the real body) for the reward sim term.

### Type B — whole-file (later track)
Regenerate an entire small `.jac` from its docstring/spec. Graded by AST-equivalence + run + stdout (stdout alone may legitimately vary). Built only after the hole-fill ladder is mapped; AST grader design is deferred to that track. This is the bridge toward the "generate new codebases" goal.

### Task extraction
Build **all** extractable hole-fill tasks from `this_is_jac/`. **51 built** so far (`rl/build_tasks.jac` → `dataset/rl/tasks.jsonl`); ~10–15 more deterministic seams are minable (see [project notes]). Reserve the holdout pool (below) before assigning any task to training.

## 2. Holdout
- **Fixed pool, ~15 tasks, never trained.** The same pool is evaluated at every rung, so generalization numbers are comparable rung-to-rung.
- **Rung 1 is the one exception** — the "memorize" rung evaluates on the single trained task itself, targeting 100% to prove the plumbing.

## 3. Training ladder
- **Rungs (train-set size): 1, 3, 5, 10, 20, all-remaining.** Run the full ladder always; the sweet spot is read off the curve (where fixed-holdout pass plateaus or peaks), not chosen by a stopping rule.
- **Per rung, per model (×2 = jac-qwen3coder + fresh qwen3coder), 3 conditions:**
  1. **SFT** LoRA on the rung's tasks → eval.
  2. **SFT checkpoint + GRPO** → eval.
  3. **raw-base GRPO** (no rung SFT) → eval — a control that should reproduce the σ=0 cold-start trap and confirm GRPO can't bootstrap from 0%.
- = 6 train/eval cells per rung + a base-eval. **Headline measurement = GRPO's marginal lift over SFT on the fixed holdout.**

> Honest prior: the weekend run already showed SFT+GRPO ≈ SFT (no lift) and raw-base GRPO stalls at σ=0. We re-run it on the ladder anyway, staged so that *if* GRPO ever helps it will show as holdout lift at a mid rung where SFT has raised base-pass above 0.

## 4. Reward (GRPO only)
Dense v2, every term defined so within-group variance is never zero:

    0.25·compiles + 0.25·runs + 0.25·output + 0.10·idiom + 0.15·body_sim

- `output`: exact stdout = 1.0, else `0.5 · difflib ratio(out, expected)`.
- `idiom`: graph/object-spatial ops weighted (`visit`/`-->`/`spawn`/`disengage`=3, `report`/`here`/`walker`/`node`/`edge`=2, plain `can`/`has`/`obj`=1), normalized `min(n/8, 1)`, gated behind `runs`.
- `body_sim`: `difflib ratio(body, gold refbody)`, computed for **every** completion including non-compiling ones — the only term not gated behind `runs`. This is what kills the σ=0 zero-advantage trap.

## 5. Eval
- **Headline:** exact-stdout pass% (Type A) / AST+run pass% (Type B).
- **Diagnostics, every run:** graded score (the reward sum), near-pass (osim ≥ 0.9), avg-osim (continuous output similarity over all tasks).
- **Two reads per rung:** memorize-score (re-eval the rung's train tasks) and generalize-score (the fixed holdout pool).

## 6. Open items
1. ✅ Fresh `qwen3coder` HF repo = **`Qwen/Qwen3-Coder-30B-A3B-Instruct`** (128 experts / 8 active; confirmed against the local `models/qwen-q4` config). Already downloaded + quantized.
2. Type B AST-equivalence grader — design when the whole-file track starts.
