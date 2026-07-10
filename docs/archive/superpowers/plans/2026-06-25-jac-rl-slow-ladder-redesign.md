# Jac RL — Slow Ladder Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reset the Jac RL effort to a slow, incremental "ladder" experiment — rewrite all `docs/rl/` documentation around it and delete the abandoned weekend RL artifacts (~200GB).

**Architecture:** Pure documentation + file removal. No training code, no harness code in this plan. Produce 4 new docs in `docs/rl/` + move the weekend results archive in, then execute a confirmed destructive cleanup of RL adapters/models/results/dataset/old-docs.

**Tech Stack:** Markdown docs. `git mv` / `rm -rf` for file ops. No build, no tests (no code produced).

## Global Constraints

These are project-wide truths every task must respect; copy verbatim into any doc that states them:

- **Dataset source:** `this_is_jac/` only (77 `.jac` files). No external task sources.
- **Two ladder models:** `jac-qwen3coder` (already SFT+DPO on jac, fused at `models/jac-qwen3coder-q4`) vs **fresh** `qwen3coder` (HF base, assume `Qwen/Qwen3-Coder-30B-A3B-Instruct`, confirm at build time). No Qwen3.6 — it is inference-only on 48GB and is being removed.
- **Pass bar:** exact stdout match (hole-fill) / AST-equivalence + run (whole-file) is the headline number. graded-score, near-pass (osim≥0.9), avg-osim reported as diagnostics beside it.
- **Two carried-forward build requirements** (hard-won in the weekend run, must survive the harness rewrite): (1) `unwrap_unit` splice fix — models emit the whole `can ... { body }` unit, so unwrap one enclosing unit to its inner block before splicing into `__HOLE__`; (2) dense **body-sim** reward term (`difflib ratio(body, gold refbody)`, computed for EVERY completion incl. non-compiling ones) to defeat the σ=0 cold-start zero-advantage trap.
- **Ladder:** fixed ~15-task holdout pool (never trained, same set every rung). Rungs (train size): **1, 3, 5, 10, 20, all-remaining**. Run the full ladder always; read the sweet spot off the curve. Per rung × per model × 3 conditions: (a) SFT, (b) SFT+GRPO, (c) raw-base GRPO control.
- **GRPO reward (dense v2):** `0.25 compiles + 0.25 runs + 0.25 output + 0.10 idiom + 0.15 body-sim`.

---

### Task 1: Move the weekend results archive into docs/rl

**Files:**
- Move: `RL_WEEKEND_RESULTS.md` → `docs/rl/RL_WEEKEND_RESULTS.md`

**Interfaces:**
- Produces: `docs/rl/RL_WEEKEND_RESULTS.md` (the prior-art / lessons archive the new docs link to).

- [ ] **Step 1: git mv the file**

```bash
git mv RL_WEEKEND_RESULTS.md docs/rl/RL_WEEKEND_RESULTS.md
```

- [ ] **Step 2: Verify it moved and old path is gone**

Run: `ls docs/rl/RL_WEEKEND_RESULTS.md && test ! -e RL_WEEKEND_RESULTS.md && echo OK`
Expected: prints the path then `OK`.

- [ ] **Step 3: Fix internal cross-references inside the moved doc**

The doc's header references `docs/rl/runlog.md`, `docs/rl/RL_RESULTS.md`, `docs/documentation/00-rl-phase.md` — all being deleted in Task 6/7. Edit that line to point at the new docs instead: replace the "Full detail in ..." sentence with `Full detail and the redesigned workflow live in docs/rl/00-overview.md and docs/rl/01-design.md.`

---

### Task 2: Write docs/rl/00-overview.md (north star)

**Files:**
- Create: `docs/rl/00-overview.md`

**Interfaces:**
- Produces: the goals/principles doc that `README.md` and `01-design.md` link to.

- [ ] **Step 1: Write the file** with this content:

```markdown
# Jac RL — Overview & North Star

## Goal
Train models that can (1) **fill in parts of Jac code** and (2) eventually **generate new Jac codebases**. We will not get best-in-class results yet — the point of this phase is to get the *workflow* right: a slow, patient, measurable ladder we trust.

## Why a reset
The weekend GRPO run ([RL_WEEKEND_RESULTS.md](RL_WEEKEND_RESULTS.md)) proved the harness, reward, and eval are correct on real 30B — but also that LoRA-GRPO does not move a 30B's greedy decoding at this scale, and the only lever that moved the needle was supervised fine-tuning. So we restart from the ground up, slowly, and measure everything.

## Dataset
`this_is_jac/` only — 77 `.jac` files, deliberately diverse (graph walkers, libs, the littlex social graph, raylib, guestbook). Diversity is why it is a good seed corpus. No external task sources.

## The two models (same ladder, head to head)
- `jac-qwen3coder` — already SFT+DPO on jac (`models/jac-qwen3coder-q4`).
- fresh `qwen3coder` — untrained HF base.

The comparison answers: **does prior jac knowledge help the memorize→generalize curve, and by how much?**

> Qwen3.6 is removed entirely — dense 27B and 35B-A3B both OOM training on 48GB (inference-only). Only the fewer-expert 30B-A3B (Qwen3-Coder) trains here.

## Principles
1. **One variable at a time.** Map the curve before chasing a score.
2. **Honest hard bar.** Exact stdout match is the headline number; diagnostics sit beside it, never replace it.
3. **Memorize before generalize.** Rung 1 proves the plumbing can overfit a single task to 100%.
4. **Carry the scars forward.** The `unwrap_unit` splice fix and the dense body-sim reward term are hard requirements of any harness rewrite — see [01-design.md](01-design.md).
5. **Slow is the feature.** Run the full ladder, read the sweet spot off the curve, don't shortcut.

## Where things live
- [01-design.md](01-design.md) — task formats, holdout, ladder, reward, eval (the "how").
- [02-results.md](02-results.md) — living results table, filled per rung.
- [RL_WEEKEND_RESULTS.md](RL_WEEKEND_RESULTS.md) — archive of what failed and why (prior art).
```

- [ ] **Step 2: Verify**

Run: `test -f docs/rl/00-overview.md && echo OK`
Expected: `OK`.

---

### Task 3: Write docs/rl/01-design.md (the how)

**Files:**
- Create: `docs/rl/01-design.md`

**Interfaces:**
- Consumes: Global Constraints (verbatim).
- Produces: the design of record for tasks, holdout, ladder, reward, eval.

- [ ] **Step 1: Write the file** with this content:

```markdown
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
Build **all** extractable hole-fill tasks from `this_is_jac/` (estimate ~50–80). Reserve the holdout pool (below) before assigning any task to training.

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
1. Confirm the exact fresh `qwen3coder` HF repo at build time (assumed `Qwen/Qwen3-Coder-30B-A3B-Instruct`).
2. Type B AST-equivalence grader — design when the whole-file track starts.
```

- [ ] **Step 2: Verify**

Run: `test -f docs/rl/01-design.md && echo OK`
Expected: `OK`.

---

### Task 4: Write docs/rl/02-results.md (living table skeleton)

**Files:**
- Create: `docs/rl/02-results.md`

**Interfaces:**
- Produces: the empty results scaffold filled in as rungs run.

- [ ] **Step 1: Write the file** with this content:

```markdown
# Jac RL — Results (living)

Filled in as each rung runs. Pass = exact stdout (Type A). Empty cells = not yet run. See [01-design.md](01-design.md) for the ladder definition.

## Hole-fill ladder

Holdout = fixed ~15-task pool (same every rung). `mem` = trained-tasks re-eval, `gen` = holdout.

| rung (train N) | model | base | SFT mem | SFT gen | SFT+GRPO gen | raw-GRPO gen |
|---|---|---|---|---|---|---|
| 1 | jac-qwen3coder | | (target 100%) | n/a | | |
| 1 | qwen3coder | | (target 100%) | n/a | | |
| 3 | jac-qwen3coder | | | | | |
| 3 | qwen3coder | | | | | |
| 5 | jac-qwen3coder | | | | | |
| 5 | qwen3coder | | | | | |
| 10 | jac-qwen3coder | | | | | |
| 10 | qwen3coder | | | | | |
| 20 | jac-qwen3coder | | | | | |
| 20 | qwen3coder | | | | | |
| all | jac-qwen3coder | | | | | |
| all | qwen3coder | | | | | |

> Rung 1 `gen` is `n/a` — the memorize rung evaluates the trained task itself (mem column).

## Diagnostics log
Per run, also record: graded score, near-pass (osim≥0.9), avg-osim. Append notes here as the curve develops.

## Sweet spot
_To be read off the curve once the ladder is run: the train-N where fixed-holdout pass plateaus._

## Whole-file track
_Started after the hole-fill ladder is mapped. Table TBD with the AST grader design._
```

- [ ] **Step 2: Verify**

Run: `test -f docs/rl/02-results.md && echo OK`
Expected: `OK`.

---

### Task 5: Write docs/rl/README.md (index)

**Files:**
- Create: `docs/rl/README.md`

- [ ] **Step 1: Write the file** with this content:

```markdown
# docs/rl

Slow, ground-up redesign of the Jac RL workflow. Read in order:

1. [00-overview.md](00-overview.md) — goal, north star, the two models, principles.
2. [01-design.md](01-design.md) — task formats, holdout, training ladder, reward, eval.
3. [02-results.md](02-results.md) — living results table, filled per rung.
4. [RL_WEEKEND_RESULTS.md](RL_WEEKEND_RESULTS.md) — archive: what the first GRPO run found and why it failed.

**One-line status:** restarting from scratch on a measurable ladder; SFT is the proven lever, GRPO re-tested as a controlled challenger; dataset is `this_is_jac` only; Qwen3.6 removed.
```

- [ ] **Step 2: Verify**

Run: `test -f docs/rl/README.md && echo OK`
Expected: `OK`.

---

### Task 6: Delete old RL docs

**Files:**
- Delete: `docs/rl/RL_RESULTS.md`, `docs/rl/runlog.md`, `docs/rl/strat.md`, `docs/rl/workflow.md`, `docs/documentation/00-rl-phase.md`

- [ ] **Step 1: Confirm the new docs exist first (don't delete before replacements are in place)**

Run: `for f in 00-overview 01-design 02-results README RL_WEEKEND_RESULTS; do test -f docs/rl/$f.md || echo "MISSING $f"; done; echo done`
Expected: `done` with no `MISSING` lines.

- [ ] **Step 2: Delete the superseded docs**

```bash
git rm docs/rl/RL_RESULTS.md docs/rl/runlog.md docs/rl/strat.md docs/rl/workflow.md docs/documentation/00-rl-phase.md
```

- [ ] **Step 3: Verify**

Run: `ls docs/rl` then `ls docs/documentation`
Expected: `docs/rl` shows only the 5 new/kept docs; `docs/documentation` no longer lists `00-rl-phase.md`.

---

### Task 7: Destructive cleanup — RL adapters, models, results, dataset, harness (~200GB)

**Files (delete):**
- `adapters/jac-qwen3coder-grpo`, `adapters/jac-qwen3coder-rft`, `adapters/jac-qwen3coder-star0-rft`, `adapters/jac-qwen3coder-star1-rft`, `adapters/jac-qwen3coder-star2-rft`, `adapters/qwen36-star0-rft`, `adapters/qwen3coder-grpo`, `adapters/qwen3coder-rft`, `adapters/qwen3coder-star0-rft`, `adapters/qwen3coder-star1-rft`, `adapters/qwen3coder-star2-rft`
- `models/jac-qwen3coder-rft-q4`, `models/jac-qwen3coder-star0-rft-q4`, `models/jac-qwen3coder-star1-rft-q4`, `models/jac-qwen3coder-star2-rft-q4`, `models/qwen3coder-rft-q4`, `models/qwen3coder-star0-rft-q4`, `models/qwen3coder-star1-rft-q4`, `models/qwen3coder-star2-rft-q4`, `models/qwen36-q4`
- `results/passk`, `results/qwen36`, `results/qwen36-star0`, `results/jac-qwen3coder-star0`, `results/jac-qwen3coder-star1`, `results/jac-qwen3coder-star2`, `results/qwen3coder`, `results/qwen3coder-star0`, `results/qwen3coder-star1`, `results/qwen3coder-star2`, `results/RL_RESULTS.md`
- `dataset/rl/` (entire dir — regenerated fresh)
- `rl/` (entire harness dir — full fresh rewrite later)

**Files (KEEP — do not touch):** `models/jac-qwen3coder-q4`, `sft_dpo/`, `results/RESULTS.md`, `results/comparison`, `results/_builder`, `results/_evals`, `results/gemma*`, `results/qwen` (SFT/DPO probe), `models/gemma-*`, `models/qwen-jac-*`, `models/qwen-q4/q8`, `models/gemma-q4/q8`, `docs/rl/RL_WEEKEND_RESULTS.md`.

> Ambiguity guard: `results/qwen3coder` (RL eval) is deleted, but `models/jac-qwen3coder-q4` (the SFT+DPO base) is KEPT. `results/qwen` is the SFT/DPO python→jac probe — KEEP. When in doubt, list the dir contents and stop.

- [ ] **Step 1: Dry-run — print exactly what will be deleted and its size, delete nothing**

```bash
RL_PATHS=(
  adapters/jac-qwen3coder-grpo adapters/jac-qwen3coder-rft
  adapters/jac-qwen3coder-star0-rft adapters/jac-qwen3coder-star1-rft adapters/jac-qwen3coder-star2-rft
  adapters/qwen36-star0-rft adapters/qwen3coder-grpo adapters/qwen3coder-rft
  adapters/qwen3coder-star0-rft adapters/qwen3coder-star1-rft adapters/qwen3coder-star2-rft
  models/jac-qwen3coder-rft-q4 models/jac-qwen3coder-star0-rft-q4 models/jac-qwen3coder-star1-rft-q4
  models/jac-qwen3coder-star2-rft-q4 models/qwen3coder-rft-q4 models/qwen3coder-star0-rft-q4
  models/qwen3coder-star1-rft-q4 models/qwen3coder-star2-rft-q4 models/qwen36-q4
  results/passk results/qwen36 results/qwen36-star0
  results/jac-qwen3coder-star0 results/jac-qwen3coder-star1 results/jac-qwen3coder-star2
  results/qwen3coder results/qwen3coder-star0 results/qwen3coder-star1 results/qwen3coder-star2
  results/RL_RESULTS.md dataset/rl rl
)
for p in "${RL_PATHS[@]}"; do
  if [ -e "$p" ]; then du -sh "$p"; else echo "ABSENT  $p"; fi
done
echo "--- KEEP sanity (these MUST still exist) ---"
for k in models/jac-qwen3coder-q4 sft_dpo results/RESULTS.md docs/rl/RL_WEEKEND_RESULTS.md; do
  test -e "$k" && echo "KEEP-OK $k" || echo "KEEP-MISSING $k"
done
```
Expected: a size line per deletable path (total ≈200GB), no `KEEP-MISSING` lines. **Review this output before Step 2.**

- [ ] **Step 2: Execute the deletion** (only after Step 1 output looks right)

```bash
RL_PATHS=(
  adapters/jac-qwen3coder-grpo adapters/jac-qwen3coder-rft
  adapters/jac-qwen3coder-star0-rft adapters/jac-qwen3coder-star1-rft adapters/jac-qwen3coder-star2-rft
  adapters/qwen36-star0-rft adapters/qwen3coder-grpo adapters/qwen3coder-rft
  adapters/qwen3coder-star0-rft adapters/qwen3coder-star1-rft adapters/qwen3coder-star2-rft
  models/jac-qwen3coder-rft-q4 models/jac-qwen3coder-star0-rft-q4 models/jac-qwen3coder-star1-rft-q4
  models/jac-qwen3coder-star2-rft-q4 models/qwen3coder-rft-q4 models/qwen3coder-star0-rft-q4
  models/qwen3coder-star1-rft-q4 models/qwen3coder-star2-rft-q4 models/qwen36-q4
  results/passk results/qwen36 results/qwen36-star0
  results/jac-qwen3coder-star0 results/jac-qwen3coder-star1 results/jac-qwen3coder-star2
  results/qwen3coder results/qwen3coder-star0 results/qwen3coder-star1 results/qwen3coder-star2
  results/RL_RESULTS.md dataset/rl rl
)
for p in "${RL_PATHS[@]}"; do rm -rf "$p"; done
echo "done"
```
Expected: `done`. (Most paths are git-ignored model weights; `git rm` is not used here. `results/RL_RESULTS.md` and any tracked files will show as deletions in `git status`.)

- [ ] **Step 3: Verify deletion + keeps intact**

Run: `du -sh adapters models results dataset 2>/dev/null; echo '--'; test -d models/jac-qwen3coder-q4 && test -d sft_dpo && echo KEEPS-OK`
Expected: `adapters`/`models` much smaller than before; `KEEPS-OK` printed.

---

### Task 8: Commit the redesign

- [ ] **Step 1: Stage and commit**

```bash
git add docs/rl docs/documentation docs/superpowers/plans
git commit -m "$(cat <<'EOF'
docs(rl): reset RL to slow ground-up ladder; remove weekend artifacts

Rewrite docs/rl around an incremental memorize->generalize ladder
(rungs 1,3,5,10,20,all; fixed ~15-task holdout; two models:
jac-qwen3coder vs fresh qwen3coder; SFT-first, GRPO as controlled
challenger). Move RL_WEEKEND_RESULTS.md into docs/rl as the lessons
archive. Drop superseded RL docs. Dataset = this_is_jac only;
Qwen3.6 removed. Destructive cleanup of ~200GB RL adapters/models/
results/dataset/harness handled separately.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 2: Verify**

Run: `git status` and `git log --oneline -1`
Expected: clean tree (ignored deleted weights aside), latest commit is the redesign.

---

## Self-Review

**Spec coverage:**
- North star / two capabilities / two models / Qwen3.6 removal → Task 2 (overview), Global Constraints. ✓
- Task formats (hole-fill + whole-file), holdout, ladder, reward, eval → Task 3 (design). ✓
- Living results scaffold → Task 4. ✓
- Move weekend archive → Task 1. ✓
- Delete old RL docs → Task 6. ✓
- Destructive cleanup (~200GB, adapters/models/results/dataset/harness) with keep-list → Task 7. ✓
- Carried bug fixes (unwrap_unit, dense body-sim) → Global Constraints + Task 3 §1/§4. ✓

**Placeholder scan:** Doc bodies are complete. Intentional "TBD" markers exist only in the *living results* doc (02-results) and the whole-file/AST-grader open items — these are genuine future-work placeholders for a results log, not plan gaps. ✓

**Type/path consistency:** All delete paths in Task 7 match the `ls` output captured during brainstorming. Keep-list and delete-list are disjoint. New doc filenames (`00-overview`, `01-design`, `02-results`, `README`) are referenced consistently across Tasks 2–6 and the cross-links inside each doc. ✓
