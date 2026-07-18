# Attempt 03: CPT → SFT/DPO → RL Stack (Design)

*Umbrella architecture. Each stage below gets its own follow-up spec+plan before build — this doc locks the shared framing so those don't get re-litigated per stage.*

## Problem

`01-sft-dpo` + `02-rl-grpo` closed with a consistent pattern (see `context.md`, `02-rl-grpo/RL_FINDINGS.md`):

- SFT fixes **syntax**: 94% syntax-error rate → 4.2% (`03-new/0710_JacLLM.pptx` slide 8).
- Problem/behavioral pass rate stays flat around **40%** regardless — the model produces syntactically valid Jac that's semantically wrong (wrong OSP idiom, wrong walker/graph pattern).
- GRPO/RL never moved greedy pass@1 above SFT, on two corpora, with a genuine (non-zero) reward gradient. Conclusion at the time: RL isn't the lever *because the ceiling above it (SFT) was itself syntax-bound, not semantics-bound*.

Hypothesis for this attempt: the missing stage is **domain adaptation** — the model has never seen enough Jac to build semantic priors (OSP graph/walker concepts, idiom), only enough to pattern-match syntax from few-shot SFT examples. Continual pre-training (CPT) on raw Jac corpora, before SFT/DPO, should move the semantic ceiling that SFT/DPO and GRPO were both bumping into.

## Architecture

Same base model through all stages, LoRA at every stage, 4-point ablation:

```
base → +CPT → +CPT+SFT/DPO → +CPT+SFT/DPO+GRPO
```

Eval (semantic MCQ + syntax/behavioral judge, see below) runs at **all four checkpoints** — this is what isolates CPT's specific contribution from SFT/DPO's and GRPO's, matching the ablation convention already used in the bakeoff and RL ladder.

## Base model

**Qwen3-Coder-30B-A3B-Instruct**, unchanged from the `01-sft-dpo` bake-off winner (`01-sft-dpo/docs/initmodelchoice/2026-06-26-sft-dpo-bakeoff-results.md`). Not re-litigating model choice here — CPT is the new variable; keep everything else fixed so the ablation is clean. Small-MoE (~3B active), Q4 fits 48GB M5 Pro for local MLX LoRA.

## Stage 1 — CPT (new)

Continual pre-training: LoRA, next-token prediction on raw text (no instruction format).

- **Corpus**: Jac docs, OSP paper, blogs — content TBD, this doc only fixes the *shape* of the corpus, not its contents. Gated by a follow-up CPT data-collection spec.
- **CF (catastrophic forgetting) guard**: mix in a slice of general code/Python data as rehearsal alongside the Jac corpus. Track a general-coding eval (e.g. HumanEval-style) at every CPT checkpoint alongside the semantic MCQ — any regression on general coding is a stop signal.
- **Open for the follow-up spec**: corpus sourcing, mix ratio, epoch count vs. data volume tradeoff (slide 16's "more data vs more epochs, avoid CF and memorizing").

## Stage 2 — SFT/DPO (redesigned data, same recipe)

Recipe and hyperparameters unchanged from `01-sft-dpo` (proven: 94% functional / 100% idiom-DPO) — runs on top of the CPT checkpoint instead of raw base.

**What changes: data composition.** Current SFT/DPO data over-indexes on syntax-fix pairs (compiles vs. doesn't-compile). Redesign shifts DPO preference-pair composition toward **semantically-correct vs. subtly-wrong OSP idiom** pairs — both compile, one uses the right walker/graph pattern and one doesn't. This is the pairing that actually exercises the semantic gap this stack targets; syntax-only contrastive pairs no longer teach the model anything new once CPT + existing SFT have fixed syntax.

## Stage 3 — RL/GRPO (redesigned corpus + reward)

Not a re-run of the closed null. `02-rl-grpo`'s RL_FINDINGS diagnosed three specific corpus limits, all fixed here:

1. **Source diversity.** Prior corpus was 100% `this_is_jac` (~66 tasks, ceiling). New corpus pulls tasks from multiple Jac source projects — this is what makes a real file-disjoint holdout possible (prior holdout leaked/degenerated because everything came from one project's files).
2. **Grading.** Prior grading was exact-stdout, all-or-nothing (no partial credit for correct-but-differently-formatted output). Build the Type-B AST-equivalence / partial-credit grader that was deferred in the prior RL work.
3. **Idiom balance.** Prior corpus had social_graph/walker tasks (the actual OSP-idiom, "semantic" tasks) at 48-57% concentration, concentrated in one file → unholdout-able. New corpus deliberately balances plain-function tasks against OSP-walker/graph tasks across multiple files.

**New falsifiable hypothesis** (distinct from the closed one): if CPT genuinely raises the semantic ceiling, GRPO — previously inert because both base and SFT were syntax-ceiling-limited, leaving no semantic headroom for RL to climb — may now show a real gradient above SFT/DPO. If GRPO is still flat after CPT, that's a stronger, cleaner null than the one already recorded.

## Evaluation

Two-track, explicitly separating semantic knowledge from behavioral correctness (pptx slide 19):

- **Semantic MCQ** (Plan 1): LLM+MCP-generated multiple-choice questions on Jac/OSP concepts, no compiler involved — isolates "does the model know the concept" from "can it currently type it correctly." Cheap, used to pick CPT checkpoints.
- **Online-judge** (Plan 2): existing compiler/pass@k harness (`02-rl-grpo/rl/`) — confirms the model can actually produce working, semantically-correct code. Run after SFT/DPO and GRPO stages.
- **Trust check**: before trusting either automated eval at scale, sample a subset, get human judgment, and compare via Hamming similarity against the LLM-graded result on that same subset (slide 19). Don't scale an eval that fails this check.

## Sequencing

This doc is the shared architecture/ablation/model/eval contract. Follow-up specs, in gating order:

1. **CPT data-collection spec** — corpus sourcing (Jac docs/OSP paper/blogs), mix ratio with general code, epoch/volume tradeoff. Gates everything else — nothing downstream can run until a CPT checkpoint exists.
2. **SFT/DPO redesign spec** — semantic-vs-syntax DPO pair composition, target ratios.
3. **RL/GRPO redesign spec** — multi-project corpus sourcing, Type-B AST-equivalence grader, idiom-balance targets.

## Out of scope for this doc

- Actual dataset contents for any stage (deferred to the three follow-up specs).
- Whether to revisit base model choice (fixed: Qwen3-Coder-30B-A3B).
- SFT/DPO recipe/hyperparameter changes (fixed: reuse `01-sft-dpo` recipe as-is; only data composition changes).
