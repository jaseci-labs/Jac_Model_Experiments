# Data Generation Strategy

*100% synthetic Jac training data for finetuning a coding model (Gemma 4 26B A4B Instruct, MoE 3.8B active; Qwen3-Coder-30B-A3B / DeepSeek-V3-Lite as fallbacks).*

| | |
|---|---|
| Constraint | No real Jac corpus exists — fully synthetic. |
| Verification | Jac compiler (hard gate), cross-compiled tests (hard gate for deterministic categories), idiom judge, sampled manual review. |
| Generators | Claude Code (Max) for quality, DeepSeek/Qwen APIs for bulk, base Gemma for free negatives, finetuned Gemma vN for self-distillation, Cursor/Codex for diversity checks. |

---

## Three anchors

Synthetic-only needs explicit substitutes for the ground-truth distribution real data provides for free.

1. **Jac grammar = distribution anchor.** Every construct must appear at a target frequency. The coverage matrix (Recipe 1) enforces this; without it generators converge on Python-shaped subsets.
2. **Jac compiler + cross-compiled tests = the unlimited oracle.** Rejection sampling is free, so cheap generators are usable when verification is strict. The compiler alone is necessary but not sufficient (code can compile and still be wrong); following MultiPL-T (Cassano et al. 2024), unit tests generated in Python are compiled to Jac via a deterministic rule-based test compiler and run against translations — zero LLM in the test layer. For deterministic categories test pass is a hard gate beside compilation.
3. **Python = proxy distribution.** Jac is built on Python; frontier models produce excellent Python. Generating Python tasks at scale and translating to idiomatic Jac inherits real-world software shape (algorithms, control flow, error patterns) — the largest single signal source available.

---

## Twelve generation recipes

Recipes compose multiplicatively: one verified artifact seeds an SFT example, two DPO pairs, a debugging trace, a multi-turn conversation, a reasoning trace, and a Python translation. Plan them as overlays, not buckets.

**R1 — Grammar-walked coverage matrix.** Parse the grammar into 40–80 constructs × 3 difficulty bands (atomic / idiomatic / composed). Target ~500 verified examples per cell → 60–120k SFT examples from coverage alone. The only mechanism that guarantees no feature is absent.

**R2 — Python ↔ Jac parallel corpus (MultiPL-T enhanced).** Highest-volume recipe (100k+ pairs/week).
1. Filter Python source aggressively: docstrings, Pyright type-check, returns a value, no TODO/FIXME, no HumanEval/MBPP overlap.
2. Generate Python unit tests for each function (5 suites at temp 0.8); require ≥90% line coverage.
3. Infer Python types from test execution; inject into the translation prompt.
4. Generate 50–100 candidate Jac translations per function at temp 0.8 (DeepSeek/Qwen).
5. Cross-compile Python tests to Jac with a deterministic rule-based compiler (handles `assert f(x)==y` for first-order values).
6. Validate translations with cross-compiled tests — hard gate, no manual review at scale.
7. Dedupe within candidates via ROUGE-L (threshold 0.6) after comment-stripping; keep diverse algorithms.

One pipeline → three datasets: (NL→Jac), (Python→Jac), (Python, Jac, NL) explanation triples. Failed translations become DPO negatives. Reject anything that compiles + passes tests but reads like Python with syntax swapped (idiom judge). **Interleaved (solution+tests) co-generation (SelfCodeAlign)** for directly-generated candidates: emit solution and tests in one completion, filter on execution — improves consistency vs separate test pass.

**R3 — Compiler-driven adversarial negatives.** Prompt: *"solve this Jac task the way a Python programmer would."* Compile-check: if it fails, you have a (broken, error, fix) triple; if it compiles but the idiom judge rejects, pair with the idiomatic version for DPO. Trains explicit avoidance of Python-shaped Jac.

**Mutual code↔test credibility ranking (CodeDPO).** Synthetic tests are untrusted: do not let any single test pick DPO winners. Per task, generate 15 candidate solutions (temp 1.5) + M candidate tests, build a bipartite pass/fail graph, iterate coupled PageRank-style scores (damping 0.85, ~10 iters). Pick DPO winners (high credibility) / losers (low credibility) from the converged scores (Spearman 0.86 vs 0.61 for naive "passes all tests"). Three hardening refinements: (a) **runtime-efficiency pairs** — among solutions passing top-credibility tests, time them and pair fast-vs-slow as preferred/rejected; (b) drop near-identical-credibility pairs (ambiguous noise); (c) **RPO loss** (DPO + weighted SFT on chosen) for stability.

**R4 — Bug-synthesis pipeline.** Mutate working code via a fixed catalog (drop return type, wrong walker dispatch, mismatched ability signature, off-by-one traversal, missing edge type, Python-style mutation). Capture (task, working, broken, error, fix). Have a generator role-play the debugging trajectory turn-by-turn for agentic data.

**R5 — Persona-stacked task generation.** Library of 30–50 personas (backend engineer migrating Flask, grad student exploring graph-spatial, performance engineer optimizing a walker, etc.). Same seed solution, different surface prompts. Include a controlled fraction of *bad* prompts (ambiguous, contradictory) for clarification-loop training.

**R6 — Evol-Instruct on Jac axes.** Five operators: deepen (topology constraint), broaden (different domain), constrain (force Jac-idiomatic), compose (chain tasks), invert (problem from solution). Track lineage for ablation. **Merge-all-rounds + Evol-Stop (WizardCoder):** keep every round merged with seed (preserves easy→hard gradient — combined beats any single round); halt against a held-out dev set when performance drops (~3 rounds optimal, non-monotonic); bound each round to ~10 new constraint words.

**R7 — Self-distillation loop.** After v0: vN generates → compiler/tests verify → keep passers → mix into training set → vN+1. Two non-negotiables: retain ≥30–50% frontier-model data per round (drift prevention), never relax verification (local-model errors are subtler). **Self-distill can beat cross-distill (SelfCodeAlign)** — validate per-axis; a stronger Python-priored teacher injects off-distribution accent the self-distilled samples avoid. **fastText recall-expansion (DeepSeek-Coder-V2):** seed a fastText classifier with validated Jac, iteratively mine the Python pool for translation-friendly functions, fold validated translations back into the seed each round.

**R8 — Multi-turn conversation synthesis.** Verified (task, code, tests) → 4–6 follow-ups sampling a fixed catalog (refinement / error report / edge case / optimization / refactor / explanation / conversion). Re-verify code after every code-changing turn. ~5 SFT samples per conversation.

**R9 — Reasoning-trace augmentation.** Every (task, code) gets a justification — explicitly including *why this Jac construct, not the Python-natural one*. Reasoning lives as a structured field in the SFT example. Reasoning-aware code data has been shown to substitute for model scaling and outperform positive-only sets at equal sample budgets — especially valuable when the base model has wrong (Python) priors.

**R10 — Doc-grounded lesson synthesis.** Per `skills.md` section: 3–5 worked examples, 5–10 practice problems with solutions, 5–10 common-mistake/correction pairs, 2–3 advanced cross-section compositions. Each example gets its own test suite for the "write tests" capability.

**R11 — OSS-Instruct snippet-seeded generation (Magicoder).** Feed the generator 1–15 random consecutive lines of real Python (the source pool) and prompt for an unrelated self-contained Jac problem *inspired by* the fragment — the orthogonal real-world domain signal is the point, not snippet reuse. **Abstract seed→concepts→instruction (SelfCodeAlign: 65.2 vs 59.8)** so the model draws on the domain idea rather than echoing an out-of-distribution format. Magicoder showed the lowest eval-set similarity yet the best downstream results — diversity, not leakage.

**R12 — Zero-seed template extraction (Magpie).** After v0 exists, prompt the finetuned Jac model with only the pre-query chat-template prefix; let auto-regression self-generate a fresh Jac instruction, then a second pass generates the response. Decode the instruction at temp 1.0–1.25 / top-p 0.99 for diversity, decode the response greedily (highest-prob tokens reflect the true learned distribution). Strongest lever against "no corpus exists" — fully novel instructions, no seed dependency. Full verification gates still apply.

---

## Generator allocation

| Generator | Best used for | Why |
|---|---|---|
| Claude Code (Max) | R1 orchestration, R4 debugging trajectories, R8 conversations, R9 reasoning, final refinement on cheaper outputs. | Quality bar setter. |
| DeepSeek / Qwen API | R2 bulk Python + translation, R4 mutations, R5 persona rewrites, R6 evolutions. Heavy rejection-sampling workloads. | Cheap; failures filtered by compiler+tests. |
| Cursor / Codex | Diversity sampling + judge-comparison only. | Limited budget — reserve for verification. |
| Base Gemma 4 (local) | Negative half of DPO pairs in R3 (and the Python-style negatives in R2). | Pre-finetune, its outputs ARE the failure mode. Free, unlimited. |
| Finetuned Gemma vN | R7 self-distillation, R12 zero-seed extraction. | Only generator with native Jac priors after v0. |

Mental model: Claude = quality + orchestration. Cheap APIs = volume. Base Gemma = free negatives. Finetuned Gemma = bootstrap.

---

## Volume targets (verified, deduplicated, gate-passed)

Generate ~5× these numbers before filtering (expect 20–40% compiler reject + 20–30% dedup/judge loss).

| Subset | Target | Source recipes |
|---|---|---|
| Core code generation SFT | 150–250k | R1, R2, R5, R6, R10 |
| Python ↔ Jac conversion (test-validated) | 80–150k | R2 |
| Debugging (broken → fix) | 30–60k | R3, R4 |
| Multi-turn conversations | 8–20k convs (~50–100k turns) | R4, R8 |
| Reasoning-augmented (overlay) | 60–120k | R9 |
| DPO preference pairs | 40–80k | R2, R3, R6 |
| Explanation (code → NL) | 20–40k | R9, R10 |

**Total clean: ~300–500k.** Raw before filtering: ~1.5–2.5M.

---

## Quality controls (synthetic-only specific)

- **Decontamination first, not last.** Write the eval set before generating; exclude any example with high 14-gram overlap. Synthetic problem spaces are bounded; chance collisions are likelier than in real-data pipelines.
- **Two-stage dedup.** MinHash on code first; sentence-embedding cosine on prose second. For R2: three stages — within the per-source candidates (ROUGE-L), then code-MinHash across the conversion dataset, then prose-cosine.
- **Distribution monitoring per batch.** Construct frequency, persona mix, difficulty bands, trigram-entropy. When a metric drifts, the generator is in a rut — adjust prompts immediately.
- **Semantic-domain coverage matrix (Magicoder).** A second axis orthogonal to the grammar matrix: cluster tasks into ~10 domains (algorithmic, DB/SQL, web, security, systems, data-processing, graph, math, CLI, domain-specific); balance them. The grammar matrix alone cannot detect domain collapse.
- **Min-neighbor-distance filtering (Magpie).** Use FAISS nearest-neighbor distance as a tunable diversity filter, not just a dedup detector — keep the most *isolated* examples. No single config wins everywhere; produce multiple differently-filtered subsets and blend.
- **Cross-family judge validation (Magpie).** Spot-check the idiom judge with an out-of-family judge (Qwen scoring Claude outputs and vice versa) to detect self-favoring bias on the most-used judge.
- **Keep a controlled fraction of stubbed samples (Magicoder).** Compiler-valid but partially-implemented code (with `pass` etc.) beat the fully-cleaned set in their ablation. Retain a small slice so the model sees realistic in-progress code.
- **Cosine-to-holdout as a quality diagnostic (Magicoder).** Prefer generation methods whose output is more *novel* (lower mean cosine to holdout) while still passing the gates. Distinct from decontamination-by-removal: here low is good.
- **Reward model on gate outcomes (DeepSeek-Coder-V2).** Train a reward model on the compiler/test 0–1 gate labels to produce a denser signal than the raw binary, then rank candidates and feed GRPO. R3 adversarial groups are ready-made GRPO comparison sets.

---

## Token accounting

Track tokens at two levels. **Per-example:** record `token_count` (and optional `prompt_token_count` / `completion_token_count`) per generated example — for context-window fit, the token-efficiency eval metric, and filtering over-long examples before training. **Aggregate:** log total tokens consumed per batch and per run, broken down by generator and recipe, for cost and budget tracking.

---

## Compounding insight

A single seed task can yield: 1 SFT example, 2 DPO pairs (R3), a debugging trace (R4), a multi-turn conversation (R8), a reasoning trace (R9), and a Python translation (R2 in reverse). Treat every verified artifact as a seed crystal that grows in multiple directions — that is how volume and diversity emerge from a fully synthetic pipeline.

---

## Open questions

- Final Jac construct list + target frequency weights for the coverage matrix.
- Idiom-judge rubric (single most-used judge — worth heavy prompt iteration).
- Eval holdout composition: 50–100 tasks per capability (generation, debugging, explanation, conversion, agentic, orchestration).
- Compiler harness throughput: ~5k examples/hour sustained needed for 1.5–2.5M raw candidates.
- Per-example provenance schema (recipe, generator, seed, evolution path, verification levels) for later ablations.
