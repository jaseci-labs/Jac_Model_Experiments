# Jac Coding Agent — Project Context

**Confidential — Jaseci Labs**
For use as persistent context in AI-assisted development sessions.

---

## What Jac Is

Jac is a programming language built on top of Python, developed as part of the Jaseci project at Jaseci Labs. It is designed around a **data-spatial programming model**, where computation is expressed in terms of nodes, edges, walkers, and abilities rather than conventional imperative or object-oriented constructs.

Jac compiles to Python and can interoperate with the Python ecosystem, but its semantics and idioms are distinct enough that models trained primarily on Python, C, and JavaScript have very weak priors on correct Jac code.

Jac introduces several constructs that do not exist in mainstream languages:

| Construct | Description |
|---|---|
| **Walkers** | Traversal agents that navigate graph structures dynamically, not pre-computed paths |
| **Nodes** | First-class graph primitives for data modeling, not Python classes |
| **Edges** | Typed connections between nodes with their own properties |
| **Abilities** | Methods attached to nodes/walkers with dispatch rules |
| **Archetypes** | Inheritance hierarchies for nodes and walkers |
| **`with entry`** | Entry points for program execution |
| **`spawn` / `++>`** | Graph construction operators |
| **`visit` / `disengage`** | Walker traversal control flow |
| **`has` variables** | Typed properties on nodes, edges, and walkers |

The graph-spatial paradigm is fundamental: the primary problem domain is graph structure, not sequential procedures. A model that has not been specifically finetuned on Jac will produce code that looks superficially Python-like but is syntactically and semantically incorrect in ways that are hard to catch without a Jac compiler.

This is the core justification for the finetuning project: **no existing general-purpose model has meaningful Jac capability out of the box.**

---

## Project Goal

Build a coding agent for Jac that is functionally equivalent to what Claude Code is for Python or what Composer 2 (Cursor) is for general software development. The agent must operate autonomously on real Jac codebases, not just produce isolated snippets.

### Target Capabilities

| Capability | Description |
|---|---|
| **Code generation** | Take natural language descriptions and produce correct, compilable, idiomatic Jac code |
| **Debugging** | Take broken Jac code and a compiler/runtime error, identify root cause, produce a corrected version |
| **Explanation** | Read Jac code and produce accurate natural language explanations at any granularity (expression, block, module) |
| **Code conversion** | Convert Python code to idiomatic Jac, preserving behavior while adapting to graph-centric model and syntax |
| **Agentic task execution** | Handle multi-step engineering tasks autonomously: planning, tool use (MCP), error recovery, and iteration |
| **Subagent orchestration** | Decompose complex tasks into subtasks and coordinate execution across multiple steps without losing context |

The quality target is not "generates plausible Jac-looking code" but **"generates code that compiles, runs correctly, and follows Jac idioms."**

### End-State Deployment

The finetuned model deploys through **Jac MCP** (Model Context Protocol) in AI coding assistants. The MCP gives the agent live Jac compiler access, file system access, project structure awareness, and real-time compilation feedback.

**The user experience:** open Claude Code or Cursor with Jac MCP enabled, describe a Jac task in English, and the agent plans, writes, compiles, debugs, and iterates — all natively in Jac.

---

## Base Model

**Model: Gemma 4 26B A4B Instruct** (`google/gemma-4-26b-a4b-it`)

| Property | Value |
|---|---|
| Architecture | Mixture of Experts (MoE) Transformer |
| Total parameters | ~26 billion |
| Active parameters per token | ~3.8 billion (MoE routing) |
| Context window | 128K tokens |
| License | Apache 2.0 (unrestricted use, distribution, commercial deployment) |
| Release date | April 2, 2026 |
| LiveCodeBench v6 | 77.1% |
| AIME 2026 | 88.3% |
| τ2-bench (agentic tool use) | 86.4% (up from 6.6% on Gemma 3 27B) |
| Arena AI Elo (31B dense sibling) | 1452 (#3) |
| Finetuning framework | Unsloth + LoRA / QLoRA (cloud); MLX + LoRA (Apple Silicon) |
| Quantization for training | Q4 (~13 GB) — fits on single A100 40GB or Mac M5 Pro 48GB |
| Quantization for evaluation | Q8 (~26 GB) — higher quality inference |

### Why This Model

**MoE efficiency is the core enabler.** Only 3.8B parameters activate per token, making LoRA finetuning feasible on a single GPU while retaining 26B total knowledge capacity. The Q4-quantized model fits in ~13 GB, leaving ample room for training state.

**Agentic capabilities transfer to Jac.** The τ2-bench leap from 6.6% (Gemma 3) to 86.4% (Gemma 4) represents pre-trained tool use, multi-step planning, error recovery, and structured output generation — exactly what a Jac coding agent needs. Jac's walker-graph paradigm is inherently agentic: walkers traverse graphs, make decisions at nodes, and dispatch abilities based on context.

**Apache 2.0 removes all constraints.** The finetuned weights can be distributed as part of the Jaseci developer toolchain without restriction.

**Strong code benchmarks.** 77.1% LiveCodeBench v6 on post-cutoff problems demonstrates genuine novel code generation ability, not pattern matching.

### The Python Prior Problem

Like all frontier models, Gemma 4 has deep Python priors. Without finetuning, it will:
- Use function-based decomposition where Jac idiom calls for walkers
- Write class hierarchies where Jac uses archetypes and nodes
- Use iterative loops where Jac's traversal semantics are more appropriate
- Default to Python naming conventions and style patterns

A model that produces Python-with-Jac-syntax is worse than one that produces nothing — it looks correct but is fundamentally non-idiomatic. This is why the pipeline includes adversarial DPO pairs that explicitly teach the model what NOT to write, using the unfinetuned Gemma itself as a free source of exactly the wrong outputs.

---

## Available Resources

### Language Knowledge

Full internal access to Jac:
- Complete language syntax and grammar
- `skills.md` — the canonical skills and capability reference for idiomatic Jac
- All internal documentation, type system details, and runtime behavior
- Knowledge of every language construct, standard library, and idiom
- Ground truth on what correct Jac code looks like at every level of complexity

### Tooling

| Tool | Purpose |
|---|---|
| **Jac compiler** | Programmatic access for validating any generated code — pass/fail with error messages. Called millions of times during data generation. |
| **Jac MCP** | Model Context Protocol for tool use integration in coding assistants |
| **Unsloth** | Efficient LoRA finetuning framework (cloud/CUDA) |
| **MLX + mlx-lm** | Apple Silicon native training and inference (Mac M5 Pro) |

### Role

The person running this project works inside Jac at the source level. They have ground truth on all language behavior and can validate any output, write any test, and produce reference implementations for any task. Dataset construction and validation can be done with a level of accuracy not possible when working with a third-party language.

---

## Data Generation Strategy

The dataset is 100% synthetic. No existing public corpus of Jac code is large or diverse enough to use directly. Generation must be deliberate, validated, and heavily reviewed before scaling.

### Three Anchors for Synthetic-Only Data

The pipeline compensates for the absence of a real Jac corpus with three substitutes:

1. **Jac grammar is the distribution anchor.** Every language construct must appear in training data in proportion to real-world usage. The coverage matrix (Recipe 1) enforces this. Without it, generators converge on the Python-shaped subset.

2. **Jac compiler is the unlimited oracle.** Rejection sampling is free — generate 100 candidates, keep the 1 that compiles and passes tests, discard the rest. This lets cheap generators handle bulk volume. Every other anchor degrades if the compiler step is skipped. The compiler alone is insufficient for semantic correctness — cross-compiled tests from Python (following MultiPL-T, Cassano et al. 2024) serve as the second oracle.

3. **Python is the proxy distribution.** Real Python inherits the entire shape of real-world software problems. Translating validated Python to idiomatic Jac is the highest-volume data source available.

### Ten Generation Recipes

Recipes compose multiplicatively — a single seed task yields SFT examples, DPO pairs, debugging traces, multi-turn conversations, and reasoning traces.

| Recipe | Goal | Volume (verified) | Primary Generator |
|---|---|---|---|
| **R1: Grammar-walked coverage matrix** | Guarantee every Jac construct appears | 60–120k SFT | Claude Max |
| **R2: Python↔Jac parallel corpus** | Highest-volume recipe via MultiPL-T | 100k+ pairs/week | DeepSeek/Qwen APIs |
| **R3: Adversarial negatives (DPO)** | Teach what NOT to write | 40–80k pairs | Base Gemma 4 (free) |
| **R4: Bug-synthesis pipeline** | Debugging data + agentic trajectories | 30–60k debug | Claude Max + cheap APIs |
| **R5: Persona-stacked task gen** | Diversify task framing (30–50 personas) | 3–5x multiplier | Cheap APIs |
| **R6: Evol-Instruct on Jac axes** | Increase complexity (deepen, broaden, constrain, compose, invert) | 2–3x evolution | Cheap APIs |
| **R7: Self-distillation loop** | Scale beyond API budgets after v0 | 10–30k/round (free) | Finetuned Gemma vN |
| **R8: Multi-turn conversation synthesis** | Agentic interaction data | 8–20k convs (~50–100k turns) | Claude Max |
| **R9: Reasoning-trace augmentation** | Teach WHY Jac choices are right | 60–120k overlay | Claude Max |
| **R10: Doc-grounded lesson synthesis** | Guarantee every documented feature has data | 15–30k | Claude Max |

### Generator Fleet Allocation

| Generator | Role | Cost |
|---|---|---|
| **Claude Max** | Quality + orchestration (R1, R4, R8, R9, final refinement) | Quality tier |
| **DeepSeek/Qwen APIs** | Bulk volume (R2, R4, R5, R6) — ~$0.14/M input tokens | Cheap bulk |
| **Base Gemma 4 (local)** | Free adversarial negatives for DPO (R3) — outputs ARE the failure mode | Free, unlimited |
| **Finetuned Gemma vN** | Self-distillation from v0 onward (R7) — behind compiler gate | Free, unlimited |
| **Cursor / Codex** | Diversity sampling and judge-comparison inputs only | Limited budget |

**Mental model:** Claude is for quality and orchestration. Cheap APIs are for volume. Base Gemma is for free negatives. Finetuned Gemma is for the bootstrap. Never burn Claude budget on tasks cheaper generators can do behind the compiler gate.

### Two Generation Workflows

**Workflow 1: Scripted API Pipeline** (code_gen, debug, explanation, conversion)
- Script calls generation APIs with full Jac context in system prompt
- Returns structured JSON batches
- Every code field validated by Jac compiler
- Routed to clean/rejected/review based on validation gates
- Cross-compiled test validation for deterministic categories

**Workflow 2: Agentic Trajectory Capture** (trajectory)
- Real agent sessions in Cursor, Claude Code, or Codex with Jac MCP attached
- Agent receives task → plans → executes with MCP tools → compiler feedback → error recovery → final output
- Session transcript IS the training example
- Cannot be scripted — the trajectory itself is the artifact

### Tier 3: Full-App Generation (Proposed)

Extends the pipeline from isolated examples to complete Jac projects:
- Multi-file project structure with tests
- Self-repair loops: initial attempt → compiler error → diagnosis → patch → validate
- DPO preference pairs from failed-vs-fixed attempts
- CLI utilities, API services, data processing apps, graph-oriented programs

---

## Verification Pipeline

Every raw example passes through verification in order of cost (cheapest first):

### Stage 1: Compiler Validation (Hard Gate)

Every code field in every example runs through the Jac compiler. No exceptions.

| Field | Rule |
|---|---|
| `code`, `fixed_code`, `jac_code` | Must compile. Failure = rejection. |
| `broken_code` (debug examples) | Must NOT compile. If it compiles, the debug example is invalid. |

30-second timeout per compilation. Expected raw compile pass rate: 60–80%.

### Stage 1.5: Cross-Compiled Test Validation (Hard Gate — Deterministic Categories)

Following MultiPL-T methodology:
1. Generate unit tests in Python for the source function
2. Verify tests pass with ≥90% line coverage
3. Compile Python assertions to Jac using a **deterministic rule-based compiler** (not an LLM)
4. Run compiled tests against Jac translation
5. Pass = clean dataset. Fail = rejected (not routed to manual review at scale).

This catches semantic errors that compilation alone misses. Combined compiler + test gate yields ~40–60% survival from raw candidates.

### Stage 2: Test Suite Execution (Soft Gate)

For non-cross-compiled deterministic examples. 3–5 test cases per example. Failures route to manual review.

### Stage 3: Idiom Judge (LLM-Based Scoring)

LLM judge scores code against `skills.md` on a 1–5 scale:

| Score | Meaning |
|---|---|
| 5 | Fully idiomatic Jac — walkers, graph-spatial patterns, abilities used correctly |
| 4 | Good Jac — minor stylistic issues |
| 3 | Acceptable — compiles but has Python-isms |
| 2 | Weak — mostly Python patterns with Jac syntax |
| 1 | Not Jac — Python that happens to compile |

Score ≥4: accepted. Score 3: manual review. Score ≤2: rejected (optionally used as DPO negative).

### Stage 4: Manual Sample Review (5–10%)

Reviewer checklist: correctness, idiomatic Jac, clear task description, accurate explanation/reasoning, no hallucinated features, accurate complexity label. If pass rate drops below 80% for any slice, pause and fix before continuing.

### Expected Survival Rates

| Stage | Pass Rate | Cumulative |
|---|---|---|
| Compiler gate | 60–80% | ~1.0–1.8M |
| Cross-compiled tests | 60–80% | ~0.7–1.3M |
| Idiom judge (score ≥4) | 40–60% | ~0.4–0.8M |
| Dedup + decontamination | 70–85% | ~0.3–0.6M |
| **Final clean dataset** | | **~300–500k** |

---

## Quality Controls

### Decontamination

Eval holdout set (300–600 tasks, 6 capabilities) created BEFORE any generation. 14-gram shingle comparison via MinHash on every batch. >50% overlap flags for review. This is a non-negotiable ordering constraint.

### Distribution Monitoring

Live dashboard tracking construct frequency, persona distribution, difficulty bands, generator distribution, and trigram entropy. When metrics drift, the generator has fallen into a rut — adjust prompts.

### Two-Stage Deduplication

1. **Code-level MinHash** (Jaccard >0.85) — cheap, catches most duplicates
2. **Prose cosine similarity** (>0.92 via sentence-transformers) — catches paraphrased duplicates

For Recipe 2 multi-candidate translations: deduplicate within candidates first (ROUGE-L, threshold 0.6), then across full dataset.

Expected dedup rate: 15–30% of verified examples are near-duplicates.

---

## Volume Targets

Targets are for verified, deduplicated, compiler-passed examples. Generate ~5× before filtering.

| Subset | Target (verified) | Source Recipes |
|---|---|---|
| Core code generation SFT | 150–250k | R1, R2, R5, R6, R10 |
| Python↔Jac conversion pairs | 80–150k | R2 |
| Debugging (broken→fix) | 30–60k | R3, R4 |
| Multi-turn agentic conversations | 8–20k convs (~50–100k turns) | R4, R8 |
| Reasoning-augmented examples | 60–120k (overlay) | R9 |
| DPO preference pairs | 40–80k | R2, R3, R6 |
| Explanation (code→NL) | 20–40k | R9, R10 |
| **Total raw before filtering** | **~1.5–2.5M** | |
| **Total verified after filtering** | **~300–500k** | |

---

## Finetuning Strategy

### Hardware

| Environment | Hardware | Use Case |
|---|---|---|
| **Primary (testing)** | Mac M5 Pro, 20-core GPU, 48GB unified RAM, MLX | Model testing, prototyping, small-scale training |
| **Primary (full-scale)** | Single A100 40/80GB, Unsloth | Full 300k+ dataset training |

### LoRA Configuration

```yaml
adapter_type: lora
rank: 32                      # Start here, increase to 64 if underfitting
alpha: 64                     # 2x rank
dropout: 0.05
targets: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
learning_rate: 5e-5           # Stage 1 start
lr_schedule: cosine
warmup_steps: 200
batch_size: 2
gradient_accumulation_steps: 16  # Effective batch = 32
max_seq_length: 4096
```

**Why these target modules:** in MoE models, expert layers (gate/up/down_proj) hold specialized knowledge. Attention layers (q/k/v/o_proj) allow attending to Jac syntax differently. Together they cover the full transformation pipeline.

### Multi-Stage Training

| Stage | Data | Volume | LR | Epochs | Purpose |
|---|---|---|---|---|---|
| **1: Core SFT** | Code gen + conversion + explanation | ~200–350k | 5e-5 | 2–3 | Foundation: Jac syntax, semantics, idioms |
| **2: Specialized SFT** | Debugging + reasoning traces | ~80–150k | 3e-5 | 2–3 | Error recovery, reasoning about Jac choices |
| **3: DPO** | Preference pairs (idiomatic vs. Python-style) | ~40–80k | 1e-5 | 1–2 | Alignment: prefer Jac-native over syntactically-valid-but-wrong |
| **4: Multi-turn SFT** | Agentic conversations (4–6 turns) | ~8–20k convs | 2e-5 | 2–3 | Agentic: follow-ups, error recovery, iteration |

Adapters from each stage merge before the next stage begins. DPO is the critical alignment step that prevents Python-pattern fallback.

### Memory Estimates (Mac M5 Pro)

| Config | Model | Optimizer | Activations | Total | Batch Size |
|---|---|---|---|---|---|
| Q4 + LoRA rank 32 | ~13 GB | ~1 GB | ~4–8 GB | ~18–22 GB | 2–4 |
| Q8 + LoRA rank 32 | ~26 GB | ~1 GB | ~4–8 GB | ~31–35 GB | 1–2 |

### Training Duration Estimates (Mac M5 Pro)

| Stage | Examples | Steps (effective batch 32) | Time (est.) |
|---|---|---|---|
| Stage 1 (Core SFT) | ~200k | ~18,750 | ~15–30 hours |
| Stage 2 (Specialized) | ~80k | ~7,500 | ~6–12 hours |
| Stage 3 (DPO) | ~40k | ~3,750 | ~3–6 hours |
| Stage 4 (Multi-turn) | ~8k convs | ~750 | ~1–2 hours |

Full-scale training may move to cloud A100 if Mac proves too slow.

---

## Evaluation

### Automated Benchmarks

| Metric | Target | Base Gemma 4 (est.) |
|---|---|---|
| Compiler pass rate | >90% | ~20–30% |
| Test pass rate | >80% | ~10–20% |
| Idiom judge mean | >4.0 | ~1.5–2.0 |
| Code gen pass@1 | >85% | ~15% |
| Debugging fix rate | >70% | ~10% |
| Conversion idiom score | >3.5 | ~1.5 |

### Capability Benchmarks (Eval Holdout — Never Seen During Training)

| Capability | Tasks | What It Tests |
|---|---|---|
| Code generation (NL→Jac) | 100 | Syntax, semantics, idiomatic construct usage across 3 difficulty bands |
| Debugging (broken→fix) | 80 | Error diagnosis across syntax, type, semantic, Python-pattern, multi-construct bugs |
| Conversion (Python→Jac) | 80 | Idiomatic translation: direct mapping, pattern transformation, paradigm shift |
| Explanation (Jac→NL) | 60 | Comprehension: code walkthrough, concept explanation, comparison, error explanation |
| Agentic (multi-step) | 50 | Multi-turn: iterative refinement, error recovery, exploration |
| Orchestration (sub-agent) | 50 | Task decomposition and coordination |

### Ablation Studies

Train without each recipe individually to measure contribution:
- Without R1 (coverage matrix): construct coverage drops
- Without R2 (Python-Jac pairs): volume drops dramatically
- Without R3 (adversarial negatives): idiom quality drops, more Python-isms
- Without R9 (reasoning traces): explanation quality drops
- Without DPO stage: Python-pattern fallback increases

Data volume learning curves (10k → 50k → 100k → 300k → 500k) reveal whether more data or better data is the bottleneck.

### Statistical Rigor

- 2–3 independent training runs (different random seeds) for final configuration
- 95% bootstrap confidence intervals on all metrics
- pass@k reporting (pass@1, pass@5, pass@10) with unbiased estimator
- McNemar's test for base vs. finetuned comparison
- Regression check: Python/general coding performance must not drop >5%

---

## Self-Distillation Loop

Once v0 exists, it becomes a free generator:

```
v0 generates → compiler/tests verify → keep passing → mix into dataset → train v1 → repeat
```

**Non-negotiables:**
- Retain ≥30–50% frontier-model data each round to prevent quality drift
- Never relax verification gates — local models produce subtler errors
- Stop after 2–3 iterations when marginal improvement drops below threshold

---

## Dataset Schema Reference

### SFT Format (Instruction-Following)

```jsonl
{"messages": [{"role": "user", "content": "Write a Jac walker..."}, {"role": "assistant", "content": "```jac\nwalker SumWalker {...}\n```"}]}
```

### DPO Format (Preference Pairs)

```jsonl
{"prompt": "Convert this Python to idiomatic Jac: ...", "chosen": "```jac\nwalker ...\n```", "rejected": "```jac\ndef ...  # Python-style\n```"}
```

### Multi-Turn Format

```jsonl
{"messages": [{"role": "user", "content": "Build a task queue..."}, {"role": "assistant", "content": "..."}, {"role": "user", "content": "Handle duplicates?"}, {"role": "assistant", "content": "..."}]}
```

### Trajectory Format

```json
[
  {"role": "user", "content": "task description"},
  {"role": "assistant", "content": "reasoning and plan"},
  {"role": "tool_call", "content": "jac_mcp.compile(...)"},
  {"role": "tool_result", "content": "compiler output"},
  {"role": "assistant", "content": "response to compiler output"},
  {"role": "assistant", "content": "final output"}
]
```

### Per-Example Metadata

```json
{
  "id": "sft-code_gen-20260601-003-0042",
  "batch_id": "20260601-code_gen-003",
  "category": "code_gen|debug|explanation|conversion|trajectory",
  "complexity": "simple|medium|hard",
  "recipe": "R1|R2|...|R10",
  "generator": "claude-max|deepseek-v3|qwen3|gemma4-base|gemma4-finetuned-v0",
  "compiler_pass": true,
  "test_pass": true,
  "cross_compiled_tests_pass": true,
  "idiom_judge_score": 4.5,
  "manually_reviewed": false,
  "seed_task_id": "coverage-walker-idiomatic-017",
  "evolution_path": ["R1:seed", "R6:deepen", "R9:reasoning"],
  "source_prompt_version": "prompt-code_gen-v3",
  "context_bundle_version": "jac-context-v2",
  "generation_date": "2026-06-01",
  "dataset_version": "jac-synth-v1.0.0"
}
```

---

## Fixed Constraints

These decisions are not subject to change:

| Constraint | Value |
|---|---|
| Target language | Jac — the agent is Jac-specific, not general-purpose |
| Base model | Gemma 4 26B A4B Instruct — model selection is finalized |
| Compiler validation | Hard requirement for all code in training data — no unvalidated code enters the training set |
| Finetuning method | LoRA (Unsloth on cloud, MLX on Mac) — full finetuning not in scope |
| Model variant | Instruct — continued pretraining on base model not in scope |
| Quality target | Compiler-correct, idiomatically valid Jac — not approximate or partially correct output |
| License | Apache 2.0 — unrestricted distribution of finetuned weights |
| Data regime | 100% synthetic — no real Jac corpus available at scale |

---

## Project Status

**Current phase:** Data generation infrastructure complete. Scaling to full dataset.

| Milestone | Status |
|---|---|
| Pipeline infrastructure (generation, validation, review, release) | Complete |
| Documentation (strategy, operations, quality gates) | Complete |
| Test suite (13 modules, ~3,000 lines) | Complete |
| Pilot generation (187 clean examples across 5 categories) | Complete |
| Model testing strategy (3-model comparison on 5k examples) | Documented, not yet executed |
| Full-app generation tier (Claude Code + Jac MCP) | Documented, not yet executed |
| Full-scale data generation (~1.5–2.5M raw → ~300–500k verified) | Not started |
| Multi-stage LoRA finetuning | Not started |
| Evaluation and ablation studies | Not started |
| Deployment via Jac MCP | Not started |

---

## Key References

| Resource | Location |
|---|---|
| Data generation strategy (10 recipes) | `docs/newdatagenstrat/strat.md` |
| Whole-stack strategy (end-to-end) | `docs/wholestack/strat.md` |
| Whole-stack workflow diagram | `docs/wholestack/workflow.md` |
| Full-app generation tier | `docs/newdatagenstrat/README.md` |
| Model testing strategy | `docs/modeltesting/strategy.md` |
| Model evaluation methodology | `docs/modeltesting/evaluation.md` |
| Gemma 4 26B setup guide | `docs/modeltesting/gemma4-26b.md` |
| Pipeline architecture | `docs/pipeline.md` |
| Quality gates | `docs/quality.md` |
| Dataset layout and schemas | `docs/dataset.md` |
| Operations command reference | `docs/operations.md` |
| Current dataset snapshot | `docs/stats.md` |
| MultiPL-T paper | `papers/2308.09895v6.pdf` (Cassano et al. 2024) |
| Source code | `src/data_generation/` (13 modules, ~4,000 lines) |
| Test suite | `tests/` (13 modules, ~3,000 lines) |

---

## Open Questions

1. **Final construct catalog and frequency weights.** The exact 40–80 Jac constructs and their target frequencies drive Recipe 1 and distribution monitoring. Requires language team input.

2. **Idiom judge prompt engineering.** The single most-used LLM judge in the pipeline. Rubric determines what "good Jac" means for the entire dataset. Needs heavy iteration and calibration.

3. **MLX DPO support.** `mlx-lm` may not natively support DPO training. Fallback options: custom MLX script, PyTorch MPS backend, or cloud GPU for DPO stage only.

4. **Jac compiler throughput at scale.** Need ~5,000 examples/hour sustained across 1.5–2.5M raw candidates. Parallelize across CPU cores (M5 Pro: 12P + 4E cores → 8–12 parallel processes → 25–60 hours total).

5. **Self-distillation quality floor.** At what idiom judge score does v0 need to be before self-distillation is worthwhile? Below 3.0 mean may be too noisy.

6. **Optimal LoRA rank for MoE.** Expert layers may need different rank than attention layers. Small-scale rank sweep (16, 32, 64, 128) on 10k subset before committing.

7. **Training duration vs. quality on M5 Pro.** Multi-day runs risk crashes and thermal throttling. Consider shorter runs with checkpoint resumption.

8. **Eval holdout contamination.** 14-gram overlap catches lexical contamination. Semantic contamination (same task described differently) may need embedding-based similarity check.
