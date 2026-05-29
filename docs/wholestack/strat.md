# Whole-Stack Strategy: Jac Model Finetuning

*End-to-end from synthetic data generation to evaluation*

| Key | Value |
|---|---|
| Target language | Jac (Jaseci Labs) |
| Base model | Gemma 4 26B A4B Instruct (MoE, 3.8B active) |
| Hardware | Mac M5 Pro, 20-core GPU, 48GB unified RAM |
| Finetuning framework | MLX + mlx-lm (Apple Silicon native) |
| Verification | Jac compiler (hard gate), tests, LLM judge, manual review |
| Data regime | 100% synthetic |
| Scope | Data generation -> Finetuning -> Evaluation |

The pipeline runs Phase 0 (prereqs) -> Phase 1 (context) -> Phase 2 (generation, 12 recipes) -> Phase 3 (verification) -> Phase 4 (quality controls) -> Phase 5 (dataset assembly) -> Phase 6 (finetuning) -> Phase 7 (evaluation). Recipe-level detail lives in `docs/datagenstrat/strat.md`; the end-to-end diagram lives in `docs/wholestack/workflow.md`.

---

## Phase 0: Prerequisites and infrastructure setup

Everything runs on a single Mac M5 Pro -- Apple Silicon native, no CUDA. Use Python 3.11+ in a fresh venv. Install jaclang, MLX, mlx-lm, transformers/tokenizers/datasets, sentencepiece, huggingface_hub, openai SDK (for DeepSeek/Qwen OpenAI-compatible endpoints), anthropic SDK, datasketch, sentence-transformers, pandas/pyarrow, jsonschema, and python-dotenv.

API keys needed in a gitignored `.env`: `ANTHROPIC_API_KEY` (Claude Max -- quality + orchestration), `DEEPSEEK_API_KEY` (bulk), `QWEN_API_KEY` (bulk + diversity), `HF_TOKEN` (Gemma 4 is gated -- request access early).

**Storage estimate (~30-50 GB working set, keep 100 GB free):**

| Item | Size |
|---|---|
| Gemma 4 26B FP16 weights | ~52 GB (only for conversion) |
| Gemma 4 26B Q4 MLX | ~13 GB (primary training format) |
| Gemma 4 26B Q8 MLX | ~26 GB (higher-quality option) |
| Raw generated data (~2M examples) | ~8-15 GB |
| Verified clean dataset (~400k examples) | ~2-4 GB |
| LoRA checkpoints | ~200-500 MB each |

**Jac compiler check.** Verify `jac run` on a hello-world and that bad syntax errors. Wrap the compiler in a thin Python helper: a 30-second-timeout subprocess wrapper around `jac check` returning pass/fail + stderr + exit code. This wrapper is called millions of times -- it must be fast, parallelizable, and never hang.

**MLX check.** Confirm `mx.default_device()` returns `Device(gpu, 0)` and unified-memory allocation succeeds. Do not proceed to training without GPU access.

**Model download and quantization.** Download Gemma 4 26B A4B Instruct from HuggingFace, convert to MLX format, and quantize to Q4 (~13 GB) for training and Q8 (~26 GB) for evaluation.

**Memory estimates for training with LoRA on 48GB unified RAM:**

| Quantization | Model size | Training memory | Feasibility |
|---|---|---|---|
| Q4 (4-bit) | ~13 GB | ~18-25 GB | Comfortable, batch size 2-4 |
| Q8 (8-bit) | ~26 GB | ~33-40 GB | Tight, batch size 1 with grad accum |
| FP16 | ~52 GB | Does not fit | Not feasible |

Start with Q4. Move to Q8 only if Q4 trains stably and quality evaluations suggest quantization artifacts.

---

## Phase 1: Context preparation

### Grammar extraction and construct catalog

Parse the Jac grammar (from the `jaclang` source or docs) into a flat construct list. Every grammar construct must appear in the training data. Target 40-80 constructs depending on granularity. List both `spawn` and `++>` because the model must recognize both forms.

| # | Construct | Category | Example syntax |
|---|---|---|---|
| 1 | `walker` | Core | `walker MyWalker { ... }` |
| 2 | `node` | Core | `node MyNode { ... }` |
| 3 | `edge` | Core | `edge MyEdge { ... }` |
| 4 | `ability` | Core | `can my_ability() -> str { ... }` |
| 5 | `archetype` | Core | `obj MyObj { ... }` |
| 6 | `with entry` | Entry | `with entry { ... }` |
| 7 | `spawn` | Graph ops | `root ++> MyNode()` |
| 8 | `visit` | Traversal | `visit [-->]` |
| 9 | `disengage` | Traversal | `disengage;` |
| 10 | Type annotations | Types | `x: int = 5` |
| 11 | `enum` | Types | `enum Color { RED, GREEN, BLUE }` |
| 12 | `test` | Testing | `test "my test" { ... }` |
| ... | ... | ... | ... |

### skills.md role

`skills.md` (the Jac MCP idiom reference) is used in three places: (1) **generator context** -- relevant sections injected into every generation prompt so the generator knows what idiomatic Jac looks like; (2) **idiom judge context** -- the LLM judge scores against it; (3) **eval rubric** -- benchmarks reference it to define correct Jac style. Build a doc corpus by extracting `skills.md` plus cheatsheet, patterns, and pitfalls, organized by topic for selective injection.

### Coverage matrix

2D grid of constructs (rows) x difficulty bands (columns). This is the master plan for Recipe 1.

| Difficulty band | Definition | Target per construct |
|---|---|---|
| Atomic | Construct in isolation, minimal context | ~100-150 examples |
| Idiomatic | Used the way real Jac would use it | ~200-250 examples |
| Composed | Combined with 2-3 other constructs | ~150-200 examples |

40-80 constructs x ~500 examples = 20,000-40,000 unique seed tasks before downstream multiplication. Track per-cell generation progress; cells where the generator struggles deserve more effort, not less.

### Eval holdout MUST be built before generation

Non-negotiable ordering: the eval set exists before any generator runs. Created afterward, decontamination is impossible. 50-100 tasks for each of six capabilities (code generation, debugging, conversion, explanation, agentic, orchestration) for 300-600 eval tasks total. Hand-written or hand-curated. Stored in `eval/holdout/` with a versioned manifest. Never used for training, never shown to generators.

### 14-gram decontamination reference

From the eval holdout, extract task-description text, reference-solution code, and 14-gram shingle sets of both. Every new generated example is checked against this reference set with MinHash. Examples sharing >50% of shingles with any eval task are flagged for manual review. The check is O(1) per example and runs on every batch -- catching contamination early saves generation budget.

---

## Phase 2: Data generation

The twelve generation recipes (R1-R12), with per-recipe goals, inputs, processes, expected yields, and failure modes, live in `docs/datagenstrat/strat.md`. This phase covers only the wholestack-specific concerns: generator allocation across providers, repo-level packing, and weekly batch sequencing.

### Generator allocation

| Generator | Budget tier | Best recipes | Rate notes |
|---|---|---|---|
| Claude Max (Anthropic) | Quality | R1 orchestration, R4 debugging, R8 multi-turn, R9 reasoning, final refinement | Max plan limits; batch carefully |
| DeepSeek API | Cheap bulk | R2 Python translations, R4 mutations, R5 persona rewrites, R6 evolutions | ~$0.14/M in, $0.28/M out |
| Qwen API | Cheap bulk | R2 parallel generation, R5, R6, diversity | Similar to DeepSeek |
| Base Gemma 4 (local) | Free | R3 adversarial negatives | Unlimited, M5 Pro inference-bound |
| Finetuned Gemma vN | Free | R7 self-distillation (v0 onward) | Unlimited, behind compiler gate |
| Cursor / Codex | Limited | Diversity sampling, judge comparison | Budget-limited |

**Mental model:** Claude for quality and orchestration, cheap APIs for volume, base Gemma for free negatives, finetuned Gemma for the bootstrap. Never burn Claude budget on tasks cheaper generators can do behind the compiler gate.

### Repo-level synthetic projects (DeepSeek-Coder)

Single-file snippets do not teach cross-file structure. Synthesize multi-file Jac projects (a handful of files with `import`/`include` and cross-file node-walker references). Build a file-dependency graph (one node per file, directed edge per dependency). Topologically sort with a min-in-degree variant that tolerates import cycles -- real Jac projects have them. Handle disconnected subgraphs separately. Prepend a file-path comment (`# path: project/walkers/traverse.jac`) to every file before packing, then concatenate the ordered files into a single training sample. DeepSeek-Coder's ablation showed removing this measurably drops cross-file completion -- it is not optional. Compile the project as a whole through verification, and apply the Phase 4 repo-level dedup.

### Batch scheduling and weekly sequencing

Not all recipes run in parallel from day one.

| Week | Recipes | Rationale |
|---|---|---|
| Week 1 | R1 pilot, R2 pilot, eval holdout finalization | Establish baselines, validate prompts, build verification pipeline |
| Week 2 | R1 scale, R2 scale, R3 starts (needs R1/R2 outputs), R10 | Volume generation begins, adversarial pairs need seed examples |
| Week 3 | R4, R5, R6 (all need seed examples from R1/R2) | Multiplicative recipes kick in |
| Week 4 | R8, R9 (need verified examples from prior recipes) | Multi-turn and reasoning overlays |
| Week 5+ | R7 (requires trained v0), continued scaling of R1-R6, R11-R12 | Self-distillation begins after first training run |

Cheap-API recipes (R2, R5, R6) run continuously in the background. Claude Max budget is reserved for R1 orchestration, R4 debugging trajectories, R8 conversations, and R9 reasoning.

---

## Phase 3: Verification pipeline

Every raw example passes through the pipeline in cost order (cheapest first).

**Stage 1 -- Compiler validation (hard gate).** Every `code`/`fixed_code`/`jac_code` field must compile; every `broken_code` field must NOT. 30-second timeout per example. Pin the Jac version. The verifier is a 30-second-timeout wrapper around `jac check` returning pass/fail + stderr. Parallelize across 8-12 processes on the M5 Pro for ~25-60 hours over 1.5-2.5M candidates.

**Stage 1.5 -- Cross-compiled tests (hard gate, deterministic categories).** For Recipe 2 outputs and other test-validatable categories, compile Python tests to Jac with a deterministic rule-based compiler (not an LLM), execute against the candidate Jac code. Failure rejects the candidate. If the test compiler cannot translate any assertion (zero surviving tests), route to manual review.

**Stage 2 -- Test suite + credibility scoring.** For deterministic code, generate 3-5 tests and compare outputs. This is a soft gate -- failures go to manual review. Replace single-test trust with CodeDPO mutual code-test credibility: weight each test by its credibility in the coupled solution-test bipartite graph (PageRank, ~10 iterations, damping 0.85); a solution's label is a credibility-weighted aggregate, not a raw count. Retain the per-example compiler/test 0-1 labels as training data for a reward model used in Phase 6 Stage 3.

**Stage 3 -- Idiom judge (LLM scoring).** Claude (or a capable open model) scores against `skills.md` on 1-5. Score >=4 accepted; 3 routed to manual review; <=2 rejected (or kept as DPO negative). Run only on examples that pass the compiler gate to save cost. Cross-validate with a judge from a different model family on a shared sample to catch family-blind-spot miscalibration.

**Stage 4 -- Manual sample review.** Randomly sample 5-10% of automated-passes. Reviewer checks task-code alignment, idiom correctness, no hallucinated constructs, and complexity-label accuracy. If a slice's manual pass rate drops below 80%, pause generation for that slice and investigate.

**Rejection routing.** raw_output -> compiler gate -> (fail) rejected/compiler_fail; (pass) -> test gate -> (fail) review/test_fail; (pass) -> idiom judge -> (<=2) rejected/idiom_fail; (==3) review/idiom_borderline; (>=4) -> manual sample -> (fail) rejected/manual_fail; (pass) clean_dataset.

**Expected survival rates:**

| Stage | Input | Pass rate | Output |
|---|---|---|---|
| Compiler gate | ~1.5-2.5M raw | 60-80% | ~1.0-1.8M |
| Cross-compiled tests | ~1.0-1.8M (deterministic) | 60-80% | ~0.7-1.3M |
| Test suite (non-cross subset) | ~1.0-1.8M | 70-90% | ~0.8-1.5M |
| Idiom judge | ~0.8-1.5M | 40-60% (>=4) | ~0.4-0.8M |
| Dedup + decontam | ~0.4-0.8M | 70-85% | ~0.3-0.6M |
| **Final clean dataset** | | | **~300-500k** |

**Retry rules.**
- Compiler fail with minor error (missing semicolon, wrong annotation): auto-fix targeted prompt, re-compile, up to 2 retries.
- Idiom judge score 3: regenerate "more idiomatic" with `skills.md` guidance, 1 retry.
- Test failure on edge cases: check if the test is wrong first (often is). If wrong, regenerate test. If code is wrong, manual review.
- Never retry more than twice. Generating new examples is cheaper than salvaging bad ones.

---

## Phase 4: Quality controls

- **Decontamination against eval holdout.** Run on every batch. 14-gram MinHash shingle compare; flag examples with Jaccard >0.5 against any eval task for manual review.
- **Distribution monitoring dashboard.** Live view of dataset shape after every batch:

| Metric | Target | Action if drifting |
|---|---|---|
| Construct frequency | Matches coverage matrix targets | Boost generation for underrepresented constructs |
| Persona distribution | Roughly uniform across 30-50 personas | Add underrepresented personas |
| Difficulty bands | 30% atomic, 40% idiomatic, 30% composed | Adjust difficulty parameter |
| Generator distribution | Per allocation table | Rebalance API budgets |
| Trigram entropy | > 0.8 (normalized) | Change prompts; add diversity instructions |
| Recipe source distribution | Per volume targets | Scale up underrepresented recipes |

- **Two-stage deduplication.** Stage 1 -- code-level MinHash (Jaccard >0.85) keeping the highest-idiom example per cluster. Stage 2 -- prose cosine similarity on task descriptions via `all-MiniLM-L6-v2`; flag pairs >0.92, manually review (some are legitimately similar with different solutions). Expected dedup rate 15-30%.
- **Category balance enforcement.** After dedup, check:

| Subset | Target | Tolerance |
|---|---|---|
| Core code generation SFT | 40-50% | +/- 5% |
| Python-Jac conversion | 20-25% | +/- 5% |
| Debugging (broken -> fix) | 8-12% | +/- 3% |
| Multi-turn conversations | 5-8% | +/- 2% |
| Reasoning-augmented overlay | 15-20% | +/- 5% |
| DPO preference pairs | 10-15% | +/- 3% |
| Explanation (code -> NL) | 5-8% | +/- 2% |

- **Batch-level regression checks.** Before merging a new batch: compiler pass rate within 10% of rolling average for that recipe, idiom judge mean shift <0.5 points, dedup rate against existing dataset <50% (else generator is exhausted).
- **Repo-level near-dedup.** A third dedup pass over concatenated synthetic projects -- two projects can share many individually-common files yet be distinct as wholes; project-level MinHash keeps distinct projects and removes genuine project duplicates.
- **Semantic-domain coverage matrix.** Orthogonal to grammar coverage: embed every example, cluster into ~10 domains (web, graph algorithms, data processing, simulation, ...), and track per-domain coverage. Drive R5 and R11 to fill thin domains.
- **Min-neighbor-distance FAISS filter (Magpie).** Embed every example, index with FAISS, keep the examples with the largest distance to their nearest neighbor (most isolated/novel); downsample dense clusters. Blend several filter configurations rather than one threshold.
- **Cross-family judge validation.** Validate the idiom judge with a judge from a different model family on a shared sample. Systematic disagreement = recalibrate the rubric.
- **Stubbed-fraction retention (Magicoder).** Do not filter down to only polished programs; keep a bounded slice of stubbed/partial samples so the model handles real incomplete code.
- **Cosine-to-holdout diagnostic per recipe.** Per-recipe mean cosine similarity of outputs vs. eval holdout. Diagnostic, not a hard filter: suspiciously close warrants a contamination audit; very low (e.g., OSS-Instruct) confirms orthogonal coverage.
- **Structural pretrain filters (DeepSeek-Coder-V2).** Cheap pre-judge filters: reject samples with avg line length >100 chars, max line >1000 chars, or <25% alphabetic characters. Catches minified, generated-blob, and data-dump content before it eats judge budget.

---

## Phase 5: Dataset assembly

### Training formats

- **SFT (instruction-following).** JSONL with messages format: `{"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": "```jac\n...\n```\n\n..."}]}`. For reasoning-augmented examples include a `<reasoning>...</reasoning>` block before the code in the assistant turn.
- **DPO (preference pairs).** JSONL: `{"prompt": ..., "chosen": ..., "rejected": ...}`.
- **Multi-turn (conversation arrays).** JSONL messages array alternating user/assistant for 4-6 turns; every code state along the way is compiler-verified.

### Stratified split rule

Train/val/test = 90/5/5. Stratify by category x difficulty x recipe source so no split is dominated by one slice. Use a stratification key like `category_difficulty_recipe` and `StratifiedShuffleSplit`. Note: this test split is for training dynamics; the Phase 1 eval holdout is the separate measurement instrument.

### Fill-in-the-Middle (FIM)

None of the recipes produce infill data. To learn IDE-style completion, take a verified complete Jac program, split into prefix/middle/suffix, reorder with PSM sentinels: `<|fim_start|>pre<|fim_hole|>suf<|fim_end|>middle<|eos|>`. Apply FIM at the document level before packing. DeepSeek-Coder's ablation: 50% FIM rate (PSM mode) is the sweet spot; 100% FIM hurts left-to-right.

### Composition ratios (DeepSeek-Coder v1.5)

| Component | Share | Source |
|---|---|---|
| Code (Jac, all recipes) | ~70% | Phase 2 output |
| Code-related NL (docs, comments, READMEs) | ~10% | Recipe 10, doc corpus |
| NL about code (explanations, Q&A) | ~7% | Recipe 9 |
| Math | ~7% | External math corpus |
| Bilingual NL | ~6% | External NL corpus |

The math and NL portions are cheap and prevent regression on reasoning while specializing in Jac.

### Per-example provenance

Every example carries full provenance, including `token_count` (prompt + completion under the target tokenizer) so packing, FIM selection, and Phase 7 token accounting need no retokenization. Fields:

```
id, recipe, generator, seed_task_id, evolution_path,
verification_levels {compiler, test_suite, idiom_judge_score, manual_reviewed},
context_bundle_version, source_prompt_version, generation_date,
token_count, dataset_version
```

### Release manifest

Each release ships a manifest with: `dataset_version`, `release_date`, `total_examples`, `splits {train,val,test}`, `by_category`, `by_difficulty`, `by_recipe`, `prompt_versions`, `context_bundle_version`, `jac_compiler_version`, and per-file `checksums`.

---

## Phase 6: Finetuning on Mac M5 Pro

MLX + mlx-lm: Apple Silicon native, unified memory (no CPU<->GPU copies), lazy evaluation, native Q4/Q8 quantization, LoRA built in. PyTorch+MPS works but is not purpose-built.

### Seed/source curation refinements

- **Return-value filter.** Keep only seed functions that return a value -- nothing-returning functions have vacuous tests and no Recipe 3 credibility signal.
- **Docstring-quality classifier.** LLM-score docstrings on seed candidates; drop missing/trivial/uninformative ones. The docstring becomes the task description, so its quality bounds the example's quality.

### LoRA configuration

- Rank 32 (start; bump to 64 if underfitting), alpha 64 (~2x rank), dropout 0.05.
- Target both attention (`q/k/v/o_proj`) and expert MLP projections (`gate_proj`, `up_proj`, `down_proj`) -- MoE expert layers are where Jac-specific knowledge concentrates.
- LR 5e-5 (Stage 1), cosine schedule, 10% warmup, weight decay 0.01.
- Max seq length 4096 (8192 for multi-turn), per-device batch 2, gradient accumulation 16 (effective batch 32).
- Checkpoint every 500 steps, eval every 250.

### Memory budget (Q4/Q8 with LoRA)

| Config | Weights | Optim states | Activations | Total | Batch |
|---|---|---|---|---|---|
| Q4 + LoRA rank 32 | ~13 GB | ~1 GB | ~4-8 GB | ~18-22 GB | 2-4 |
| Q4 + LoRA rank 64 | ~13 GB | ~2 GB | ~4-8 GB | ~19-23 GB | 2-3 |
| Q8 + LoRA rank 32 | ~26 GB | ~1 GB | ~4-8 GB | ~31-35 GB | 1-2 |
| Q8 + LoRA rank 64 | ~26 GB | ~2 GB | ~4-8 GB | ~32-36 GB | 1 |

Start with Q4 + rank 32 + grad accum 8-16. ~25 GB headroom.

### Four-stage training

**Stage 1 -- Core SFT.** Code generation + Python-Jac conversion + explanation, ~200-350k examples, 2-3 epochs, LR 5e-5 cosine + 10% warmup. Starts from base Gemma. Estimate ~18,750 steps total, ~1-2 sec/step on M5 Pro = ~15-30 hours.

**Stage 2 -- Specialized SFT.** Debugging + reasoning traces, ~80-150k examples, 2-3 epochs, LR 3e-5 (lower to avoid catastrophic forgetting). Starts from Stage 1 adapter.

**Stage 3 -- Preference alignment (DPO or GRPO).** ~40-80k pairs, 1-2 epochs (DPO overfits fast), LR 1e-5, DPO beta 0.1. Starts from Stage 2 adapter. mlx-lm may not natively support DPO -- fallback options: custom MLX script, PyTorch+MPS, or convert adapters to PyTorch for cloud GPU DPO only and convert back. **GRPO alternative (DeepSeek-Coder-V2):** drops the PPO critic, estimates the baseline from the mean reward of a sampled group on the same prompt. The Recipe 3 adversarial groups are ready-made GRPO comparison groups; the Phase 3 reward model trained on retained gate outcomes (binary compiler/test labels) supplies a continuous reward signal denser than the raw gate. **Source-matched LR (SelfCodeAlign):** ~1e-5 for self-generated data (R7, R12), ~2e-5 for cross-model data; if mixed, lean toward the lower rate.

**Stage 4 -- Multi-turn fine-tuning.** ~8-20k conversations (4-6 turns), 2-3 epochs, LR 2e-5, max seq 8192. Starts from Stage 3 adapter (or Stage 2 if DPO is done separately).

### Mac training tips

- Close other apps -- everything shares the 48 GB. A browser can easily eat 4-8 GB the trainer needs.
- Use Q4 to maximize batch; LoRA adapters are full-precision, so quality loss is small.
- Use `--grad-checkpoint` to trade ~20-30% more compute for ~30-40% less memory.
- Save adapters every 250-500 steps -- crashes lose everything since the last checkpoint, and adapters are small (~200-500 MB).
- Run under `caffeinate -dims ...` to prevent sleep/dim/disk-spindown during multi-day runs.
- Watch Activity Monitor: Memory Pressure must stay green/yellow; GPU Utilization should stay >80% (low = data loader is the bottleneck).
- Expect ~3-5x slower than an A100; a 4-hour A100 stage may take 12-20 hours here. Test with 100 examples / 10 steps before launching multi-day runs.

---

## Phase 7: Evaluation

Every metric is computed on the Phase 1 eval holdout (never seen during training).

### Automated metrics

- **Compiler pass rate.** Single most important metric. Target >90% (base Gemma 4 likely <30%). Report overall and per construct category / difficulty band.
- **Test pass rate.** Run generated code on deterministic eval tasks. Target >80%.
- **Construct coverage.** Parse each response, count which Jac constructs are used, compare to coverage-matrix targets. Catches model collapse onto a subset of features.
- **Syntax validity rate.** Line-by-line validity; distinguishes localized failures from systemic syntax confusion.

### Judge metrics

- **Idiom judge.** Same 1-5 rubric from Phase 3 applied to eval outputs; target mean >=4.0, with delta from base >=1.5.
- **Reasoning quality judge.** Scores explanation accuracy, specificity (does it name constructs and explain why?), and insight (does it explain why Jac patterns beat Python equivalents?).
- **Code quality judge.** Readability, efficiency, and pattern adherence (graph/walker usage where appropriate).

### Capability benchmarks (50-100 tasks each, held out from training)

- **Code generation (NL -> Jac).** 20 simple / 40 medium / 40 hard. Compiler pass rate, test pass rate, idiom judge.
- **Debugging.** 50 broken-code + error-message -> fix tasks across syntax / type / dispatch / scope / Python-ism errors. Fix success rate and fix accuracy (minimal vs. rewrite).
- **Conversion (Python -> Jac).** 50 tasks. Critical metric is idiom score, not compilation -- include tasks where the "obvious" translation is wrong (Python class -> Jac node, not Jac object).
- **Explanation (Jac -> NL).** 50 tasks. LLM judge on accuracy, completeness, Jac-specific insight; ROUGE/BERTScore vs. expert references for surface similarity.
- **Agentic.** 50 multi-step tasks requiring planning, tool use, error recovery. Completion rate, steps to completion, intermediate reasoning quality.
- **Orchestration.** 50 sub-agent coordination tasks. Decomposition quality, sub-task correctness, assembly correctness.

### Base vs. finetuned delta

| Metric | Base | Finetuned target | Delta |
|---|---|---|---|
| Compiler pass rate | ~20-30% | >90% | +60-70% |
| Test pass rate | ~10-20% | >80% | +60-70% |
| Idiom judge mean | ~1.5-2.0 | >4.0 | +2.0-2.5 |
| Code gen pass@1 | ~15% | >85% | +70% |
| Debugging fix rate | ~10% | >70% | +60% |
| Conversion idiom score | ~1.5 | >3.5 | +2.0 |

Run a regression check on Python HumanEval before vs. after; if Python drops >5%, LoRA rank or LR is too aggressive.

### Ablations

| Ablation | Expected impact |
|---|---|
| Without R1 (coverage matrix) | Construct coverage drops, rare constructs missing |
| Without R2 (Python-Jac pairs) | Volume drops, conversion capability disappears |
| Without R3 (adversarial negatives) | Idiom quality drops, more Python-isms |
| Without R4 (bug synthesis) | Debugging capability drops |
| Without R8 (multi-turn) | Agentic capability drops, follow-ups break |
| Without R9 (reasoning traces) | Explanation quality drops, less "why" |
| Without Stage 3 DPO | Idiom discrimination drops |
| Data volume sweep (10k/50k/100k/300k/500k) | Reveals whether quality or volume is the bottleneck |

### Statistical rigor

- 2-3 independent training runs (different seeds) for the final configuration; report mean and std.
- 95% Wilson intervals on compiler/test pass-rate proportions.
- Report pass@1, pass@5, pass@10 for code generation using the unbiased `pass@k = 1 - C(n-c, k) / C(n, k)` estimator.
- Use McNemar (paired binary) or bootstrap on the delta for base-vs-finetuned and ablation-vs-full comparisons.

### Token accounting

Track tokens as a first-class output. Per-example prompt and completion token counts read from the `token_count` provenance field. Aggregate per batch and per run, broken out by generator and recipe so the cost per surviving example is attributable. A recipe with high token cost per accepted example may be worth retiring even at high raw volume.

### Runtime efficiency

Measured only over the correct subset (so the model is not rewarded for being fast-but-wrong). Two signals: **tokens-to-correct** (completion tokens spent on correct solutions) and **execution time** (wall-clock runtime of the compiled Jac program on test inputs). Pair fast-and-correct against slow-and-correct to track the runtime-efficiency axis introduced as a DPO/GRPO signal in Recipe 3.

---

## Open questions

1. **Final construct catalog and frequency weights.** Exact list and target frequencies need input from the Jac language team.
2. **Idiom judge prompt engineering.** The judge's rubric defines what "good Jac" means for the entire dataset; needs heavy calibration against human judgments before scaling.
3. **MLX DPO support.** May not be native in mlx-lm yet -- evaluate custom script vs. PyTorch+MPS vs. cloud-GPU-for-DPO-only.
4. **Jac compiler throughput at scale.** Need ~5k examples/hour sustained over 1.5-2.5M candidates; benchmark parallel-process ceiling on M5 Pro.
5. **Self-distillation quality floor.** At what v0 idiom score is Recipe 7 worthwhile? If v0 mean <3.0, self-distilled data may be too noisy even past verification.
6. **Optimal LoRA rank for MoE.** Run a small rank sweep (16/32/64/128) on a 10k subset before committing the full budget.
7. **Eval holdout semantic contamination.** 14-gram catches surface contamination; add embedding-based similarity to catch same-task-different-words, with a threshold tuned against false positives.
