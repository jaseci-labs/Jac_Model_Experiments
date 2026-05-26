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

---

## Phase 0: Prerequisites and infrastructure setup

### Software environment

The entire stack runs on a single Mac M5 Pro. No cloud GPU instances, no CUDA, no Docker-for-GPU. Everything must be Apple Silicon native or pure Python.

Create a dedicated conda or venv environment to isolate dependencies. Python 3.11+ is required because MLX and mlx-lm target recent Python releases and the Jac compiler itself requires 3.11+.

```bash
# Create and activate environment
python3.11 -m venv ~/jac-finetune-env
source ~/jac-finetune-env/bin/activate

# Core dependencies
pip install mlx mlx-lm
pip install jaclang                    # Jac compiler
pip install transformers tokenizers datasets
pip install sentencepiece protobuf     # Gemma tokenizer dependencies
pip install huggingface_hub            # Model download

# Data pipeline dependencies
pip install anthropic                  # Claude API
pip install openai                     # DeepSeek/Qwen API (OpenAI-compatible endpoints)
pip install datasketch                 # MinHash deduplication
pip install sentence-transformers      # Cosine similarity dedup
pip install pandas pyarrow             # Dataset manipulation
pip install jsonschema                 # Schema validation
pip install tqdm rich                  # Progress and display
```

### API keys and access

| Service | Purpose | Environment variable |
|---|---|---|
| Anthropic (Claude Max) | Quality generation, orchestration, reasoning traces | `ANTHROPIC_API_KEY` |
| DeepSeek API | Bulk code generation, mutations, persona rewrites | `DEEPSEEK_API_KEY` |
| Qwen API (Alibaba Cloud) | Alternate bulk generator, diversity | `QWEN_API_KEY` |
| Hugging Face | Model download (Gemma 4 gated access) | `HF_TOKEN` |

Request Gemma 4 access on Hugging Face before anything else. Gated model access can take hours to days. Log into the HF CLI:

```bash
huggingface-cli login --token $HF_TOKEN
```

Store all keys in a `.env` file at the project root (already gitignored). Never commit API keys. Load them at runtime via `python-dotenv` or shell `source .env`.

### Storage estimates

| Item | Size estimate | Notes |
|---|---|---|
| Gemma 4 26B FP16 weights | ~52 GB | Full precision, used only for conversion |
| Gemma 4 26B Q4 (MLX format) | ~13 GB | Primary training format |
| Gemma 4 26B Q8 (MLX format) | ~26 GB | Higher quality option if memory allows |
| Raw generated data (~2M examples) | ~8-15 GB | JSON with code, prompts, metadata |
| Verified clean dataset (~400k examples) | ~2-4 GB | After filtering |
| Training checkpoints (LoRA adapters) | ~200-500 MB per checkpoint | LoRA weights only, not full model |
| Evaluation logs and results | ~500 MB | Benchmark outputs |
| **Total working storage** | **~30-50 GB** | Plus headroom for intermediate files |

Ensure at least 100 GB free on the target volume before beginning. SSD read/write speed matters for training throughput -- the internal SSD on M5 Pro is sufficient.

### Jac compiler setup

The Jac compiler is the single most important tool in the entire pipeline. Every code example must pass through it. Verify the installation works before generating a single example:

```bash
# Install Jac
pip install jaclang

# Verify compiler works
echo 'with entry { print("hello from jac"); }' > /tmp/test.jac
jac run /tmp/test.jac
# Expected output: hello from jac

# Verify compiler catches errors
echo 'with entry { print("missing paren" }' > /tmp/test_bad.jac
jac run /tmp/test_bad.jac 2>&1
# Expected: compiler error output
```

Build a thin Python wrapper around the compiler that accepts a code string, writes it to a temp file, runs `jac run` or `jac check` (syntax-only mode if available), captures stdout/stderr, returns a structured result with pass/fail, error message, and exit code. This wrapper will be called millions of times -- it must be fast, reliable, and never hang. Set a 30-second timeout per compilation.

### MLX installation and verification

```bash
pip install mlx mlx-lm

# Verify MLX sees the GPU
python3 -c "import mlx.core as mx; print(mx.default_device())"
# Expected: Device(gpu, 0)

# Verify MLX can allocate on unified memory
python3 -c "import mlx.core as mx; a = mx.ones((1000, 1000)); mx.eval(a); print('OK')"
```

If `mx.default_device()` returns `cpu` instead of `gpu`, something is wrong with the MLX installation or the system's Metal support. Do not proceed to training until GPU access is confirmed.

### Model download and quantization

```bash
# Download Gemma 4 26B A4B Instruct (requires HF access approval)
huggingface-cli download google/gemma-4-26b-a4b-it --local-dir ~/models/gemma-4-26b-a4b-it

# Convert to MLX format
python -m mlx_lm.convert \
  --hf-path google/gemma-4-26b-a4b-it \
  --mlx-path ~/models/gemma-4-26b-a4b-it-mlx

# Quantize to Q4 (fits comfortably in 48GB with room for training)
python -m mlx_lm.convert \
  --hf-path google/gemma-4-26b-a4b-it \
  --mlx-path ~/models/gemma-4-26b-a4b-it-mlx-q4 \
  --quantize \
  --q-bits 4

# Quantize to Q8 (tighter fit, higher quality)
python -m mlx_lm.convert \
  --hf-path google/gemma-4-26b-a4b-it \
  --mlx-path ~/models/gemma-4-26b-a4b-it-mlx-q8 \
  --quantize \
  --q-bits 8

# Quick inference test
python -m mlx_lm.generate \
  --model ~/models/gemma-4-26b-a4b-it-mlx-q4 \
  --prompt "Write a hello world program in Python" \
  --max-tokens 200
```

Memory estimates for training with LoRA adapters:

| Quantization | Model size | Estimated training memory | Feasibility on 48GB |
|---|---|---|---|
| Q4 (4-bit) | ~13 GB | ~18-25 GB (model + optimizer + activations) | Comfortable, batch size 2-4 possible |
| Q8 (8-bit) | ~26 GB | ~33-40 GB (model + optimizer + activations) | Tight, batch size 1 with gradient accumulation |
| FP16 | ~52 GB | Does not fit | Not feasible for training |

Start with Q4 quantization. It leaves roughly 23-30 GB free for optimizer states, activations, and gradient accumulation -- enough to experiment with batch sizes. Move to Q8 only after confirming Q4 training runs stably and if quality evaluations suggest quantization artifacts are a problem.

### Directory structure

```
DataGeneration/
  dataset/
    raw_output/          # Unvalidated generator output
      code_gen/
      debug/
      explanation/
      conversion/
      trajectory/
      dpo/
      reasoning/
      multi_turn/
    clean_dataset/       # Compiler-verified, judge-passed examples
      sft/
      dpo/
      multi_turn/
    rejected/            # Failed validation, kept for analysis
    review/              # Manual review samples and notes
    releases/            # Frozen dataset versions for training
    logs/                # Generation and validation logs
  models/
    base/                # Downloaded base model weights
    mlx/                 # MLX-converted and quantized models
    checkpoints/         # LoRA adapter checkpoints during training
    merged/              # Final merged model weights
  eval/
    holdout/             # Eval tasks (created BEFORE generation)
    results/             # Benchmark outputs per model version
    ablations/           # Ablation study results
  configs/
    lora/                # LoRA configuration files
    training/            # Training hyperparameter configs
    generation/          # Generator prompt configs and templates
  docs/
    wholestack/          # This document
    newdatagenstrat/     # Data-generation-only strategy
  src/
    data_generation/     # Generation scripts
    validation/          # Compiler wrapper, judge, dedup
    training/            # MLX training scripts and configs
    evaluation/          # Eval harness and benchmarks
```

---

## Phase 1: Context preparation

### Jac grammar extraction and construct cataloging

The grammar is the ground truth for what Jac supports. Parse the Jac grammar file (from the `jaclang` package source or Jac documentation) into a flat construct list. Every construct that appears in the grammar must appear in the training data.

The construct catalog should look like this (abbreviated):

| # | Construct | Category | Example syntax |
|---|---|---|---|
| 1 | `walker` | Core | `walker MyWalker { ... }` |
| 2 | `node` | Core | `node MyNode { ... }` |
| 3 | `edge` | Core | `edge MyEdge { ... }` |
| 4 | `ability` | Core | `can my_ability() -> str { ... }` |
| 5 | `archetype` | Core | `obj MyObj { ... }` |
| 6 | `with entry` | Entry point | `with entry { ... }` |
| 7 | `spawn` | Graph ops | `root ++> MyNode()` |
| 8 | `visit` | Traversal | `visit [-->]` |
| 9 | `disengage` | Traversal | `disengage;` |
| 10 | Type annotations | Type system | `x: int = 5` |
| 11 | `enum` | Types | `enum Color { RED, GREEN, BLUE }` |
| 12 | `test` | Testing | `test "my test" { ... }` |
| ... | ... | ... | ... |

Target 40-80 constructs depending on granularity. Some constructs overlap (e.g., `spawn` and `++>` are different syntactic forms for the same operation) -- list both, because the model must recognize both forms.

### Skills.md and doc corpus preparation

The `skills.md` file (or equivalent Jac MCP guidance) is the idiom reference. It defines what "idiomatic Jac" means -- not just syntactically valid Jac, but Jac that uses the language the way its designers intended. This document is used in three places:

1. **Generator context:** every generation prompt includes relevant sections of `skills.md` so the generator knows what idiomatic Jac looks like.
2. **Idiom judge context:** the LLM judge uses `skills.md` to score whether generated code is truly Jac-idiomatic or just Python-with-Jac-syntax.
3. **Eval rubric:** evaluation benchmarks reference `skills.md` to define what "correct Jac style" means.

Prepare the doc corpus by:
- Extracting the full `skills.md` content
- Extracting the Jac cheatsheet from `jac://docs/cheatsheet`
- Extracting patterns from `jac://guide/patterns`
- Extracting pitfalls from `jac://guide/pitfalls`
- Organizing by topic for selective injection into prompts (not every prompt needs the full corpus)

### Coverage matrix design

The coverage matrix is a 2D grid: constructs (rows) by difficulty bands (columns). It is the master plan for Recipe 1 and the backbone distribution target for the entire dataset.

| Difficulty band | Definition | Target per construct |
|---|---|---|
| Atomic | The construct in isolation, minimal surrounding code | ~100-150 examples |
| Idiomatic | Used the way real Jac would use it, with appropriate context | ~200-250 examples |
| Composed | Combined with 2-3 other constructs in a non-trivial program | ~150-200 examples |

With 40-80 constructs and ~500 examples per construct across all bands, the coverage matrix alone targets 20,000-40,000 unique seed tasks. Each seed task then multiplies through other recipes (DPO pairs, debugging traces, multi-turn conversations, reasoning augmentations) to reach the 60-120k floor from Recipe 1 alone.

Design the matrix in a spreadsheet or structured JSON file. Track generation progress per cell. Flag cells where the generator consistently struggles (these are the constructs where the model most needs training data, so they deserve extra effort, not less).

### Eval holdout set creation BEFORE generation

This is a non-negotiable ordering constraint. The eval set must exist before any generation begins. If the eval set is created after generation, there is no way to guarantee decontamination -- the generator may have produced examples that are semantically identical to eval tasks.

The eval holdout set should contain 50-100 tasks for each of the six target capabilities:

| Capability | Eval tasks | Source |
|---|---|---|
| Code generation (NL -> Jac) | 50-100 | Hand-written or hand-curated from Jac docs |
| Debugging (broken code -> fix) | 50-100 | Hand-crafted bugs in known-good Jac programs |
| Conversion (Python -> Jac) | 50-100 | Real Python programs with known Jac equivalents |
| Explanation (Jac -> NL) | 50-100 | Jac programs with expert-written explanations |
| Agentic (multi-step tasks) | 50-100 | Multi-step scenarios requiring planning |
| Orchestration (sub-agent coordination) | 50-100 | Tasks requiring agent spawning and delegation |

Total: 300-600 eval tasks. These tasks are never used for training, never shown to generators, and never included in any training split. They are the measurement instrument -- contaminating them destroys the ability to measure progress.

Store the eval holdout in `eval/holdout/` with a manifest listing every task ID and its expected outputs. Version the holdout set. If tasks are added or removed later, create a new version and re-run all benchmarks on the new version for comparability.

### Decontamination reference set assembly

Extract from the eval holdout set: (a) the text of every task description, (b) the code of every reference solution, (c) 14-gram shingle sets for both. Store these as the decontamination reference. During generation, every new example is checked against this reference. Any example with high 14-gram overlap (threshold: >50% of shingles shared) is flagged and routed to manual review before inclusion.

The decontamination check is cheap (shingle comparison is O(1) per example with MinHash) and must run on every batch, not just at the end. Catching contamination early prevents wasting generation budget on examples that will be discarded.

---

## Phase 2: Data generation

This phase produces the raw material. The ten recipes below are self-contained generation pipelines that compose multiplicatively. A single seed task from Recipe 1 can yield an SFT example, two DPO pairs via Recipe 3, a debugging trace via Recipe 4, a multi-turn conversation via Recipe 8, a reasoning trace via Recipe 9, and a Python translation pair via Recipe 2. Plan every verified artifact as a seed crystal that grows in multiple directions.

### Generator allocation

| Generator | Budget tier | Best recipes | Rate limit notes |
|---|---|---|---|
| Claude Max (Anthropic) | Quality | R1 orchestration, R4 debugging, R8 multi-turn, R9 reasoning, final refinement | Max plan rate limits; batch carefully |
| DeepSeek API | Cheap bulk | R2 Python generation, R4 mutations, R5 persona rewrites, R6 evolutions | ~$0.14/M input, $0.28/M output tokens |
| Qwen API | Cheap bulk | R2 parallel generation, R5, R6, diversity | Similar pricing to DeepSeek |
| Base Gemma 4 (local) | Free | R3 adversarial negatives (DPO rejected half) | Unlimited, speed limited by M5 Pro inference |
| Finetuned Gemma vN | Free | R7 self-distillation (from v0 onward) | Unlimited, behind compiler gate |
| Cursor / Codex | Limited | Diversity sampling, judge comparison inputs | Budget-limited, use sparingly |

**Mental model:** Claude is for quality and orchestration. Cheap APIs are for volume. Base Gemma is for free negatives. Finetuned Gemma is for the bootstrap. Never burn Claude budget on tasks that cheaper generators can do behind the compiler gate.

### Recipe 1 -- Grammar-walked coverage matrix

**Goal:** guarantee every Jac construct appears in training data at controlled frequency.

**Inputs:** construct catalog (Phase 1), difficulty band definitions, `skills.md` as context.

**Process:**
1. For each cell in the coverage matrix (construct x difficulty band), compose a generation prompt that explicitly names the target construct and difficulty level.
2. Include in the prompt: the construct's grammar definition, 2-3 examples from `skills.md`, the difficulty band definition, and an instruction to generate N diverse tasks using that construct at that difficulty.
3. Generate in batches of 10-20 tasks per cell. Use Claude Max for the first pass to establish quality baselines, then switch to cheaper generators for volume once the prompt is validated.
4. Each generated task includes: natural language description, Jac code solution, expected behavior description, and metadata (construct, difficulty, generator).
5. Every code solution goes through the compiler gate immediately. Track pass rate per cell.

**Prompt template (abbreviated):**
```
You are generating training data for a Jac programming language model.

Target construct: {construct_name}
Difficulty: {difficulty_band}
Construct definition: {grammar_excerpt}
Idiomatic usage examples: {skills_md_excerpt}

Generate {batch_size} diverse Jac programming tasks that use the {construct_name} construct at {difficulty_band} difficulty.

For each task, provide:
- A natural language task description (2-4 sentences)
- Complete, compilable Jac code that solves the task
- A brief explanation of how the construct is used

Requirements:
- Code MUST be valid, compilable Jac -- it will be verified by the Jac compiler
- Use idiomatic Jac patterns, not Python patterns with Jac syntax
- Vary the problem domain across tasks (graph algorithms, data processing, web, utilities)
- {difficulty_specific_instructions}

Output as a JSON array.
```

**Expected yield:** 120-240 cells x 500 examples/cell = 60,000-120,000 SFT examples from coverage alone.

**Failure modes:**
- Generator collapses to a few constructs it knows well. Mitigation: explicit construct naming in every prompt, reject examples that don't use the target construct.
- Atomic examples are too trivial to be useful training data. Mitigation: require atomic examples to still have a meaningful task description, not just "use construct X."
- Composed examples exceed the model's training context window. Mitigation: set a code length limit (e.g., 200 lines) in the prompt.

### Recipe 2 -- Synthetic Python-to-Jac parallel corpus

**Goal:** highest-volume single recipe. Leverages the entire Python programming distribution as a proxy.

**Inputs:** synthetic Python tasks (generated by cheap APIs), `skills.md` idiom mappings.

**Process:**
1. **Step 1: Generate (NL_task, Python_code) pairs.** Use DeepSeek/Qwen APIs to generate diverse Python programming tasks and solutions. These models produce excellent Python. Generate across domains: web APIs, data processing, algorithms, CLI tools, utilities, graph operations.
2. **Step 2: Translate Python to idiomatic Jac.** For each Python solution, generate a Jac translation with an explicit idiom-mapping pass. The prompt must instruct: "Convert this Python code to idiomatic Jac. Do not simply swap syntax -- use Jac-native patterns where appropriate: class to node/archetype, function chains to walker traversals, imperative loops to graph operations." Include relevant `skills.md` sections as context.
3. **Step 3: Dual compilation.** Run both the Python code (via `python3`) and the Jac code (via `jac run`) to verify both are syntactically valid.
4. **Step 4: Behavioral equivalence check.** For deterministic functions, generate shared test inputs and verify both produce the same outputs. This catches translations that compile but have different semantics.

**Output per pipeline run (3 datasets from 1):**
- (NL, Jac_code) -- code generation SFT data
- (Python_code, Jac_code) -- conversion training data
- (Python_code, Jac_code, NL) -- explanation training data (describe the differences)

**Failed translations become DPO negatives.** A Jac translation that compiles but reads like Python with syntax swapped is scored by the idiom judge and paired with the idiomatic version for DPO training.

**Expected yield:** 100,000+ pairs per week at cheap API rates. This is the volume engine.

**Failure modes:**
- Translations that compile but are not idiomatic (the most common failure). Mitigation: idiom judge with `skills.md` scoring.
- Python tasks that have no natural Jac equivalent (e.g., heavy use of Python libraries with no Jac analog). Mitigation: filter Python tasks to domains that map well to Jac's strengths (graph operations, data modeling, traversal).
- Test equivalence failures due to floating point, ordering, or randomness. Mitigation: restrict behavioral checks to deterministic, pure functions.

### Recipe 3 -- Compiler-driven adversarial negatives

**Goal:** teach the model what NOT to write. Critical for a language with strong Python interference.

**Inputs:** verified (task, Jac_code) pairs from any recipe, base Gemma 4 (local, free).

**Process:**
1. For every working Jac example, prompt: "Solve this Jac task the way a Python programmer would, using Python idioms applied to Jac syntax." Use the base (unfinetuned) Gemma 4 running locally -- its outputs ARE the failure mode you are training against, and generation is free.
2. Compile-check the adversarial version:
   - **Doesn't compile:** Perfect. You now have a (broken_code, error_message, correct_code) triple where the broken version is exactly the mistake the finetuned model is most at risk of making. Use for SFT debugging data and as the rejected half of DPO pairs.
   - **Compiles but non-idiomatic:** Confirm with the idiom judge against `skills.md`. Pair with the idiomatic version for DPO. The model learns to prefer Jac-native patterns over syntactically-valid-but-wrong Python patterns.

**Expected yield:** 1 DPO pair per input example. With 100k+ verified examples as input, this produces 40,000-80,000 DPO pairs.

**Failure modes:**
- Base Gemma produces outputs that are accidentally idiomatic (unlikely before finetuning, but possible). Mitigation: run the idiom judge; if the adversarial version scores high, discard the pair.
- Some constructs have no meaningful Python-style alternative. Mitigation: skip constructs that are unique to Jac with no Python analog.

### Recipe 4 -- Bug-synthesis pipeline

**Goal:** first-class debugging data and agentic error-recovery trajectories.

**Inputs:** verified (task, code) pairs, mutation catalog.

**Mutation catalog (systematic, not random):**

| Mutation type | Example | Expected error |
|---|---|---|
| Drop return type | `can foo() -> str` to `can foo()` | Type error or wrong behavior |
| Wrong walker dispatch | Visit wrong node type | Runtime traversal error |
| Mismatched ability signature | Wrong parameter types | Type error |
| Off-by-one in traversal | Skip first/last node | Wrong output |
| Missing edge type declaration | Use undeclared edge | Compiler error |
| Python-style mutation | `list.append()` instead of Jac equivalent | Compiler error or wrong idiom |
| Missing import | Use module without import | Name error |
| Wrong scope | Access variable from wrong scope | Scope error |
| Incorrect type annotation | `str` where `int` needed | Type error |
| Missing entry point | No `with entry` block | No execution |

**Process:**
1. Take a verified (task, code) pair.
2. Apply one mutation from the catalog. The mutation must be plausible -- something a real developer would write by mistake, not a random character flip.
3. Run the mutated code through the compiler. Capture the error message.
4. Record the tuple: (original_task, working_code, mutated_code, error_message, fix_description).
5. Optionally, have a generator (Claude Max) role-play the debugging trajectory: "Looking at error X, I notice Y, which suggests Z. Let me try changing W..." This multi-turn trajectory becomes agentic training data.

**Expected yield:** 30,000-60,000 debugging examples. Each verified example can produce multiple mutations (different error types).

**Failure modes:**
- Mutations that produce unhelpful errors (e.g., cascading errors that obscure the root cause). Mitigation: prefer mutations that produce a single, clear error message.
- Debugging trajectories that are too formulaic. Mitigation: vary the persona and reasoning style in the trajectory generation prompt.

### Recipe 5 -- Persona-stacked task generation

**Goal:** diversify task framing without changing the underlying task content.

**Inputs:** seed tasks from any recipe, persona library.

**Persona library (30-50 personas):**
- Backend engineer migrating Flask to Jac
- Graduate student exploring graph-spatial computation
- Data engineer building ETL pipelines
- Someone who learned Jac yesterday
- Performance engineer optimizing a walker
- Tech lead doing code review
- Security engineer auditing Jac code
- ML engineer building a feature pipeline
- Game developer modeling entity relationships
- DevOps engineer building deployment automation
- Teacher creating Jac course materials
- (... expand to 30-50)

**Process:**
1. For each seed task, select N personas (3-5 per task).
2. Generate the task description from each persona's perspective. Same underlying solution, different surface form, vocabulary, and assumed knowledge level.
3. Some personas (10-20%) should produce intentionally bad task descriptions: ambiguous, missing context, contradictory requirements. The training signal is: "agent asks clarifying question, gets answer, implements." This builds clarification-loop capability.

**Expected yield:** 3-5x multiplier on seed tasks. If 20,000 seeds, then 60,000-100,000 persona-varied examples.

**Failure modes:**
- Persona variation is purely cosmetic (same words, different ordering). Mitigation: score persona diversity with embedding similarity; reject pairs that are too close.
- Bad task descriptions are too obviously broken. Mitigation: the ambiguity should be subtle enough that the model must think about whether to ask or guess.

### Recipe 6 -- Evol-Instruct on Jac axes

**Goal:** increase complexity along controlled, Jac-specific dimensions.

**Five evolution axes:**

| Axis | Operation | Example |
|---|---|---|
| Deepen | Add graph-topology constraint | "Must work on cyclic graphs," "must handle empty traversals" |
| Broaden | Same problem, different domain | Auth flow -> recommendation graph -> state machine |
| Constrain | Add Jac-idiomatic constraint | "Must use walker abilities, no top-level functions" |
| Compose | Chain with another task | "After computing X via walker, persist via Y and expose via Z" |
| Invert | Given solution, generate problem statements | 3 alternative NL descriptions that produce the same code |

**Process:**
1. Take a verified (task, code) pair.
2. Select an evolution axis.
3. Prompt a generator to produce the evolved version: new task description + new code solution.
4. Compile-verify the evolved code.
5. Track lineage: original task ID -> evolution axis -> evolved task ID. This lineage is essential for ablation studies later (which evolution axes contribute most to model quality).

**Expected yield:** 2-3 evolved versions per seed task. Applied to 50,000 seeds = 100,000-150,000 evolved examples (pre-filtering).

**Failure modes:**
- Evolution produces tasks that are too similar to the original (dedup catches this).
- Composed tasks exceed reasonable complexity. Mitigation: cap composition depth at 2-3 chained tasks.

### Recipe 7 -- Self-distillation loop

**Goal:** scale beyond external-model budgets after the first finetuned model exists.

**Process:**
1. Train v0 of finetuned Gemma on the initial dataset (Phases 2-6 without this recipe).
2. Use v0 as a generator: give it tasks, collect outputs.
3. Run every v0 output through the full verification pipeline (compiler, tests, idiom judge).
4. Keep what passes. Mix into the training set alongside frontier-model data.
5. Train v1 on the expanded dataset.
6. Repeat. v1 generates -> verify -> v2 trains on the expanded set.

**Non-negotiables:**
- **Retain frontier-model data each round.** At least 30-50% of the training set must remain frontier-model-generated to prevent quality drift. The self-distilled data supplements; it does not replace.
- **Do not relax verification.** Local models produce subtler errors than frontier models -- errors that look plausible but are wrong. The compiler and judge gates must stay at full strictness.

**Expected yield:** each round adds 10,000-30,000 verified examples at zero API cost. After 2-3 rounds, the model produces data the frontier models wouldn't generate in the same shape -- it has internalized Jac patterns and generates novel compositions.

**Failure modes:**
- Model collapse: the model trains on its own errors and amplifies them. Mitigation: frontier data retention + strict verification.
- Diminishing returns after 2-3 rounds. This is expected. Stop when the marginal quality improvement per round drops below a threshold.

### Recipe 8 -- Multi-turn conversation synthesis

**Goal:** agentic interaction data, not just single-shot completions.

**Inputs:** verified (task, code, tests) artifacts.

**Follow-up catalog:**

| Follow-up type | Example |
|---|---|
| Refinement | "Make this walker handle disconnected subgraphs" |
| Error report | "I get this error when the graph is empty: [error]" |
| Edge case | "What happens when the input graph has cycles?" |
| Optimization | "This is too slow for graphs over 10k nodes" |
| Refactor | "Can you restructure this to use abilities instead of standalone functions?" |
| Explanation | "Why did you use a walker here instead of a recursive function?" |
| Conversion | "I have the Python version -- can you show me the idiomatic Jac?" |

**Process:**
1. Take a verified (task, code) pair.
2. Generate 4-6 follow-up turns sampling from the catalog.
3. For each turn, generate the user's follow-up and the assistant's response.
4. Re-verify the code after every turn that modifies it. If the code breaks, the conversation must include the error and recovery.
5. The result is a multi-turn conversation where every code state is compiler-verified.

**Expected yield:** 8,000-20,000 conversations with ~4-6 turns each = 40,000-100,000 SFT training turns.

**Failure modes:**
- Conversations become repetitive (same follow-up patterns). Mitigation: sample from the catalog randomly, track which follow-up types have been used per seed task.
- Code verification fails mid-conversation and the generator cannot recover. Mitigation: allow 2 retry attempts per turn; if still failing, truncate the conversation at the last verified state.

### Recipe 9 -- Reasoning-trace augmentation

**Goal:** teach the model WHY Jac choices are right, not just what they are.

**Inputs:** verified (task, code) pairs.

**Process:**
1. For each verified pair, prompt (Claude Max preferably -- reasoning quality matters here):
   "Explain the reasoning behind this Jac implementation. Specifically address:
   - Why this Jac construct was chosen over the Python-natural alternative
   - What graph-spatial properties make this the right approach
   - What would go wrong if the Python pattern were used instead
   - Any performance or readability trade-offs"
2. The reasoning trace becomes a structured field in the SFT example. Format: `{instruction, code, reasoning}`.
3. Verify that the reasoning accurately describes the code (no hallucinated constructs, no wrong explanations). This requires a secondary LLM check or manual spot-review.

**Expected yield:** 60,000-120,000 reasoning-augmented examples (overlay on existing examples -- same code, new field).

**Failure modes:**
- Reasoning is generic ("Jac is better because it's designed for this"). Mitigation: require specific construct names and concrete comparisons in the prompt.
- Reasoning contradicts the code. Mitigation: secondary LLM verification pass that checks reasoning-code alignment.

### Recipe 10 -- Doc-grounded lesson synthesis

**Goal:** guarantee every documented feature has training data. Ensure no gap between what Jac supports and what the model knows.

**Inputs:** `skills.md`, Jac cheatsheet, patterns guide, pitfalls guide.

**Process:**
1. For each section of `skills.md`, generate a "lesson pack":
   - 3-5 worked examples illustrating the section's content
   - 5-10 practice problems with solutions
   - 5-10 common-mistake examples paired with corrections
   - 2-3 advanced compositions combining this section with other sections
2. The doc text goes into the generator's context as authoritative reference. The generator must ground its examples in the documentation, not hallucinate features.
3. Each example gets its own test suite, which becomes data for the "write tests for this code" capability.

**Expected yield:** 15,000-30,000 doc-grounded examples (depends on documentation volume).

**Failure modes:**
- Generator ignores the documentation context and produces generic code. Mitigation: require explicit references to documented patterns in the output.
- Documentation is incomplete or has errors. Mitigation: use this recipe as a documentation audit -- if the generator cannot produce valid examples for a section, the documentation may need updating.

### Batch scheduling and sequencing strategy

Not all recipes run in parallel from day one. The optimal sequencing:

| Week | Recipes | Rationale |
|---|---|---|
| Week 1 | R1 (pilot), R2 (pilot), eval holdout finalization | Establish baselines, validate prompts, build verification pipeline |
| Week 2 | R1 (scale), R2 (scale), R3 (starts after R1/R2 have outputs), R10 | Volume generation begins, adversarial pairs need seed examples |
| Week 3 | R4, R5, R6 (all need seed examples from R1/R2) | Multiplicative recipes kick in |
| Week 4 | R8, R9 (need verified examples from all prior recipes) | Multi-turn and reasoning overlays |
| Week 5+ | R7 (requires trained v0), continued scaling of R1-R6 | Self-distillation begins after first training run |

Run cheap-API recipes (R2, R5, R6) continuously in the background. Use Claude Max budget strategically for R1 orchestration, R4 debugging trajectories, R8 conversations, and R9 reasoning traces.

---

## Phase 3: Verification pipeline

Every raw example must pass through the verification pipeline before entering the clean dataset. The pipeline has four stages, applied in order of cost (cheapest first, most expensive last).

### Stage 1: Compiler validation (hard gate)

This is the foundational quality gate. Every code field in every example is run through the Jac compiler. There are no exceptions, no soft passes, no "probably compiles."

**Rules:**
- `code`, `fixed_code`, `jac_code` fields: must compile. Failure = rejection.
- `broken_code` fields (in debugging examples): must NOT compile. If it compiles, the debugging example is invalid.
- Compilation timeout: 30 seconds per example. Anything longer is rejected as potentially infinite-looping.
- Compile with the same Jac version throughout the project. Pin the version in requirements.

**Implementation:**

```python
import subprocess
import tempfile
from pathlib import Path

def verify_compiles(code: str, timeout: int = 30) -> dict:
    """Run Jac compiler on code string. Returns structured result."""
    with tempfile.NamedTemporaryFile(suffix='.jac', mode='w', delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                ['jac', 'check', f.name],
                capture_output=True, text=True, timeout=timeout
            )
            return {
                'compiles': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {'compiles': False, 'error': 'timeout'}
        finally:
            Path(f.name).unlink(missing_ok=True)
```

**Expected throughput:** the compiler should handle ~5,000 examples/hour sustained. For 1.5-2.5M raw candidates, this means 300-500 hours of compiler time. Parallelize across CPU cores (the M5 Pro has 12 performance + 4 efficiency cores). Running 8-12 compiler processes in parallel brings this down to 25-60 hours.

### Stage 2: Test suite execution (deterministic functions)

For examples with deterministic behavior (pure functions, graph traversals with fixed inputs), generate and run a small test suite:

1. Generate 3-5 test cases with known expected outputs.
2. Run the Jac code with each test input.
3. Compare actual vs. expected output.
4. Pass = all tests match. Fail = flag for review (the code may be correct with wrong tests, or vice versa).

This stage is a soft gate: failures route to manual review, not automatic rejection. Test generation itself can have errors.

### Stage 3: Idiom judge (LLM-based scoring)

The idiom judge is an LLM (Claude or a capable open model) that scores code against `skills.md` on a 1-5 scale:

| Score | Meaning |
|---|---|
| 5 | Fully idiomatic Jac; uses graph-spatial patterns, walkers, abilities as intended |
| 4 | Good Jac; minor stylistic issues, could be slightly more idiomatic |
| 3 | Acceptable; compiles and works, but has some Python-isms |
| 2 | Weak; mostly Python patterns with Jac syntax, compiles but misses the point |
| 1 | Not Jac; Python code that happens to compile as Jac, or uses constructs incorrectly |

**Exact rubric (included in judge prompt):**
```
Score 5 if ALL of these hold:
- Uses walkers for traversal instead of recursive functions
- Uses nodes/edges for data modeling instead of plain classes
- Uses abilities instead of standalone functions where appropriate
- Uses Jac-specific control flow (spawn, visit, disengage) correctly
- Type annotations are present and correct
- Code organization follows Jac conventions

Score 4 if most of the above hold with minor gaps.
Score 3 if code works but mixes Python and Jac patterns.
Score 2 if code is predominantly Python-style with Jac syntax.
Score 1 if code fundamentally misuses Jac constructs.
```

**Thresholds:**
- Score >= 4: accepted into clean dataset
- Score 3: routed to manual review for upgrade or rejection
- Score <= 2: rejected (optionally used as DPO negative)

**Cost management:** the idiom judge is an LLM call per example, which is expensive at scale. Run it only on examples that pass the compiler gate. For cheap-API-generated code, batch judge calls to reduce overhead. Consider using a smaller, finetuned judge model after the first round.

### Stage 4: Manual sample review

Human review of 5-10% of examples that pass all automated gates.

**Reviewer checklist:**
- [ ] Code does what the task description says
- [ ] Code is idiomatic Jac (agrees with idiom judge score)
- [ ] Task description is clear and unambiguous
- [ ] Explanation (if present) is accurate
- [ ] Reasoning trace (if present) correctly describes the code
- [ ] No hallucinated Jac features (constructs that don't exist)
- [ ] Code is not trivially similar to another reviewed example
- [ ] Complexity label (simple/medium/hard) is accurate

**Process:** randomly sample 5-10% of each batch after automated verification. Track pass rate per batch, per recipe, per generator. If the manual pass rate for any slice drops below 80%, pause generation for that slice, investigate, and fix the prompt or pipeline before resuming.

### Rejection routing

```
raw_output/
  |
  v
[Compiler gate] ---fail---> rejected/compiler_fail/
  |
  pass
  v
[Test suite] ---fail---> review/test_fail/ (soft gate, manual review)
  |
  pass
  v
[Idiom judge] ---score<=2---> rejected/idiom_fail/
              ---score==3---> review/idiom_borderline/
  |
  score>=4
  v
[Manual sample] ---fail---> rejected/manual_fail/
  |
  pass
  v
clean_dataset/
```

### Expected survival rates

| Stage | Input | Expected pass rate | Output |
|---|---|---|---|
| Compiler gate | ~1.5-2.5M raw | 60-80% | ~1.0-1.8M |
| Test suite | ~1.0-1.8M | 70-90% (of testable subset) | ~0.8-1.5M |
| Idiom judge | ~0.8-1.5M | 40-60% (score >= 4) | ~0.4-0.8M |
| Dedup + decontamination | ~0.4-0.8M | 70-85% | ~0.3-0.6M |
| **Final clean dataset** | | | **~300-500k** |

### Retry logic for borderline failures

Not every failure is terminal. Borderline cases deserve a retry:

- **Compiler fail with minor error (missing semicolon, wrong type annotation):** auto-fix with a targeted prompt and re-compile. Allow 2 retries.
- **Idiom judge score 3 (borderline):** prompt the generator to "make this more idiomatic" with specific `skills.md` guidance. Re-score. Allow 1 retry.
- **Test failure on edge cases:** check if the test is wrong (common). If the test is wrong, regenerate the test. If the code is wrong, flag for manual review.

Never retry more than twice. Diminishing returns kick in fast, and retrying bad outputs is more expensive than generating new ones.

---

## Phase 4: Quality controls

### Decontamination against eval holdout

Run decontamination on every batch, not just at the end. The check:

1. Extract 14-gram shingle sets from the new example's task description and code.
2. Compare against the eval holdout's shingle sets using MinHash (Jaccard similarity approximation).
3. If similarity > 0.5 (50% of shingles shared) with any eval task: flag the example.
4. Flagged examples go to manual review. If they are genuinely similar to an eval task, reject them. If the similarity is coincidental (common substrings in Jac syntax), keep them.

**Implementation:**

```python
from datasketch import MinHash, MinHashLSH

def build_shingle_set(text: str, n: int = 14) -> set:
    """Extract character n-gram shingles from text."""
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def check_contamination(example_text: str, eval_lsh: MinHashLSH, threshold: float = 0.5) -> bool:
    """Returns True if example is potentially contaminated."""
    m = MinHash(num_perm=128)
    for shingle in build_shingle_set(example_text):
        m.update(shingle.encode('utf8'))
    results = eval_lsh.query(m)
    return len(results) > 0
```

### Distribution monitoring dashboard

Maintain a live view of the dataset's shape. Check after every batch:

| Metric | Target | Action if drifting |
|---|---|---|
| Construct frequency | Matches coverage matrix targets | Increase generation for underrepresented constructs |
| Persona distribution | Roughly uniform across 30-50 personas | Add underrepresented personas to next batch |
| Difficulty band distribution | 30% atomic, 40% idiomatic, 30% composed | Adjust difficulty parameter in prompts |
| Generator distribution | Per allocation table | Rebalance API budgets |
| Trigram entropy | > 0.8 (normalized) | Generator is in a rut; change prompts, add diversity instructions |
| Recipe source distribution | Per volume targets table | Scale up underrepresented recipes |

The dashboard does not need to be fancy. A Python script that reads the metadata of all clean examples, computes these metrics, and prints a summary table is sufficient. Run it after every batch and before every training run.

### Two-stage deduplication

**Stage 1: Code-level MinHash dedup.** Tokenize each code example, compute a MinHash signature, and group examples with Jaccard similarity > 0.85 into clusters. Keep one example per cluster (the one with the highest idiom judge score). This is fast and catches the most duplicates.

**Stage 2: Prose cosine similarity dedup.** For surviving examples, embed the task descriptions using a sentence transformer model (e.g., `all-MiniLM-L6-v2`). Compute pairwise cosine similarity within each category. Flag pairs with similarity > 0.92. Manually review flagged pairs -- some are legitimately similar tasks with different solutions (keep both), others are duplicates with cosmetic variation (keep one).

**Expected dedup rate:** 15-30% of examples that pass verification are near-duplicates. Generators repeat themselves more than expected.

### Category balance enforcement

After deduplication, check that the dataset matches the target distribution:

| Subset | Target proportion | Tolerance |
|---|---|---|
| Core code generation SFT | 40-50% | +/- 5% |
| Python-Jac conversion pairs | 20-25% | +/- 5% |
| Debugging (broken -> fix) | 8-12% | +/- 3% |
| Multi-turn conversations | 5-8% | +/- 2% |
| Reasoning-augmented | 15-20% (overlay) | +/- 5% |
| DPO preference pairs | 10-15% | +/- 3% |
| Explanation (code -> NL) | 5-8% | +/- 2% |

If any category is overrepresented, downsample randomly. If underrepresented, generate more (go back to Phase 2 for that category).

### Batch-level quality regression checks

Before adding a new batch to the clean dataset, compare its quality metrics against the rolling average:

- Compiler pass rate: must be within 10% of the rolling average for that recipe
- Idiom judge score distribution: must not shift more than 0.5 points mean
- Dedup rate against existing dataset: a batch with >50% duplicates suggests the generator is exhausted for that prompt configuration

If a batch fails regression checks, investigate before adding it. Common causes: prompt degradation (API provider changed model version), generator fatigue (prompt is stale), or dataset saturation (the construct/difficulty space is fully covered).

---

## Phase 5: Dataset assembly

### Format conversion for training objectives

The clean dataset must be converted into the exact formats that MLX training scripts expect. Different training objectives require different formats.

#### SFT format (instruction-following)

MLX's `mlx_lm.lora` expects JSONL with a messages format compatible with the model's chat template:

```jsonl
{"messages": [{"role": "user", "content": "Write a Jac walker that traverses a binary tree and sums all node values."}, {"role": "assistant", "content": "```jac\nwalker SumWalker {\n    has total: int = 0;\n    \n    can visit_node with Node entry {\n        self.total += here.value;\n        visit [-->];\n    }\n}\n```\n\nThis walker uses..."}]}
```

For reasoning-augmented examples, include the reasoning in the assistant response:

```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "<reasoning>\nI'm using a walker rather than a recursive function because...\n</reasoning>\n\n```jac\n...\n```"}]}
```

#### DPO format (preference pairs)

```jsonl
{"prompt": "Convert this Python code to idiomatic Jac: ...", "chosen": "```jac\nwalker ...\n```", "rejected": "```jac\ndef ...  # Python-style\n```"}
```

#### Multi-turn format (conversation arrays)

```jsonl
{"messages": [{"role": "user", "content": "Build a task queue with priority ordering in Jac."}, {"role": "assistant", "content": "Here's a priority queue using nodes and edges..."}, {"role": "user", "content": "Can you make it handle duplicate priorities?"}, {"role": "assistant", "content": "Sure, I'll add a timestamp tiebreaker..."}]}
```

### Train/val/test split strategy

Split the clean dataset into train (90%), validation (5%), and test (5%) sets. The split must be stratified by:

- **Category:** each split has the same proportion of code_gen, debug, explanation, conversion, trajectory examples.
- **Difficulty:** each split has the same proportion of simple, medium, hard examples.
- **Recipe source:** each split has the same proportion of examples from each recipe. This prevents the validation set from being dominated by one recipe.

**Critical:** the test split here is NOT the eval holdout from Phase 1. The eval holdout is a completely separate set created before generation. The test split is for monitoring training dynamics (overfitting detection). The eval holdout is for final model evaluation.

Implementation:

```python
from sklearn.model_selection import StratifiedShuffleSplit

# Create stratification key combining category, difficulty, and recipe
df['strat_key'] = df['category'] + '_' + df['difficulty'] + '_' + df['recipe']

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
train_idx, temp_idx = next(splitter.split(df, df['strat_key']))

# Split temp into val and test
temp_df = df.iloc[temp_idx]
splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_idx, test_idx = next(splitter2.split(temp_df, temp_df['strat_key']))
```

### Metadata and provenance per example

Every example in the final dataset carries full provenance:

```json
{
  "id": "sft-code_gen-20260601-003-0042",
  "recipe": "R2",
  "generator": "deepseek-v3",
  "seed_task_id": "coverage-walker-idiomatic-017",
  "evolution_path": ["R1:seed", "R6:deepen", "R9:reasoning"],
  "verification_levels": {
    "compiler": true,
    "test_suite": true,
    "idiom_judge_score": 4.5,
    "manual_reviewed": false
  },
  "context_bundle_version": "jac-context-v2",
  "source_prompt_version": "prompt-code_gen-v3",
  "generation_date": "2026-06-01",
  "dataset_version": "jac-synth-v1.0.0"
}
```

### Version manifest

Each dataset release includes a manifest file:

```json
{
  "dataset_version": "jac-synth-v1.0.0",
  "release_date": "2026-06-15",
  "total_examples": 387420,
  "splits": {
    "train": 348678,
    "val": 19371,
    "test": 19371
  },
  "by_category": {
    "code_gen_sft": 178000,
    "conversion": 89000,
    "debug": 42000,
    "multi_turn": 12000,
    "reasoning": 85000,
    "dpo": 58000,
    "explanation": 28000
  },
  "by_difficulty": {
    "simple": 110000,
    "medium": 170000,
    "hard": 107420
  },
  "by_recipe": { "R1": 95000, "R2": 112000, "...": "..." },
  "prompt_versions": ["prompt-code_gen-v3", "prompt-debug-v2", "..."],
  "context_bundle_version": "jac-context-v2",
  "jac_compiler_version": "0.7.2",
  "checksums": {
    "train.jsonl": "sha256:abc123...",
    "val.jsonl": "sha256:def456...",
    "test.jsonl": "sha256:ghi789..."
  }
}
```

---

## Phase 6: Finetuning on Mac M5 Pro

This is the phase where hardware constraints dominate every decision. The Mac M5 Pro with 48GB unified RAM is capable of finetuning a 26B MoE model, but only with the right framework, quantization, and configuration.

### Framework: MLX + mlx-lm

**Why MLX, not PyTorch/CUDA:** MLX is Apple's machine learning framework designed for Apple Silicon. It operates on unified memory -- the same physical memory serves both CPU and GPU. There is no data transfer bottleneck between CPU and GPU RAM because they share the same address space. PyTorch with MPS (Metal Performance Shaders) backend is an alternative, but MLX is purpose-built for this hardware and has native LoRA support through `mlx-lm`.

**Key advantages for this project:**
- No CUDA required (Apple Silicon has no CUDA)
- Unified memory means the full 48GB is available to the model (no separate CPU/GPU pools)
- Lazy evaluation -- MLX only computes what is needed, reducing peak memory
- Native quantization support (Q4, Q8) with minimal quality loss
- LoRA training built into `mlx-lm` with a single command

**Installation:**

```bash
pip install mlx mlx-lm

# Verify GPU access
python3 -c "
import mlx.core as mx
print(f'Device: {mx.default_device()}')
print(f'Metal available: {mx.metal.is_available()}')
"
# Expected: Device: Device(gpu, 0), Metal available: True
```

### Model preparation

**Step 1: Download from HuggingFace**

```bash
# Gemma 4 26B A4B Instruct (gated -- requires approval)
huggingface-cli download google/gemma-4-26b-a4b-it \
  --local-dir ~/models/gemma-4-26b-a4b-it
```

**Step 2: Convert to MLX format**

```bash
python -m mlx_lm.convert \
  --hf-path google/gemma-4-26b-a4b-it \
  --mlx-path ~/models/gemma-4-26b-a4b-it-mlx
```

**Step 3: Quantize**

```bash
# Q4 -- recommended starting point
python -m mlx_lm.convert \
  --hf-path google/gemma-4-26b-a4b-it \
  --mlx-path ~/models/gemma-4-26b-a4b-it-mlx-q4 \
  --quantize \
  --q-bits 4

# Q8 -- higher quality, tighter memory
python -m mlx_lm.convert \
  --hf-path google/gemma-4-26b-a4b-it \
  --mlx-path ~/models/gemma-4-26b-a4b-it-mlx-q8 \
  --quantize \
  --q-bits 8
```

**Memory estimates after quantization:**

| Config | Model weights | Optimizer states | Activations (est.) | Total (est.) | Batch size |
|---|---|---|---|---|---|
| Q4 + LoRA rank 32 | ~13 GB | ~1 GB | ~4-8 GB | ~18-22 GB | 2-4 |
| Q4 + LoRA rank 64 | ~13 GB | ~2 GB | ~4-8 GB | ~19-23 GB | 2-3 |
| Q8 + LoRA rank 32 | ~26 GB | ~1 GB | ~4-8 GB | ~31-35 GB | 1-2 |
| Q8 + LoRA rank 64 | ~26 GB | ~2 GB | ~4-8 GB | ~32-36 GB | 1 |

**Recommendation:** start with Q4 + LoRA rank 32. This leaves ~25 GB of headroom for activations and gradient accumulation, allowing batch size 2-4 with gradient accumulation of 8-16 to achieve an effective batch size of 16-64.

### LoRA configuration

```yaml
# configs/lora/default.yaml
adapter_type: lora
lora_parameters:
  rank: 32                    # Start here, increase to 64 if underfitting
  alpha: 64                   # 2x rank is a common starting point
  dropout: 0.05               # Light regularization
  scale: 1.0                  # alpha / rank

# Target all attention + MLP projection layers
lora_layers:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# Training hyperparameters
learning_rate: 5e-5           # Conservative start
lr_schedule: cosine
warmup_steps: 200             # ~10% of first stage
weight_decay: 0.01
max_seq_length: 4096          # Gemma 4 supports up to 128k, but limit for memory
batch_size: 2                 # Per-device, limited by memory
gradient_accumulation_steps: 16  # Effective batch = 2 * 16 = 32
num_epochs: 3                 # SFT stages
save_every: 500               # Checkpoint frequency in steps
eval_every: 250               # Validation frequency
```

**Why these target modules:** in MoE models, the expert layers (gate_proj, up_proj, down_proj) are where specialized knowledge lives. Training LoRA adapters on these layers teaches the model Jac-specific patterns without disturbing the general routing mechanism. The attention layers (q/k/v/o_proj) allow the model to attend to Jac syntax patterns differently. Together, they cover the full transformation pipeline.

### Multi-stage training strategy

Training proceeds in four stages, each building on the previous. Adapters from each stage are merged before the next stage begins (or stacked if mlx-lm supports adapter composition).

#### Stage 1: Core SFT (code generation + conversion + explanation)

- **Data:** the largest, most diverse subset -- core code generation SFT, Python-Jac conversion pairs, and explanation examples. ~200,000-350,000 examples.
- **Purpose:** teach the model Jac syntax, semantics, and idioms. This is the foundation.
- **Epochs:** 2-3
- **Learning rate:** 5e-5 with cosine schedule and 10% warmup
- **Expected duration on M5 Pro:** estimating ~1-2 seconds per training step with batch size 2 and 4096 max sequence length. With 200k examples, batch size 2, gradient accumulation 16: ~6,250 steps/epoch x 3 epochs = ~18,750 steps. At 1-2 sec/step = ~5-10 hours per epoch, ~15-30 hours total for stage 1.

```bash
python -m mlx_lm.lora \
  --model ~/models/gemma-4-26b-a4b-it-mlx-q4 \
  --train \
  --data ~/datasets/jac-synth-v1/sft_core/ \
  --adapter-path ~/models/checkpoints/stage1/ \
  --batch-size 2 \
  --lora-rank 32 \
  --lora-layers 16 \
  --num-epochs 3 \
  --learning-rate 5e-5 \
  --steps-per-eval 250 \
  --save-every 500 \
  --max-seq-length 4096 \
  --grad-checkpoint
```

#### Stage 2: Specialized SFT (debugging + reasoning traces)

- **Data:** debugging examples (broken -> fix) and reasoning-augmented examples. ~80,000-150,000 examples.
- **Purpose:** specialized capabilities that build on the foundation from stage 1. Debugging requires understanding what can go wrong; reasoning requires understanding why Jac patterns are right.
- **Epochs:** 2-3
- **Learning rate:** 3e-5 (slightly lower than stage 1 to avoid catastrophic forgetting)
- **Start from:** stage 1 adapter

#### Stage 3: DPO (preference alignment)

- **Data:** DPO preference pairs -- idiomatic Jac (chosen) vs. Python-style Jac (rejected). ~40,000-80,000 pairs.
- **Purpose:** teach the model to discriminate between syntactically valid but non-idiomatic code and truly idiomatic Jac. This is the alignment step that prevents the model from falling back to Python patterns.
- **Epochs:** 1-2 (DPO is more sensitive to overfitting)
- **Learning rate:** 1e-5 (lower for alignment)
- **DPO beta:** 0.1 (controls how strongly preferences are enforced)
- **Start from:** stage 2 adapter

Note: as of this writing, mlx-lm may not natively support DPO training. If not, options include:
1. Implement DPO loss in a custom MLX training script (the math is straightforward: log-probability ratio between chosen and rejected, with a KL penalty).
2. Use the TRL library with PyTorch + MPS backend for the DPO stage only (slower but proven).
3. Convert adapters to PyTorch for DPO training on a cloud A100, then convert back to MLX.

#### Stage 4: Multi-turn conversation fine-tuning

- **Data:** multi-turn conversations with 4-6 turns each. ~8,000-20,000 conversations.
- **Purpose:** agentic capabilities -- the model must handle follow-up questions, error recovery, and iterative refinement in a conversation context.
- **Epochs:** 2-3
- **Learning rate:** 2e-5
- **Max sequence length:** 8192 (conversations are longer than single-turn examples)
- **Start from:** stage 3 adapter (or stage 2 if DPO is done separately)

### Training monitoring

**Loss curves per stage:** plot training loss and validation loss per step. Watch for:
- Training loss not decreasing: learning rate too low, data quality issue, or model already knows the material.
- Validation loss increasing while training loss decreases: overfitting. Reduce epochs, increase dropout, or reduce learning rate.
- Loss spikes: data quality issue (corrupted example, extremely long sequence causing OOM).

**Gradient norms:** monitor per-step gradient norms. Spikes indicate problematic examples. Consider gradient clipping (max norm 1.0) if spikes are common.

**Checkpointing:** save adapter weights every 500 steps. Training crashes on Mac lose everything since the last checkpoint. With 48GB unified memory, an unexpected memory pressure spike from another process can crash the training job.

**Validation loss for early stopping:** evaluate on the validation split every 250 steps. If validation loss has not improved for 3 consecutive evaluations (750 steps), consider stopping early. This is especially important for DPO (stage 3) where overfitting is fast.

**Memory monitoring:** open Activity Monitor during training and watch:
- Memory Pressure gauge: should stay green/yellow. Red = system is swapping, training will slow catastrophically.
- Memory Used vs. Physical Memory: track how much headroom exists.
- GPU Utilization (under GPU History): should be consistently high (>80%). Low GPU utilization means the data loading pipeline is the bottleneck.

### Practical tips for Mac training

1. **Close other apps.** Safari, Slack, VS Code, Docker -- everything competes for unified memory. The 48GB is shared between everything running on the machine. A web browser with many tabs can easily consume 4-8 GB that the training job needs.

2. **Use Q4 quantization to maximize batch size.** The quality difference between Q4 and Q8 for LoRA training is usually small because the LoRA adapters themselves are full precision. The quantized base model provides the forward pass, but the learning happens in the adapter weights.

3. **Expect slower training than an A100.** Rough comparison: an A100-80GB with PyTorch + flash attention trains at ~3-5x the speed of an M5 Pro with MLX for the same model and batch size. Plan training schedules accordingly. A stage that would take 4 hours on an A100 may take 12-20 hours on M5 Pro.

4. **MLX supports gradient checkpointing.** Use `--grad-checkpoint` flag to trade compute for memory. This recomputes activations during the backward pass instead of storing them, reducing memory by ~30-40% at the cost of ~20-30% more compute time. Essential for Q8 training or larger batch sizes.

5. **Save checkpoints frequently.** Set `--save-every 500` or even `--save-every 250`. Training crashes lose everything since the last checkpoint. LoRA adapter checkpoints are small (~200-500 MB) so disk space is not a concern.

6. **Monitor thermals.** Extended training runs heat up the M5 Pro. If the machine thermal-throttles, training speed drops. Ensure good ventilation. Consider an external cooling solution for multi-day training runs. The M5 Pro's thermal design is good but not designed for 24+ hour sustained GPU load.

7. **Use `caffeinate` to prevent sleep.**
   ```bash
   caffeinate -dims python -m mlx_lm.lora --train ...
   ```
   This prevents the Mac from sleeping, dimming the display, or spinning down the disk during training.

8. **Test with a tiny dataset first.** Before launching a multi-day training run, test the entire pipeline with 100 examples for 10 steps. Verify that loss decreases, checkpoints save correctly, validation runs, and memory usage is stable.

---

## Phase 7: Evaluation

Evaluation answers one question: did the finetuning work, and by how much? Every evaluation metric is computed on the Phase 1 eval holdout set (never seen during training) and optionally on the test split of the training data (for sanity checks).

### Automated benchmarks

#### Compiler pass rate

The most important single metric. Generate Jac code for every eval task and run it through the compiler.

- **Target:** >90% compiler pass rate on held-out tasks (base Gemma 4 likely scores <30% on Jac-specific tasks before finetuning).
- **Methodology:** prompt the model with each eval task, generate code (temperature 0 for deterministic evaluation, or temperature 0.6 with pass@k for stochastic evaluation), compile.
- **Reporting:** overall pass rate, pass rate per construct category, pass rate per difficulty band.

```bash
# Generate responses for eval set
python -m mlx_lm.generate \
  --model ~/models/gemma-4-26b-a4b-it-mlx-q4 \
  --adapter-path ~/models/checkpoints/stage4-final/ \
  --prompt "$(cat eval/holdout/task_001.txt)" \
  --max-tokens 2048 \
  --temp 0.0
```

#### Test pass rate

For eval tasks with deterministic expected outputs, run the compiled code and check outputs.

- **Target:** >80% test pass rate on deterministic tasks.
- **Methodology:** compile the generated code, run it with test inputs, compare outputs to expected values.

#### Construct coverage

Does the model use all Jac features, or does it collapse to a subset?

- **Analysis:** for each generated eval response, parse the code and count which Jac constructs are used. Compare against the coverage matrix targets.
- **Warning sign:** if the model uses walkers for everything but never uses edges, or uses abilities but never uses archetypes, the training data distribution was skewed.

#### Syntax validity rate

Broader than compiler pass rate -- includes partial correctness. Even if the full program doesn't compile, how much of the syntax is valid?

- **Methodology:** parse the generated code line-by-line and count syntactically valid vs. invalid lines.
- **Use case:** understanding whether failures are localized (one wrong line) or systemic (the model doesn't understand Jac syntax).

### Judge-based evaluation

#### Idiom quality judge

Same LLM judge from Phase 3, but applied to eval outputs:

- Score every eval response on the 1-5 idiom scale.
- **Target:** mean score >= 4.0 across all eval tasks.
- **Comparison:** score base model outputs on the same tasks. The improvement from base to finetuned should be >= 1.5 points mean.

#### Reasoning quality judge

For explanation and reasoning tasks in the eval set:

- Score on accuracy (does the explanation match the code?), specificity (does it mention specific constructs and why?), and insight (does it explain why Jac patterns are preferred over Python patterns?).
- Use a separate rubric from the idiom judge.

#### Code quality judge

For all code generation tasks:

- Readability: is the code well-structured, properly indented, with meaningful variable names?
- Efficiency: does it use appropriate data structures and algorithms?
- Pattern adherence: does it use graph/walker patterns where appropriate?

### Capability-specific benchmarks

Design 50-100 tasks per capability, held out from training:

#### Code generation benchmark (NL -> Jac)

- 20 simple tasks (single construct, short code)
- 40 medium tasks (multiple constructs, moderate complexity)
- 40 hard tasks (composed constructs, graph algorithms, full programs)
- Metric: compiler pass rate, test pass rate, idiom judge score

#### Debugging benchmark

- 50 tasks: broken Jac code + error message -> fix
- Vary error types: syntax errors, type errors, wrong walker dispatch, scope issues, Python-isms
- Metric: fix success rate (does the fixed code compile and pass tests?), fix accuracy (is the fix minimal and correct, not a rewrite?)

#### Conversion benchmark (Python -> Jac)

- 50 tasks: Python code -> idiomatic Jac
- Critical metric: idiom score, not just compilation. A syntactic translation that compiles is not enough -- it must use Jac patterns.
- Include tasks where the "obvious" translation is wrong (e.g., Python class should become a Jac node, not a Jac object).

#### Explanation benchmark

- 50 tasks: Jac code -> natural language explanation
- Evaluated by LLM judge on accuracy, completeness, and Jac-specific insight.
- Compare against expert-written reference explanations using ROUGE or BERTScore for surface-level similarity, plus LLM judge for semantic accuracy.

#### Agentic benchmark

- 50 multi-step tasks requiring planning, tool use, and error recovery.
- Format: provide a task, let the model generate a plan and implementation step by step.
- Metric: task completion rate, number of steps to completion, quality of intermediate reasoning.

#### Orchestration benchmark

- 50 tasks requiring sub-agent coordination.
- Format: tasks that are too complex for a single pass and require decomposition.
- Metric: decomposition quality, sub-task correctness, final assembly correctness.

### Base model comparison

Run every benchmark on both the base Gemma 4 26B A4B (before finetuning) and the finetuned model. Report:

| Metric | Base model | Finetuned | Delta |
|---|---|---|---|
| Compiler pass rate | est. 20-30% | target >90% | +60-70% |
| Test pass rate | est. 10-20% | target >80% | +60-70% |
| Idiom judge mean | est. 1.5-2.0 | target >4.0 | +2.0-2.5 |
| Code gen pass@1 | est. 15% | target >85% | +70% |
| Debugging fix rate | est. 10% | target >70% | +60% |
| Conversion idiom score | est. 1.5 | target >3.5 | +2.0 |

**Regression checks:** finetuning on Jac should not hurt the model's general coding ability (Python, JavaScript, etc.) or its reasoning capabilities. Run a subset of standard coding benchmarks (e.g., HumanEval in Python) before and after finetuning. If Python performance drops by more than 5%, the LoRA rank or learning rate may be too aggressive.

### Ablation studies

Ablation studies answer: "which parts of the pipeline mattered?" They are expensive (each requires a separate training run) but essential for optimizing the pipeline for v2.

#### Contribution per recipe

Train the model without each recipe (one at a time) and measure the impact:

| Ablation | Expected impact |
|---|---|
| Without R1 (coverage matrix) | Construct coverage drops, rare constructs missing |
| Without R2 (Python-Jac pairs) | Volume drops dramatically, conversion capability disappears |
| Without R3 (adversarial negatives) | Idiom quality drops, more Python-isms in output |
| Without R4 (bug synthesis) | Debugging capability drops |
| Without R9 (reasoning traces) | Explanation quality drops, less "why" in outputs |
| Without R8 (multi-turn) | Agentic capability drops, can't handle follow-ups |

#### Effect of DPO stage

Train with and without stage 3 (DPO). Measure idiom judge scores. DPO should improve discrimination between idiomatic and non-idiomatic Jac. If it doesn't, the DPO data quality may be insufficient.

#### Effect of reasoning traces

Train with and without reasoning-augmented examples. Measure explanation quality and code quality. Research suggests reasoning traces can substitute for model scale -- this ablation tests whether that holds for Jac.

#### Data volume sensitivity (learning curves)

Train the model at different data volumes and plot performance vs. volume:

| Training examples | Expected behavior |
|---|---|
| 10k | Basic syntax, limited constructs |
| 50k | Most constructs, moderate idiom quality |
| 100k | Good coverage, good idiom quality |
| 300k | Near-target quality |
| 500k | Diminishing returns expected |

This curve reveals whether more data or better data is the bottleneck. If the curve flattens before the target quality, the data quality or diversity is the issue, not the volume.

### Statistical rigor

- **Multiple training runs:** run 2-3 independent training runs (different random seeds) for the final configuration. Report mean and standard deviation of all metrics.
- **Confidence intervals:** for compiler pass rate and test pass rate, report 95% confidence intervals (Wilson score interval for proportions).
- **Pass@k for code generation:** report pass@1, pass@5, and pass@10 for code generation tasks. pass@k measures the probability that at least one of k generated samples passes. Use the unbiased estimator:
  
  pass@k = 1 - C(n-c, k) / C(n, k)
  
  where n = total samples, c = correct samples.

- **Significance testing:** when comparing base vs. finetuned or ablation vs. full, use McNemar's test (paired binary outcomes) or bootstrap confidence intervals on the delta.

---

## Open questions

1. **Final construct catalog and frequency weights.** The exact list of 40-80 Jac constructs and their target frequencies in the coverage matrix drives Recipe 1 and the distribution monitoring. This requires input from the Jac language team to determine which constructs are most important in practice.

2. **Idiom judge prompt engineering.** The idiom judge is the single most-used LLM judge in the pipeline. Its rubric determines what "good Jac" means for the entire dataset. This prompt needs heavy iteration and calibration against human judgments before scaling.

3. **MLX DPO support.** As of this writing, `mlx-lm` may not natively support DPO training. If not, the fallback options (custom MLX script, PyTorch MPS, cloud GPU for DPO only) need to be evaluated for feasibility and quality trade-offs.

4. **Jac compiler throughput at scale.** The verification pipeline needs to handle ~5,000 examples/hour sustained across 1.5-2.5M raw candidates. Benchmark the compiler on the M5 Pro to confirm this is achievable with parallel processes, and identify the parallelism ceiling.

5. **Self-distillation quality floor.** Recipe 7 (self-distillation) assumes that the finetuned model's outputs are good enough to train on after verification. At what idiom judge score does the v0 model need to be before self-distillation is worthwhile? If v0 scores below 3.0 mean, self-distilled data may be too noisy even with compiler verification.

6. **Optimal LoRA rank for MoE models.** The MoE architecture (3.8B active params from 26B total) may respond differently to LoRA rank than dense models. The expert layers may need higher rank than attention layers. Run a small-scale rank sweep (16, 32, 64, 128) on a 10k subset before committing to the full training budget.

7. **Training duration vs. quality trade-off on M5 Pro.** Multi-day training runs are feasible but risky (crashes, thermal throttling). Should training be split into shorter runs with checkpoint resumption? How much quality is lost to Q4 quantization during training -- is Q8 worth the slower training speed?

8. **Eval holdout contamination monitoring.** The decontamination check uses 14-gram overlap, but semantic contamination (same task described differently) is harder to catch. Should the decontamination also include embedding-based similarity, and what threshold prevents false positives from blocking legitimate training examples?
