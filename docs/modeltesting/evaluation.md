# Model Comparison Evaluation Methodology

*Systematic evaluation framework for comparing Gemma 4 26B A4B and Qwen3-Coder-30B-A3B after finetuning on 5,000 Jac examples.*

---

## Evaluation design principles

The evaluation must answer one question: which of the two candidate models learns Jac best from the same synthetic training data? To answer this question reliably, the evaluation framework enforces three principles:

**Controlled comparison.** Every variable except the base model is held constant. Same training data, same LoRA rank, same learning rate, same number of training steps, same evaluation tasks, same evaluation prompts, same sampling parameters. The only difference is the model architecture and pre-trained weights. This isolates the effect of the base model on Jac learning.

**Multi-dimensional assessment.** A single metric (e.g., compiler pass rate) is insufficient. A model might compile 90% of its outputs but produce Python-with-Jac-syntax rather than idiomatic Jac. Another model might compile only 70% but produce genuinely idiomatic code when it succeeds. The evaluation captures multiple dimensions of capability and weights them according to their importance for the project's goals.

**Statistical rigor.** With a relatively small evaluation set (50-100 tasks per capability area), individual task results can be noisy. The evaluation accounts for this by running each model multiple times (2-3 runs with different random seeds), computing confidence intervals, and requiring statistically meaningful differences before declaring a winner.

**Conversion-probe subset.** The pre-step conversion probe ([`conversion_probe.md`](conversion_probe.md)) reuses this framework but restricted to **Area 3 (conversion)** with a reduced metric set: cross-compiled test pass rate is the primary objective metric, supported by compiler pass rate, construct diversity, and idiom adherence. Metric definitions below are unchanged; the probe simply applies them to the conversion holdout.

---

## Benchmark suite design

The evaluation benchmark consists of five capability areas, each with 50-100 tasks. The total evaluation set is 350-500 tasks, held entirely separate from the 5,000 training examples. No evaluation task appears in the training data, and no training example has high n-gram overlap with any evaluation task (decontamination is performed before training).

### Area 1: Code generation (NL to Jac)

**100 tasks** across three difficulty levels.

| Difficulty | Count | Description | Example |
|---|---|---|---|
| Atomic | 35 | Single Jac construct in isolation | "Define a node `Person` with a `name` string and `age` integer." |
| Idiomatic | 40 | Construct used in natural Jac context | "Create a walker that traverses a social graph and collects all friends-of-friends." |
| Composed | 25 | Multiple constructs in a non-trivial program | "Build a recommendation engine using walkers to traverse a user-item graph, scoring edges by weight and returning the top-5 unvisited items." |

Each task includes:
- Natural language description (the prompt)
- Reference Jac solution (compiler-verified, test-verified, idiom-judge-verified)
- Test suite (3-5 test cases with expected outputs)
- Construct coverage annotation (which Jac constructs the task exercises)
- Difficulty rating (atomic, idiomatic, composed)

The 100 tasks span the full construct coverage matrix: nodes, edges, walkers, abilities, archetypes, type annotations, control flow, graph operations, dispatch, and module organization. No major Jac construct is absent from the evaluation set.

### Area 2: Debugging (broken Jac + error to fix)

**80 tasks** with systematic bug categories.

| Bug category | Count | Description |
|---|---|---|
| Syntax errors | 15 | Missing semicolons, wrong delimiters, malformed declarations |
| Type errors | 15 | Wrong type annotations, mismatched types, missing type declarations |
| Semantic errors | 20 | Wrong walker dispatch, incorrect traversal, misused abilities |
| Python-pattern errors | 20 | Python idioms applied to Jac that compile but produce wrong behavior |
| Multi-construct errors | 10 | Bugs that span multiple interacting constructs |

Each task includes:
- The broken Jac code
- The compiler error message (or runtime error for semantic bugs)
- A description of the intended behavior
- The correct fix (reference solution)
- The minimal diff between broken and fixed code

The model receives the broken code and error message and must produce the fixed code. Evaluation checks both that the fix compiles and that it produces the correct behavior (matches the reference solution's test output).

### Area 3: Conversion (Python to idiomatic Jac)

**80 tasks** testing the model's ability to translate Python to Jac.

The critical distinction: conversion means translating to idiomatic Jac, not just swapping syntax. A correct conversion replaces Python classes with nodes/archetypes where appropriate, replaces function chains with walker traversals where the problem domain is graph-spatial, and uses Jac-specific features (abilities, dispatch, has-variables) where they are idiomatic.

| Conversion type | Count | Description |
|---|---|---|
| Direct mapping | 20 | Python constructs with clear Jac equivalents (class to node, function to ability) |
| Pattern transformation | 30 | Python patterns that require rethinking in Jac (iterator to walker, class hierarchy to archetype hierarchy) |
| Paradigm shift | 20 | Python code that requires fundamental restructuring for Jac's graph-spatial paradigm |
| Partial conversion | 10 | Large Python programs where only specific components should be converted to Jac idiom |

Each task includes:
- The Python source code (working, with tests)
- A description of which aspects should be converted to Jac idiom
- The reference Jac solution
- Shared test inputs and expected outputs (same behavior in both languages)
- An idiom score rubric specific to this task

### Area 4: Explanation (Jac code to NL)

**60 tasks** testing the model's comprehension and communication.

| Explanation type | Count | Description |
|---|---|---|
| Code walkthrough | 20 | Line-by-line explanation of what Jac code does |
| Concept explanation | 15 | Explain a Jac concept (walker semantics, graph construction, dispatch) using a code example |
| Comparison | 15 | Explain how a Jac program differs from its Python equivalent and why the Jac approach is better |
| Error explanation | 10 | Explain why a piece of Jac code produces a specific error |

Each task includes:
- The Jac code to explain
- The expected key points that a correct explanation must cover
- A rubric for scoring explanation quality (completeness, accuracy, clarity)

Explanation tasks are evaluated by a judge model (Claude) using the rubric. The judge scores each explanation on a 1-5 scale for accuracy, completeness, and clarity. The rubric ensures consistent scoring across both models.

### Area 5: Multi-turn conversation

**50 tasks** testing agentic interaction.

Each task is a multi-turn conversation with 3-5 turns. The first turn is a Jac programming task. Subsequent turns are follow-ups: refinement requests, error reports, edge case questions, optimization requests, or explanation requests.

| Conversation pattern | Count | Description |
|---|---|---|
| Iterative refinement | 15 | Initial code + "make it handle X" + "now optimize for Y" |
| Error recovery | 15 | Initial code + "this produces error X" + "the fix broke Y" |
| Exploratory | 10 | "How would I..." + "What about..." + "Can you show..." |
| Mixed | 10 | Combinations of refinement, error recovery, and exploration |

Each task includes:
- The full reference conversation (all turns)
- At each turn, the expected code state (compiler-verified)
- The expected key content of each response
- A rubric for conversational quality (context maintenance, error recovery, progressive improvement)

Multi-turn evaluation runs the model interactively: the model receives turn 1, generates a response, then receives turn 2 (which may depend on the model's turn-1 response), and so on. The evaluation checks:
1. Does each response's code compile?
2. Does the final code pass all tests?
3. Does the model maintain context across turns (does it remember previous turns)?
4. Does the model recover from introduced errors?

---

## Automated metrics

These metrics are computed automatically with no human judgment. They are the primary metrics for the comparison because they are objective, reproducible, and cheap to compute.

### Compiler pass rate

**Definition**: the fraction of model outputs that compile successfully under the Jac compiler.

**Computation**: for each evaluation task that expects code output, extract the code from the model's response (stripping any markdown formatting, thinking traces, or prose), save it to a temporary `.jac` file, and run the Jac compiler. Record pass/fail.

```bash
# For each model output:
echo "$MODEL_OUTPUT_CODE" > /tmp/eval_task.jac
jac check /tmp/eval_task.jac
# Exit code 0 = pass, non-zero = fail
```

**Target**: >80% compiler pass rate is strong, >90% is excellent. Below 60% indicates the model has not meaningfully learned Jac syntax.

**Breakdown**: report compiler pass rate per difficulty level (atomic, idiomatic, composed) and per construct category. This reveals whether the model's failures are concentrated in specific areas.

### Test pass rate

**Definition**: of the outputs that compile, the fraction that produce correct outputs on the test suite.

**Computation**: for each output that compiles, run the associated test suite and record pass/fail. A test passes only if all test cases in the suite pass.

```bash
# For each compiled output:
jac test /tmp/eval_task.jac
# All tests must pass for the task to count as "test passed"
```

**Target**: >60% test pass rate (of compiled outputs) is strong. Note that test pass rate is conditional on compilation — a model with 90% compiler rate and 70% test rate produces functionally correct code for 63% of tasks overall.

### Construct diversity score

**Definition**: a measure of how many distinct Jac constructs the model uses in its outputs, relative to how many it should use based on the task descriptions.

**Computation**: for each evaluation task, annotate the expected constructs (e.g., "this task should use a walker, a node, and an edge"). Then parse the model's output and count which expected constructs are present. The construct diversity score is:

```
construct_diversity = (constructs used by model) / (constructs expected by task) averaged across all tasks
```

**Why it matters**: a model might achieve high compiler pass rate by collapsing to a subset of Jac features — using basic functions instead of walkers, simple data structures instead of nodes, avoiding edges entirely. The construct diversity score catches this failure mode. A score below 0.7 indicates the model is avoiding Jac-specific features.

### Token efficiency

**Definition**: the average number of output tokens the model needs to produce a correct solution, measured across all tasks where the model produces correct code.

**Computation**: for each correctly solved task, count the total output tokens (including any explanation or reasoning, not just the code). Compare across models.

**Why it matters**: a model that produces correct code but is extremely verbose (long explanations, redundant code, repeated attempts) is less useful than one that produces correct code concisely. Token efficiency also directly affects inference cost in deployment.

---

## Judge-based metrics

These metrics require a judge model (Claude) to evaluate quality aspects that are not captured by automated metrics. They are secondary metrics — used to differentiate models that perform similarly on automated metrics.

### Idiom adherence score

**Definition**: how Jac-idiomatic is the model's code output? Scored 1-5 by a judge.

| Score | Description |
|---|---|
| 1 | Pure Python with Jac syntax — no use of Jac-specific features |
| 2 | Mostly Python patterns with occasional Jac features |
| 3 | Mix of Python and Jac patterns — some idiomatic, some not |
| 4 | Mostly Jac-idiomatic with occasional Python fallbacks |
| 5 | Fully Jac-idiomatic — uses walkers, nodes, edges, abilities where appropriate |

**Judge prompt** (applied to each code output):

```
You are evaluating whether a piece of Jac code is written idiomatically.
Idiomatic Jac code uses:
- Walkers for traversal logic (not recursive functions)
- Nodes and edges for data modeling (not plain classes)
- Abilities for node/walker behaviors (not standalone functions)
- Has-variables with types for node/edge properties
- Jac-specific control flow where appropriate
- Graph-spatial patterns where the problem domain is graph-like

The code to evaluate:
{code}

The task description:
{task}

Score this code from 1 to 5 on Jac idiom adherence. 
Explain your reasoning in 2-3 sentences, then give the score.
```

**Target**: average idiom score >3.5 indicates meaningful Jac learning. Below 2.5 indicates the model is producing Python-with-Jac-syntax.

### Code quality score

**Definition**: overall code quality (readability, efficiency, correctness of patterns), scored 1-5.

| Score | Description |
|---|---|
| 1 | Poorly structured, hard to read, inefficient, incorrect patterns |
| 2 | Basic structure but with significant issues |
| 3 | Acceptable quality — works but not well-crafted |
| 4 | Good quality — clean, readable, efficient |
| 5 | Excellent — production-quality code with proper patterns |

This score is orthogonal to idiom adherence. Code can be idiomatic but poorly structured, or non-idiomatic but well-organized. Both dimensions matter.

### Explanation clarity score

**Definition**: for explanation tasks only, how clear and accurate is the model's explanation? Scored 1-5.

| Score | Description |
|---|---|
| 1 | Incorrect or incomprehensible explanation |
| 2 | Partially correct but confusing or incomplete |
| 3 | Correct but could be clearer or more complete |
| 4 | Clear and accurate with good coverage |
| 5 | Excellent — clear, accurate, well-structured, complete |

---

## Comparison methodology

### Controlled variables

To ensure the comparison is fair, the following variables are held constant across both models:

| Variable | Value | Rationale |
|---|---|---|
| Training data | Exact same 5,000 examples, same order | Isolate model capability |
| Train/validation split | 90/10 (4,500 train, 500 validation) | Consistent evaluation |
| LoRA rank | 16 | Same adapter capacity |
| LoRA alpha | 32 | Same scaling |
| LoRA dropout | 0.05 | Same regularization |
| Learning rate | 2.0e-5 | Same optimization |
| LR schedule | Cosine decay | Same annealing |
| Effective batch size | 8 (micro-batch 1, accumulation 8) | Same gradient statistics |
| Training iterations | 1,875 (3 epochs) | Same data exposure |
| Warmup steps | 100 | Same warmup |
| Max sequence length | 2,048 | Same context |
| Quantization (training) | Q4 | Same precision |
| Quantization (evaluation) | Q8 | Same precision |
| Evaluation prompts | Identical text | Same input |
| Generation temperature | 0.0 (deterministic) | Reproducible |
| Max generation tokens | 512 (code), 1024 (explanation) | Same constraints |
| Random seed | 42 | Reproducible |

### Variables that differ (by necessity)

| Variable | Why it differs | Mitigation |
|---|---|---|
| LoRA target modules | Different architectures name layers differently | Target equivalent layers (attention projections) |
| Chat template | Different tokenizer formats | Use each model's native template |
| Tokenizer | Different vocabularies | Report token efficiency as a metric |
| Number of LoRA layers | Some architectures have more/fewer layers | Use 16 for all; if a model has fewer than 16 layers, adapt all layers |

### Multiple runs for variance

Each model is trained and evaluated 2-3 times with different random seeds (42, 123, 456). This serves two purposes:

1. **Measure training variance**: does the model converge to the same performance level regardless of initialization? High variance suggests the training is unstable or the data is insufficient.

2. **Enable statistical comparison**: with 2-3 runs per model, the evaluation metrics are distributions rather than point estimates. The comparison uses the mean and standard deviation across runs to determine whether differences are meaningful or within noise.

The random seed affects:
- LoRA adapter initialization
- Data shuffling order
- Dropout masks during training

It does not affect:
- The training data content (same 5,000 examples)
- The evaluation tasks or prompts
- The model's pre-trained weights

---

## Decision matrix

The final model selection uses a weighted scoring matrix. Each dimension is scored 1-5 based on the evaluation results, multiplied by its weight, and summed to produce a weighted total.

### Scoring table template

| Dimension | Weight | Gemma 4 26B A4B | Qwen3-Coder-30B-A3B |
|---|---|---|---|
| Compiler pass rate | 25% | _/5 | _/5 |
| Functional correctness (test pass) | 20% | _/5 | _/5 |
| Idiom adherence | 20% | _/5 | _/5 |
| Training efficiency | 10% | _/5 | _/5 |
| Inference speed | 10% | _/5 | _/5 |
| Construct diversity | 10% | _/5 | _/5 |
| License and ecosystem | 5% | _/5 | _/5 |
| **Weighted total** | **100%** | **_/5** | **_/5** |

### Scoring rubric per dimension

**Compiler pass rate** (weight: 25%)
| Score | Criteria |
|---|---|
| 1 | <40% compiler pass rate |
| 2 | 40-60% compiler pass rate |
| 3 | 60-75% compiler pass rate |
| 4 | 75-90% compiler pass rate |
| 5 | >90% compiler pass rate |

**Functional correctness** (weight: 20%)
| Score | Criteria |
|---|---|
| 1 | <30% of compiled outputs pass tests |
| 2 | 30-50% of compiled outputs pass tests |
| 3 | 50-65% of compiled outputs pass tests |
| 4 | 65-80% of compiled outputs pass tests |
| 5 | >80% of compiled outputs pass tests |

**Idiom adherence** (weight: 20%)
| Score | Criteria |
|---|---|
| 1 | Average judge score <2.0 |
| 2 | Average judge score 2.0-2.5 |
| 3 | Average judge score 2.5-3.5 |
| 4 | Average judge score 3.5-4.0 |
| 5 | Average judge score >4.0 |

**Training efficiency** (weight: 10%)
| Score | Criteria |
|---|---|
| 1 | Training failed or required major intervention |
| 2 | Training completed but with instability (loss spikes, restarts needed) |
| 3 | Training completed smoothly, moderate speed |
| 4 | Training completed smoothly and quickly |
| 5 | Training completed quickly with excellent convergence behavior |

**Inference speed** (weight: 10%)
| Score | Criteria |
|---|---|
| 1 | <5 tokens/second on M5 Pro |
| 2 | 5-10 tokens/second |
| 3 | 10-20 tokens/second |
| 4 | 20-35 tokens/second |
| 5 | >35 tokens/second |

**Construct diversity** (weight: 10%)
| Score | Criteria |
|---|---|
| 1 | Construct diversity score <0.4 |
| 2 | Construct diversity score 0.4-0.6 |
| 3 | Construct diversity score 0.6-0.75 |
| 4 | Construct diversity score 0.75-0.9 |
| 5 | Construct diversity score >0.9 |

**License and ecosystem** (weight: 5%)
| Score | Criteria |
|---|---|
| 1 | Restrictive license or no community support |
| 2 | Permissive license but limited ecosystem |
| 3 | Permissive license with moderate ecosystem |
| 4 | Apache 2.0 / MIT with good ecosystem |
| 5 | Apache 2.0 with excellent ecosystem and tooling |

---

## Statistical significance

### Why statistical rigor matters at this sample size

The evaluation set contains 350-500 tasks. When comparing two models with pass rates of, say, 78% and 82%, the question is whether this 4-percentage-point difference reflects a real capability difference or is within the noise of the evaluation. With 100 tasks per capability area, a 4-point difference is only about 4 tasks — well within the range of random variation.

Statistical rigor prevents two failure modes:
1. **Declaring a false winner**: choosing a model that appears better but is actually equivalent, wasting the advantage of the comparison
2. **Declaring "too close to call" when there is a real winner**: being too conservative and not extracting the information the comparison provides

### Bootstrap confidence intervals

For each metric and each model, compute 95% confidence intervals using bootstrap resampling:

```python
import numpy as np

def bootstrap_ci(scores, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval for the mean."""
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        boot_means.append(np.mean(sample))
    
    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return lower, upper

# Example: compiler pass rates for Model A
# scores = [1, 1, 0, 1, 1, 0, 1, ...]  (1 = pass, 0 = fail)
# lower, upper = bootstrap_ci(scores)
# print(f"Compiler pass rate: {np.mean(scores):.1%} (95% CI: {lower:.1%} - {upper:.1%})")
```

### When to declare a winner

A model wins a dimension if its 95% confidence interval does not overlap with the second-place model's 95% confidence interval. If confidence intervals overlap, the models are statistically tied on that dimension.

For the weighted total:
- **Clear winner**: weighted total difference > 0.5 points (on a 5-point scale) AND at least 3 of 7 dimensions are statistically significant wins
- **Marginal winner**: weighted total difference 0.3-0.5 points OR 2 dimensions are significant wins — consider additional tiebreaker evaluation
- **Too close to call**: weighted total difference < 0.3 points AND fewer than 2 dimensions are significant wins — default to Gemma 4 (original primary target) unless secondary factors favor another model

### Minimum sample sizes

The evaluation set sizes are designed to detect meaningful differences:

| Effect size | Tasks needed per group | Detectable with 100 tasks? |
|---|---|---|
| 20 percentage points (e.g., 60% vs 80%) | ~25 | Yes |
| 10 percentage points (e.g., 75% vs 85%) | ~100 | Borderline |
| 5 percentage points (e.g., 80% vs 85%) | ~400 | No |

With 100 tasks per capability area, the evaluation can reliably detect 10+ percentage point differences. Differences smaller than 10 points may not be statistically significant at this sample size. If two models are within 10 points on all metrics, they are effectively equivalent for the purposes of this comparison, and the decision should be made on secondary factors (speed, ecosystem, license).

---

## Evaluation automation

### Evaluation script structure

The evaluation pipeline is fully automated. After both models are trained and fused, a single script runs the complete evaluation:

```bash
# Run evaluation for both models
python eval/run_evaluation.py \
  --models \
    ./models/gemma4-26b-jac-fused-q8 \
    ./models/qwen3-coder-30b-jac-fused-q8 \  --eval-set ./eval/benchmark_tasks/ \
  --output-dir ./eval/results/ \
  --runs 3 \
  --seeds 42,123,456
```

The script:
1. Loads each model in sequence (one at a time due to memory constraints)
2. For each model, runs all evaluation tasks with each seed
3. Collects compiler pass/fail, test pass/fail, token counts
4. Runs the judge model on code outputs for idiom and quality scores
5. Computes all metrics with confidence intervals
6. Populates the decision matrix
7. Generates a comparison report

### Output format

The evaluation produces:
- `results/comparison_table.md` — the filled decision matrix
- `results/{model_name}/scores.json` — per-task scores for each model
- `results/{model_name}/outputs/` — raw model outputs for manual inspection
- `results/statistical_tests.md` — confidence intervals and significance tests
- `results/per_dimension_breakdown.md` — detailed per-dimension analysis

---

## Post-evaluation analysis

Beyond the decision matrix, the evaluation results enable deeper analysis:

### Failure mode analysis

For each model, categorize the tasks it fails on:
- **Syntax failures**: code does not parse (compiler error)
- **Semantic failures**: code compiles but produces wrong output
- **Idiom failures**: code works but uses Python patterns instead of Jac patterns
- **Completeness failures**: code is truncated or incomplete
- **Format failures**: code is embedded in prose/markdown that breaks compilation

If the two models fail on different tasks, this reveals complementary strengths that could inform ensemble strategies or data augmentation for the full pipeline.

### Learning curve analysis

Using the checkpoints saved every 250 steps, evaluate each model at multiple points during training:

| Checkpoint | Steps | Effective examples seen |
|---|---|---|
| Checkpoint 1 | 250 | ~2,000 |
| Checkpoint 2 | 500 | ~4,000 |
| Checkpoint 3 | 750 | ~6,000 |
| Checkpoint 4 | 1000 | ~8,000 |
| Checkpoint 5 | 1250 | ~10,000 |
| Checkpoint 6 | 1500 | ~12,000 |
| Checkpoint 7 | 1875 | ~15,000 |

This reveals the learning curve: does the model improve steadily, or does it plateau early? A model that is still improving at step 1,875 has more room to grow from additional data (promising for the full 300k+ dataset). A model that plateaus at step 500 may have a lower ceiling regardless of data volume.

### Construct-level analysis

Break down performance by Jac construct:

| Construct | Gemma pass rate | Qwen pass rate |
|---|---|---|
| Walker definition | _% | _% |
| Node definition | _% | _% |
| Edge definition | _% | _% |
| Ability dispatch | _% | _% |
| Graph traversal | _% | _% |
| Type annotations | _% | _% |
| ... | ... | ... |

This analysis reveals whether specific Jac constructs are harder for certain models to learn. If one model struggles with walkers but excels at nodes, and another shows the reverse pattern, this information helps design the training data distribution for the full pipeline — allocate more examples to the constructs that the selected model finds hardest.
