# Model Testing Strategy

*Comparative evaluation of 2 candidate base models before committing to full-scale synthetic data generation for Jac*

| | |
|---|---|
| **Objective** | Identify which base model learns Jac best from synthetic finetuning |
| **Candidates** | Gemma 4 26B A4B, Qwen3-Coder-30B-A3B |
| **Test sample** | 5,000 verified examples (curated mini-dataset) |
| **Hardware** | Mac M5 Pro, 20-core GPU, 48GB unified RAM |
| **Framework** | MLX (Apple Silicon native) |
| **Evaluation** | Compiler pass rate, test pass rate, idiom adherence, judge scoring |

---

## Conversion Probe (pre-step)

Before this full 5,000-example comparison, run a cheaper single-category **conversion probe** ([`conversion_probe.md`](conversion_probe.md)). It finetunes both candidates on ~1,500 Python→Jac conversion pairs (plus a small DPO set) and ranks them by cross-compiled test pass rate — an objective metric that needs no subjective judge. Its only job is to drop clear losers cheaply: a model that cannot improve on the most-verifiable category is disqualified before the full comparison spends compute on it. Models that pass the probe (or tie within noise) advance here. The probe does not replace this comparison; it gates entry to it.

---

## Why test before committing to full-scale generation

The full-scale data generation pipeline targets 300,000 to 500,000 verified examples. Producing that volume requires significant investment across multiple dimensions: API costs for Claude Max and other frontier generators run into thousands of dollars, compiler verification time for the 1.5-2.5 million raw candidates needed to yield that volume is measured in days of sustained compute, and the human review effort for quality control is substantial. Once the data is generated, the LoRA finetuning run itself is a multi-day process. If the base model turns out to be poorly suited to learning Jac from synthetic data, all of that investment is wasted.

Testing two candidate models on a small but representative 5,000-example sample costs a fraction of the full pipeline. The API cost for generating 5,000 high-quality examples with Claude Max is roughly 1-2% of the full budget. Each finetuning run on 5,000 examples completes in hours rather than days. The evaluation suite can be run in minutes. The entire two-model comparison can be completed in under a week, compared to the months required for full-scale generation and training.

More importantly, the 5,000-example test reveals failure modes that no amount of pre-training benchmark analysis can predict. A model might score well on LiveCodeBench (Python, JavaScript, C++) but fail to generalize its code understanding to Jac's walker-graph paradigm. A model might handle single-construct Jac code fine but collapse when constructs compose. MoE routing might not adapt well to LoRA updates for one architecture but work perfectly for another. These are empirical questions that can only be answered by actually finetuning on Jac data and measuring outcomes.

The cost of choosing wrong is asymmetric. Spending one week testing two models costs roughly 3% of the total budget. Choosing the wrong model and discovering it after full-scale generation costs 100% of the budget plus the time to start over. This is a straightforward expected-value calculation that strongly favors testing first.

---

## The 5,000-example test sample

### Design principles

The 5,000-example test sample is not a random draw from the full generation pipeline. It is a carefully curated mini-dataset designed to stress-test each model's ability to learn Jac across the full spectrum of capabilities, difficulty levels, and data formats. The sample must be small enough to train on quickly (hours, not days) but representative enough that performance on it predicts performance on the full dataset.

The key design constraint is that every major axis of variation in the full dataset must be represented in the test sample. If a model fails on debugging tasks but excels at code generation, the test sample must contain enough debugging tasks to reveal that. If a model handles atomic-difficulty examples but collapses on composed-difficulty examples, the test sample must include sufficient composed examples to expose that failure mode. If a model responds well to SFT but poorly to DPO, both formats must be present.

### Distribution breakdown

The 5,000 examples are allocated across task categories as follows:

| Category | Count | Percentage | Rationale |
|---|---|---|---|
| Code generation (NL to Jac) | ~2,000 | 40% | Core capability, largest share of full dataset |
| Debugging (broken Jac + error to fix) | ~800 | 16% | Tests error-recovery and code understanding |
| Explanation (Jac code to NL) | ~600 | 12% | Tests comprehension and articulation |
| Conversion (Python to idiomatic Jac) | ~600 | 12% | Tests idiom mapping via cross-compiled test validation, critical for de-Python-ification |
| Multi-turn conversations | ~500 | 10% | Tests agentic interaction and context maintenance |
| DPO preference pairs | ~500 | 10% | Tests whether model learns from negative signal |

Within each category, examples are distributed across three difficulty bands:

| Difficulty band | Description | Share per category |
|---|---|---|
| Atomic | Single construct in isolation (e.g., define a walker that visits one node) | ~30% |
| Idiomatic | Construct used in context as real Jac would use it (e.g., walker with proper dispatch and abilities) | ~40% |
| Composed | 2-3 constructs combined in a non-trivial program (e.g., walker traversal with edge filtering and node transformation) | ~30% |

### Construct coverage

The test sample must include examples that exercise all major Jac language constructs. Even at 5,000 examples, the coverage matrix should span:

- **Graph primitives**: node, edge, walker definitions and instantiation
- **Abilities and archetypes**: method definitions, archetype inheritance, dispatch
- **Walker semantics**: traversal, visit, disengage, spawn, report
- **Type system**: type annotations, generics, has-variables with types
- **Control flow**: if/elif/else, for/while loops, match/case, Jac-specific flow
- **Data spatial**: graph construction, edge connections, node references
- **Async and concurrent**: async abilities, concurrent walkers where applicable
- **Module system**: imports, includes, module-level declarations
- **Testing**: test blocks, assertions, test organization

Not every construct needs hundreds of examples at the test scale. The goal is coverage, not volume. Even 5-10 examples per construct is enough to detect whether a model can learn that construct at all, which is the question the test is designed to answer.

### Data quality requirement

All 5,000 examples are generated using Claude Max (the highest-quality generator available) to isolate the variable being tested. If examples were generated with a mix of generators, poor performance could be attributed to data quality rather than model capability. By using the best generator uniformly, any differences in model performance are attributable to the models themselves.

Every example in the 5,000-example set must pass:

1. **Compiler verification**: all code examples must compile successfully under the Jac compiler
2. **Functional testing**: deterministic examples must produce correct output on test inputs
3. **Idiom judge**: a judge prompt scores each example for Jac-idiomatic usage (not Python-with-Jac-syntax)
4. **Manual spot-check**: a random 10% sample is manually reviewed for quality, correctness, and naturalness
5. **Cross-compiled test validation**: for deterministic code_gen and conversion examples, Python-generated tests compiled to Jac must pass. This follows the MultiPL-T methodology.

### SFT and DPO format inclusion

The test sample includes both SFT (supervised fine-tuning) and DPO (direct preference optimization) format data. Of the 5,000 examples, approximately 4,500 are in SFT format (instruction, response pairs or multi-turn conversations) and approximately 500 are DPO preference pairs (chosen vs. rejected responses for the same prompt).

The DPO pairs are critical because one of the key hypotheses is that negative signal (teaching the model what not to write) is especially important for Jac, where the base model's Python priors actively interfere. If a model responds well to DPO signal in the 5,000-example test, that is a strong positive indicator for the full pipeline where DPO pairs constitute 40,000-80,000 examples.

---

## Mac M5 Pro hardware setup

### Hardware specifications

| Component | Specification |
|---|---|
| Chip | Apple M5 Pro |
| CPU cores | High-performance + efficiency cores |
| GPU cores | 20 |
| Unified RAM | 48 GB |
| Memory bandwidth | ~300 GB/s (estimated for M5 Pro) |
| Storage | NVMe SSD |

### Why MLX, not CUDA

The entire testing pipeline runs on Apple Silicon using the MLX framework. This is not a compromise — it is the intended deployment path for the testing phase. MLX is purpose-built for Apple Silicon and exploits the unified memory architecture where CPU and GPU share the same physical RAM with no copy overhead. For MoE models specifically, this is advantageous: the full model weights live in unified memory and only the active expert parameters need GPU compute on each forward pass.

CUDA-based alternatives (PyTorch, Unsloth, vLLM) are designed for discrete GPU architectures with separate VRAM. They do not run on Apple Silicon. The testing phase uses MLX exclusively; the full-scale training may move to cloud A100 infrastructure if the Mac proves too slow for the 300k+ dataset, but the model selection decision is made entirely from Mac-based results.

### Memory budget for Q4 training

The memory budget for LoRA finetuning a Q4-quantized MoE model on the M5 Pro:

| Component | Estimated memory |
|---|---|
| Q4 model weights (~26B params, MoE) | ~13 GB |
| LoRA adapter weights (rank 16-32) | ~200-400 MB |
| Optimizer states (AdamW for LoRA params only) | ~600 MB - 1.2 GB |
| Activations and gradients (batch size 1-2) | ~4-8 GB |
| Data loading and tokenization buffers | ~2-4 GB |
| MLX framework overhead | ~1-2 GB |
| **Total estimated** | **~21-29 GB** |
| **Remaining headroom (of 48 GB)** | **~19-27 GB** |

This budget leaves comfortable headroom. The key constraint is batch size: with Q4 weights at ~13 GB and the rest of the training state, batch sizes above 2-4 will push memory usage toward the 48 GB ceiling. Gradient accumulation over multiple micro-batches compensates for the small per-step batch size.

### Memory budget for Q8 inference evaluation

| Component | Estimated memory |
|---|---|
| Q8 model weights (~26B params, MoE) | ~26 GB |
| KV cache (8K context) | ~2-4 GB |
| MLX inference overhead | ~1-2 GB |
| **Total estimated** | **~29-32 GB** |
| **Remaining headroom** | **~16-19 GB** |

Q8 is feasible for inference but not for training (the combined Q8 weights + training state would exceed 48 GB). The evaluation workflow is: train with Q4 weights + LoRA adapters, then merge adapters into Q8 weights for final evaluation. This gives the best quality evaluation (Q8 has less quantization error than Q4) while keeping training within memory limits.

### Expected training times

These are rough estimates based on MLX benchmarks for similar-sized MoE models on Apple Silicon:

| Model | Q4 training on 5k examples (3 epochs) | Q8 inference eval (500 tasks) |
|---|---|---|
| Gemma 4 26B A4B | ~4-8 hours | ~2-4 hours |
| Qwen3-Coder-30B-A3B | ~4-8 hours | ~2-4 hours |

Total wall-clock time for both models: approximately 8-16 hours of training plus 4-8 hours of evaluation, spread across 1-2 days with some parallelism in data preparation and result analysis.

---

## Evaluation framework

Both models are evaluated using the identical evaluation suite, described in detail in [`evaluation.md`](evaluation.md). The key principle is that every variable except the base model is held constant:

- **Same training data**: the exact same 5,000-example dataset, in the same order, with the same train/validation split
- **Same LoRA configuration**: same rank, same alpha, same target modules (adjusted only where architectures differ), same learning rate schedule
- **Same number of training steps**: same number of epochs, same effective batch size
- **Same evaluation tasks**: the exact same held-out evaluation set, run with the same prompts, same temperature, same sampling parameters
- **Same evaluation metrics**: compiler pass rate, test pass rate, idiom adherence score, construct diversity score, token efficiency
- **Same hardware**: all runs on the same Mac M5 Pro

The only variable is the base model. This controlled comparison is what makes the results actionable.

---

## Decision criteria

The decision matrix uses a weighted scoring system across seven dimensions. Each dimension is scored 1-5 for each model, then multiplied by its weight. The model with the highest weighted total is selected.

| Dimension | Weight | What it measures |
|---|---|---|
| Compiler pass rate | 25% | Hard metric: does generated code compile? |
| Functional correctness | 20% | Does compiled code produce correct outputs? |
| Idiom adherence | 20% | Is the output Jac-idiomatic or Python-with-Jac-syntax? |
| Training efficiency | 10% | Time to train, memory usage, stability of loss curves |
| Inference speed | 10% | Tokens per second during generation |
| Construct diversity | 10% | Does the model use the full Jac feature set or collapse to a subset? |
| License and ecosystem | 5% | Apache 2.0, community support, tooling availability |

A model wins outright if it leads by more than 0.5 points on the weighted total (on a 5-point scale). If the two models are within 0.5 points, additional tiebreaker runs are conducted with a larger evaluation set, or the model with better idiom adherence wins (since that is the hardest capability to train and the most important for Jac adoption).

---

## Timeline

| Day | Activity |
|---|---|
| Pre-step | Run the conversion probe ([`conversion_probe.md`](conversion_probe.md)), ~3-5 days. Drop any model that does not improve on cross-compiled test pass rate; advance the rest into the schedule below. |
| Day 1 | Generate 5,000-example test dataset using Claude Max. Run compiler verification, test suite, idiom judge. Manual spot-check 500 examples. |
| Day 2 | Download and quantize both models. Verify MLX compatibility. Dry-run training with 100 examples each to confirm memory fits and training loop works. |
| Day 3-4 | Train Model 1 (Gemma 4 26B A4B). Monitor loss curves, save checkpoints every 500 steps. |
| Day 4-5 | Train Model 2 (Qwen3-Coder-30B-A3B). Same monitoring. |
| Day 5 | Merge LoRA adapters into Q8 base weights for both models. Run full evaluation suite on both. |
| Day 6 | Analyze results. Populate decision matrix. Write comparison report. Make recommendation. |

This is a conservative schedule that assumes sequential training (one model at a time, since each saturates the M5 Pro's memory). If any model shows catastrophic failure during training (loss divergence, NaN gradients, memory overflow), it is disqualified early and the schedule compresses.

---

## Risk mitigation

### Model does not fit in memory

Despite the memory budget analysis showing headroom, edge cases exist (unexpected activation sizes, MLX-specific overhead, model architecture quirks). Mitigation: start each model with a 100-example dry run before committing to the full 5,000-example training. If memory overflows, reduce LoRA rank from 32 to 16 or 8, reduce batch size to 1, or reduce sequence length.

### MLX does not support a model architecture

MLX's model support is broad but not universal. Some MoE architectures may require custom conversion scripts. Mitigation: download and convert both models on Day 2 before any training begins. If a model fails to convert, investigate community forks or alternative quantization tools. If conversion is fundamentally blocked, the model is disqualified.

### Both models perform poorly

If no model achieves >50% compiler pass rate on the evaluation set after finetuning, the problem is likely data quality or data format rather than model choice. Mitigation: inspect failure modes, adjust the training data format, and re-run with a revised 5,000-example set. Do not proceed to full-scale generation until at least one model demonstrates meaningful Jac capability improvement from finetuning.

### Results are too close to call

If both models score within 0.5 weighted points of each other, the decision is less about which model is "best" and more about secondary factors: community ecosystem, long-term maintenance, inference deployment options. In this case, default to Gemma 4 26B A4B (the original primary target) unless a compelling reason emerges to switch.

---

## Relationship to full-scale generation

The model testing phase does not change the data generation strategy described in [`../datagenstrat/strat.md`](../datagenstrat/strat.md). The ten generation recipes, the verification pipeline, the volume targets, and the quality controls all remain the same regardless of which base model is selected. The only thing that changes is the target model name in the finetuning configuration.

This is by design. The data pipeline was built to be model-agnostic: it generates high-quality Jac training data that any capable base model should benefit from. The 5,000-example test dataset should include conversion examples validated with cross-compiled tests to ensure the selected model can learn from test-validated translations, which form the backbone of Recipe 2 at scale. The model testing phase answers the question of which base model benefits the most.
