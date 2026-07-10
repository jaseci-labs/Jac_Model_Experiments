# Gemma 4 26B A4B — Model Testing Guide

*Primary candidate for Jac finetuning. MoE architecture, 3.8B active parameters, Apache 2.0.*

---

## Model specifications

| Attribute | Value |
|---|---|
| **Full name** | Gemma 4 26B A4B Instruct |
| **Developer** | Google DeepMind |
| **Total parameters** | ~26 billion |
| **Active parameters per token** | ~3.8 billion (MoE routing) |
| **Architecture** | Mixture-of-Experts Transformer |
| **Context length** | 128K tokens |
| **License** | Apache 2.0 |
| **Release date** | April 2, 2026 |
| **LiveCodeBench v6** | 77.1% |
| **AIME 2026** | 88.3% |
| **tau2-bench (agentic)** | 86.4% |
| **Arena AI Elo (31B dense sibling)** | 1452 (#3) |
| **Hugging Face ID** | `google/gemma-4-26b-a4b-it` |

---

## Why this model

Gemma 4 26B A4B is the primary candidate for Jac finetuning for several reinforcing reasons.

**MoE efficiency is the core enabler.** The model has 26 billion total parameters but only activates 3.8 billion per token through its mixture-of-experts routing. This means inference compute is comparable to a 4B dense model while the total knowledge capacity is that of a 26B model. For LoRA finetuning, this translates to dramatically lower memory and compute requirements than a dense model of equivalent capability. The entire Q4-quantized model fits in approximately 13 GB, leaving ample room for LoRA adapters and training state within the 48 GB unified memory budget.

**Agentic capabilities transfer to Jac.** The tau2-bench score of 86.4% represents a massive leap from Gemma 3 27B's 6.6%. This agentic training — tool use, multi-step planning, error recovery, structured output generation — directly benefits a Jac coding agent. Jac's walker-graph paradigm is inherently agentic: walkers traverse graphs, make decisions at nodes, and dispatch abilities based on context. A model pre-trained for agentic behavior should adapt more naturally to this paradigm than one trained purely on static code completion.

**Apache 2.0 licensing removes all constraints.** The finetuned model can be distributed, commercially deployed, and modified without restriction. This is non-negotiable for the Jaseci ecosystem, where the finetuned model will be distributed as part of the developer toolchain.

**Google's training infrastructure and data quality.** Gemma 4 was trained on one of the largest and most diverse training corpora available, using Google's proprietary infrastructure. The resulting model has strong multilingual capabilities, broad code understanding across many languages, and robust instruction-following behavior. This breadth means the model has seen more diverse programming patterns, which should help it generalize to a new language like Jac.

**Strong code benchmarks.** The 77.1% LiveCodeBench v6 score demonstrates genuine code generation capability, not just pattern matching. LiveCodeBench tests are post-training-cutoff problems, so this score reflects the model's ability to solve novel coding tasks — exactly what is needed for generating Jac code from natural language descriptions.

---

## Mac M5 Pro feasibility

### Quantization options

| Quantization | Approximate size | Fits in 48GB? | Use case |
|---|---|---|---|
| Q4 (4-bit) | ~13 GB | Yes, with ample headroom | Training (LoRA finetuning) |
| Q8 (8-bit) | ~26 GB | Yes, with constraints | Inference evaluation |
| BF16 (full precision) | ~52 GB | No | Not feasible on this hardware |
| Q6 (6-bit) | ~19.5 GB | Yes | Alternative training precision |

**Recommended configuration**: Q4 for all training runs, Q8 for final evaluation inference. The Q4 training introduces some quantization noise in the frozen weights, but LoRA adapters are trained in full precision, so the adapter learning is not degraded. The final evaluation uses Q8 to minimize quantization artifacts when measuring the model's true capability after finetuning.

### Detailed memory analysis for Q4 training

| Component | Memory estimate | Notes |
|---|---|---|
| Q4 model weights | ~13 GB | 26B params at 4 bits per param, plus MoE overhead |
| LoRA adapter weights | ~200-400 MB | Rank 16-32, applied to attention projections |
| LoRA optimizer states | ~400 MB - 1 GB | AdamW: 2x adapter size for momentum + variance |
| Activations (batch size 1) | ~2-4 GB | Depends on sequence length (2048 default) |
| Gradient computation | ~2-4 GB | Transient, freed after backward pass |
| Tokenizer and data pipeline | ~1-2 GB | Tokenized datasets, data loader buffers |
| MLX framework overhead | ~1-2 GB | Graph compilation, memory allocator |
| **Total** | **~20-27 GB** | |
| **Headroom** | **~21-28 GB** | Comfortable margin |

### Detailed memory analysis for Q8 inference

| Component | Memory estimate | Notes |
|---|---|---|
| Q8 model weights (with merged LoRA) | ~26 GB | Full 8-bit quantized weights |
| KV cache (8K context) | ~2-4 GB | Scales with context length |
| MLX inference buffers | ~1-2 GB | Token processing, sampling |
| **Total** | **~29-32 GB** | |
| **Headroom** | **~16-19 GB** | Sufficient for evaluation |

---

## Setup instructions

### Step 1: Install dependencies

```bash
# Ensure MLX and related packages are installed
pip install mlx mlx-lm huggingface-hub

# Verify MLX can see the GPU
python -c "import mlx.core as mx; print(mx.default_device())"
# Expected output: Device(gpu, 0)
```

### Step 2: Download the model

```bash
# Download the instruct variant (chat-tuned, which is the correct base for finetuning)
huggingface-cli download google/gemma-4-26b-a4b-it \
  --local-dir ./models/gemma4-26b-hf \
  --local-dir-use-symlinks False
```

This download is approximately 52 GB (full-precision weights). The quantization step below will produce the smaller Q4 and Q8 variants.

### Step 3: Convert to MLX format and quantize

```bash
# Convert to MLX format with Q4 quantization (for training)
mlx_lm.convert \
  --hf-path google/gemma-4-26b-a4b-it \
  --mlx-path ./models/gemma4-26b-q4 \
  --quantize \
  --q-bits 4

# Convert to MLX format with Q8 quantization (for evaluation)
mlx_lm.convert \
  --hf-path google/gemma-4-26b-a4b-it \
  --mlx-path ./models/gemma4-26b-q8 \
  --quantize \
  --q-bits 8
```

### Step 4: Verify the model loads and generates

```bash
# Quick smoke test: generate a short response
mlx_lm.generate \
  --model ./models/gemma4-26b-q4 \
  --prompt "Write a simple Jac walker that visits all nodes in a graph." \
  --max-tokens 256
```

If this command completes without memory errors and produces coherent output, the model is ready for finetuning. The output will almost certainly be incorrect Jac (the model has not been finetuned yet), but it should demonstrate that the model loads, tokenizes, and generates without crashing.

---

## LoRA finetuning configuration

### Training configuration file

Create the file `configs/gemma4_lora.yaml`:

```yaml
# Gemma 4 26B A4B - LoRA finetuning config for MLX
model: "./models/gemma4-26b-q4"
train: true
data: "./data/jac_5k_train"
valid: "./data/jac_5k_valid"
test: "./data/jac_5k_test"

# LoRA configuration
lora_layers: 16            # Number of layers to apply LoRA (last N layers)
lora_parameters:
  rank: 16                 # LoRA rank — 16 is a good balance of capacity vs. memory
  alpha: 32                # Scaling factor, typically 2x rank
  dropout: 0.05            # Light dropout to prevent overfitting on 5k examples
  scale: 10.0              # LoRA scaling

# Training hyperparameters
learning_rate: 2.0e-5      # Conservative LR for MoE — too high risks destabilizing routing
lr_schedule: cosine        # Cosine decay to minimum
batch_size: 1              # Micro-batch size (memory constrained)
grad_accumulation_steps: 8 # Effective batch size = 8
iters: 1875                # 5000 examples * 3 epochs / 8 effective batch = 1875 steps
warmup_steps: 100          # ~5% of total steps
weight_decay: 0.01         # Light regularization

# Sequence length
max_seq_length: 2048       # Sufficient for most Jac examples; increase if needed

# Checkpointing
save_every: 250            # Save checkpoint every 250 steps
adapter_path: "./adapters/gemma4-26b-jac"

# Evaluation
val_batches: 50            # Evaluate on 50 validation batches
steps_per_eval: 100        # Evaluate every 100 training steps

# Seed for reproducibility
seed: 42
```

### Training command

```bash
mlx_lm.lora \
  --config configs/gemma4_lora.yaml
```

### Expected training behavior

**Loss curve**: expect the training loss to drop rapidly in the first 200-300 steps as the model picks up basic Jac syntax patterns. The loss should then plateau around step 500-800 and continue a slow decline through the remaining steps. If the loss spikes or oscillates after the initial drop, the learning rate is too high — reduce to 1.0e-5.

**Validation loss**: should track training loss with a small gap. If validation loss starts increasing while training loss continues decreasing, the model is overfitting. With only 5,000 examples and 3 epochs, mild overfitting is expected and acceptable. Severe overfitting (validation loss increasing by >0.5 while training loss decreases) indicates the LoRA rank is too high or the learning rate needs reduction.

**Memory usage**: monitor with `sudo powermetrics --samplers gpu_power -i 1000` or Activity Monitor. Peak memory should stay under 30 GB during training. If memory exceeds 40 GB, reduce `max_seq_length` to 1024 or reduce `lora_layers` from 16 to 8.

**Training time**: approximately 4-8 hours for the full 1,875 steps on the M5 Pro. Each step processes one micro-batch of 1 example through forward and backward passes. With gradient accumulation over 8 steps, one optimizer update happens every ~15-30 seconds.

### Merging LoRA adapters for evaluation

After training completes, merge the LoRA adapters into the Q8 base model for evaluation:

```bash
# Fuse LoRA adapters into the base model
mlx_lm.fuse \
  --model ./models/gemma4-26b-q8 \
  --adapter-path ./adapters/gemma4-26b-jac \
  --save-path ./models/gemma4-26b-jac-fused-q8 \
  --de-quantize  # Merge at full precision, then re-quantize
```

The fused model is a standalone model that can be used for inference without loading separate adapter weights. This is the model that gets evaluated.

---

## Known strengths for Jac finetuning

### Strong agentic capabilities

The tau2-bench score of 86.4% means the model has been extensively trained on agentic behavior: planning multi-step actions, using tools, recovering from errors, maintaining state across interactions. Jac's walker-graph paradigm is fundamentally agentic — walkers are autonomous agents that traverse graphs, make decisions, and take actions. A model with strong agentic priors should map onto this paradigm more naturally than a model trained purely on static code completion.

Specifically, the agentic training should help with:
- Multi-turn conversation tasks where the model needs to refine Jac code iteratively
- Debugging tasks where the model needs to diagnose compiler errors and propose fixes
- Composed-difficulty examples where the model needs to plan how multiple constructs interact
- Walker traversal logic where the model needs to reason about state changes across graph nodes

### Good structured output generation

Gemma 4's instruction-following capabilities are strong, which matters for generating Jac code in specific formats. The finetuning data uses structured prompt-response pairs, and the model needs to produce well-formatted Jac code consistently. Models with weak instruction following tend to add extra commentary, deviate from requested formats, or produce incomplete outputs — all of which reduce the signal-to-noise ratio of the training data.

### MoE architecture benefits for finetuning

The MoE architecture means that LoRA only needs to adapt a subset of the model's total parameters. During each forward pass, only the active expert parameters (3.8B) are engaged, so the LoRA updates affect a smaller computational graph. This can lead to more stable training because the gradient signal is not diluted across 26B parameters but concentrated on the 3.8B that are actually active.

There is also a potential advantage for specialization: different experts may learn to handle different aspects of Jac. One expert might specialize in walker semantics, another in graph construction, another in type annotations. This is speculative but consistent with how MoE models are observed to develop specialization during pre-training.

---

## Known risks for Jac finetuning

### Python-priored generation

Like all frontier language models, Gemma 4 has been trained predominantly on Python code. Its default code generation behavior is to produce Python-like patterns. When asked to generate Jac code, it will tend to:
- Use function-based decomposition where Jac idiom calls for walkers
- Write class hierarchies where Jac uses archetypes and nodes
- Use iterative loops where Jac's traversal semantics are more appropriate
- Default to Python naming conventions and style patterns

This is the core challenge that the synthetic data pipeline is designed to overcome. The 5,000-example test will reveal how quickly and thoroughly the LoRA finetuning can override these Python priors with Jac-idiomatic patterns.

### MoE routing may not adapt to small finetunes

MoE routing is learned during pre-training and determines which expert processes each token. LoRA finetuning does not directly modify the routing weights — it adds low-rank updates to the expert weights themselves. If the pre-trained routing consistently sends Jac-relevant tokens to experts that are not well-suited for Jac, the LoRA updates may need to work harder to compensate.

This is a theoretical risk. In practice, MoE routing tends to be flexible enough that LoRA-adapted expert weights can shift the model's behavior without routing changes. But it is worth monitoring: if the model's performance plateaus early during training despite continuing loss reduction, routing limitations may be the bottleneck.

### Google's chat template may conflict with custom formats

Gemma 4 Instruct uses Google's specific chat template with turn markers and role tokens. The finetuning data must be formatted to match this template exactly, or the model may struggle to distinguish between system prompts, user inputs, and assistant responses. If the Jac training data uses a different format than Gemma expects, the model may produce outputs with misplaced turn markers or inconsistent formatting.

Mitigation: use the Gemma 4 tokenizer's built-in chat template when formatting the training data. The `mlx_lm` library handles this automatically when data is provided in the standard conversation format (list of role/content dicts).

### Quantization noise during training

Q4 quantization introduces noise in the frozen model weights. While LoRA adapters are trained in full precision, the forward pass through quantized weights produces slightly different activations than would occur with full-precision weights. For most models and tasks, this noise is negligible. But for a specialized task like Jac code generation — where the model is learning a new language with strict syntax requirements — quantization noise could manifest as inconsistent syntax production.

Mitigation: if evaluation results show the Q4-trained model producing syntactically inconsistent code (sometimes correct, sometimes garbled), try training at Q6 (which uses approximately 19.5 GB for weights, still within budget). If Q6 resolves the issue, the final full-scale training should use Q6.

---

## What to watch during training

### Loss curves

Plot training and validation loss every 100 steps. Expected behavior:

- **Steps 0-200**: rapid loss decrease as model learns basic Jac syntax
- **Steps 200-600**: slower decrease as model refines construct usage
- **Steps 600-1200**: plateau with minor fluctuations
- **Steps 1200-1875**: slight continued improvement or stable

Red flags:
- **Loss spikes**: reduce learning rate by 2x
- **Monotonic validation increase after step 500**: overfitting, reduce LoRA rank or increase dropout
- **Loss not decreasing at all**: LoRA targets are wrong (check which layers are being adapted), or learning rate is too low
- **NaN loss**: catastrophic failure, likely a precision issue — try Q6 quantization or reduce learning rate to 5.0e-6

### Generated samples during training

Every 250 steps (at each checkpoint), generate 5-10 sample outputs using the current adapter weights on a fixed set of prompts. Visually inspect:

- Is the output syntactically valid Jac? (Even before compiler verification)
- Is the output Jac-idiomatic or Python-with-Jac-syntax?
- Does the output address the prompt correctly?
- Is the output complete (not cut off mid-statement)?

This qualitative monitoring catches issues that loss numbers miss. A model can have low loss but still produce non-idiomatic code if the loss is driven by format matching rather than semantic correctness.

### MoE routing behavior

MLX may not expose MoE routing statistics directly, but if available (through model internals or logging), monitor:

- Which experts are being activated most frequently for Jac tokens
- Whether the routing distribution changes during finetuning
- Whether specific Jac constructs consistently route to the same experts

This information is useful for understanding the model's internal representation of Jac but is not critical for the go/no-go decision.

### Memory and throughput

Monitor throughout training:
- Peak memory usage (should stay under 35 GB)
- Steps per second (should remain roughly constant; degradation suggests memory pressure)
- GPU utilization (via `sudo powermetrics --samplers gpu_power`)

If memory usage creeps upward during training, there may be a memory leak in the training loop or the KV cache is growing unbounded. Restart training from the last checkpoint with explicit cache clearing.

---

## Post-training evaluation

After training, the fused Q8 model is evaluated using the full evaluation suite described in [`evaluation.md`](evaluation.md). The Gemma-specific considerations for evaluation:

1. **Use Gemma's chat template for all prompts**: the fused model retains Gemma's expected input format. Evaluation prompts must use the same template as training data.

2. **Set temperature to 0.0 for deterministic evaluation**: this ensures reproducible results across multiple runs. For creativity-sensitive tasks (explanation, multi-turn), run a separate evaluation at temperature 0.7 to assess diversity.

3. **Monitor inference speed**: Gemma 4's MoE routing should give it an inference speed advantage over dense models of similar total size. Record tokens per second as a metric for the comparison.

4. **Check for Gemma-specific artifacts**: look for outputs that include Gemma's special tokens, safety disclaimers, or formatting markers where they should not appear. These are artifacts of the base model's instruction tuning that may leak through the LoRA finetuning.

---

## Gemma 4 26B A4B — quick reference

```
Download:     huggingface-cli download google/gemma-4-26b-a4b-it
Convert Q4:   mlx_lm.convert --hf-path google/gemma-4-26b-a4b-it --mlx-path ./models/gemma4-26b-q4 --quantize --q-bits 4
Convert Q8:   mlx_lm.convert --hf-path google/gemma-4-26b-a4b-it --mlx-path ./models/gemma4-26b-q8 --quantize --q-bits 8
Train:        mlx_lm.lora --config configs/gemma4_lora.yaml
Fuse:         mlx_lm.fuse --model ./models/gemma4-26b-q8 --adapter-path ./adapters/gemma4-26b-jac --save-path ./models/gemma4-26b-jac-fused-q8
Generate:     mlx_lm.generate --model ./models/gemma4-26b-jac-fused-q8 --prompt "..." --max-tokens 512
```
