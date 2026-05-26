# Qwen3-Coder-30B-A3B — Model Testing Guide

*Second candidate for Jac finetuning. Code-specialized MoE architecture, ~3B active parameters, Apache 2.0.*

---

## Model specifications

| Attribute | Value |
|---|---|
| **Full name** | Qwen3-Coder-30B-A3B |
| **Developer** | Alibaba Cloud (Qwen Team) |
| **Total parameters** | ~30 billion |
| **Active parameters per token** | ~3 billion (MoE routing) |
| **Architecture** | Mixture-of-Experts Transformer |
| **Context length** | 128K tokens (up to 256K with YaRN) |
| **License** | Apache 2.0 |
| **Release date** | 2026 |
| **HumanEval+** | Strong (code-specialized variant) |
| **MBPP+** | Strong (code-specialized variant) |
| **LiveCodeBench** | Competitive with similarly-sized code models |
| **Hugging Face ID** | `Qwen/Qwen3-Coder-30B-A3B` |

---

## Why this model

Qwen3-Coder-30B-A3B brings a different set of strengths to the Jac finetuning challenge compared to Gemma 4. Its inclusion in the comparison is motivated by several factors that make it a credible alternative.

**Code-specialized training.** Unlike Gemma 4, which is a general-purpose model that happens to be good at code, Qwen3-Coder is explicitly a code-specialized variant. The Qwen team trained it with an emphasis on programming tasks: code generation, code completion, debugging, refactoring, and code explanation. This specialization means the model has seen more diverse code patterns during training, has stronger representations of programming concepts, and has better-calibrated priors about code structure. For Jac finetuning, this could translate to faster learning of Jac syntax and semantics — the model already "thinks in code" and needs to learn a new language, rather than learning to think in code and learn a new language simultaneously.

**MoE efficiency comparable to Gemma 4.** The ~3B active parameters per token put Qwen3-Coder in the same efficiency class as Gemma 4's 3.8B active parameters. This means the memory and compute requirements for LoRA finetuning are essentially identical: Q4 quantization yields approximately 15 GB of model weights, well within the M5 Pro's 48 GB budget. The training time estimates are similar, so the comparison is fair — neither model has a hardware advantage.

**Apache 2.0 licensing.** Same unrestricted licensing as Gemma 4. The finetuned model can be distributed and commercially deployed without restrictions.

**Strong ecosystem and community.** The Qwen model family has a large and active community, with extensive documentation, community-contributed tooling, and regular model updates. The MLX ecosystem has good support for Qwen architectures, with community-maintained conversion scripts and quantization recipes. This reduces the risk of encountering unsupported architecture features during the testing phase.

**Alibaba's training data diversity.** Qwen models are trained on a diverse multilingual corpus that includes significant non-English content. While this is less directly relevant for Jac (which uses English-language keywords), the training data diversity may help the model generalize to new syntactic patterns — it has encountered more varied token sequences than a model trained predominantly on English text.

---

## Mac M5 Pro feasibility

### Quantization options

| Quantization | Approximate size | Fits in 48GB? | Use case |
|---|---|---|---|
| Q4 (4-bit) | ~15 GB | Yes, with headroom | Training (LoRA finetuning) |
| Q8 (8-bit) | ~30 GB | Yes, tight | Inference evaluation |
| BF16 (full precision) | ~60 GB | No | Not feasible on this hardware |
| Q6 (6-bit) | ~22.5 GB | Yes | Alternative training precision |

**Recommended configuration**: Q4 for training, Q8 for evaluation, same as Gemma 4. The slightly larger total parameter count (30B vs. 26B) means Q8 inference is tighter on memory (~30 GB for weights alone), but still feasible with the remaining 18 GB for KV cache and inference overhead.

### Memory analysis for Q4 training

| Component | Memory estimate | Notes |
|---|---|---|
| Q4 model weights | ~15 GB | 30B params at 4 bits, MoE overhead |
| LoRA adapter weights | ~200-400 MB | Rank 16-32, attention projections |
| Optimizer states (AdamW) | ~400 MB - 1 GB | 2x adapter size |
| Activations (batch size 1) | ~2-4 GB | Sequence length dependent |
| Gradients | ~2-4 GB | Transient |
| Data pipeline and tokenizer | ~1-2 GB | Buffers |
| MLX overhead | ~1-2 GB | Framework |
| **Total** | **~22-29 GB** | |
| **Headroom** | **~19-26 GB** | Comfortable |

### Memory analysis for Q8 inference

| Component | Memory estimate | Notes |
|---|---|---|
| Q8 model weights (fused) | ~30 GB | Full 8-bit quantized |
| KV cache (8K context) | ~2-4 GB | Sequence length dependent |
| MLX inference buffers | ~1-2 GB | Token processing |
| **Total** | **~33-36 GB** | |
| **Headroom** | **~12-15 GB** | Tighter than Gemma, but feasible |

Note: the Q8 inference for Qwen3-Coder is tighter than Gemma 4 (12-15 GB headroom vs. 16-19 GB). If memory issues arise during Q8 evaluation, fall back to Q6 (approximately 22.5 GB weights) which provides a good balance between quantization quality and memory usage.

---

## Setup instructions

### Step 1: Install dependencies

```bash
# Same MLX stack as Gemma
pip install mlx mlx-lm huggingface-hub

# Verify MLX GPU access
python -c "import mlx.core as mx; print(mx.default_device())"
```

### Step 2: Download the model

```bash
# Download Qwen3-Coder-30B-A3B
huggingface-cli download Qwen/Qwen3-Coder-30B-A3B \
  --local-dir ./models/qwen3-coder-30b-hf \
  --local-dir-use-symlinks False
```

This download is approximately 60 GB for the full-precision weights.

### Step 3: Convert to MLX format and quantize

```bash
# Q4 quantization for training
mlx_lm.convert \
  --hf-path Qwen/Qwen3-Coder-30B-A3B \
  --mlx-path ./models/qwen3-coder-30b-q4 \
  --quantize \
  --q-bits 4

# Q8 quantization for evaluation
mlx_lm.convert \
  --hf-path Qwen/Qwen3-Coder-30B-A3B \
  --mlx-path ./models/qwen3-coder-30b-q8 \
  --quantize \
  --q-bits 8
```

### Step 4: Verify the model loads and generates

```bash
mlx_lm.generate \
  --model ./models/qwen3-coder-30b-q4 \
  --prompt "Write a simple Jac walker that visits all nodes in a graph." \
  --max-tokens 256
```

As with Gemma, the output will not be correct Jac (no finetuning yet), but this verifies the model loads and generates without errors.

---

## LoRA finetuning configuration

### Training configuration file

Create the file `configs/qwen3_coder_lora.yaml`:

```yaml
# Qwen3-Coder-30B-A3B - LoRA finetuning config for MLX
model: "./models/qwen3-coder-30b-q4"
train: true
data: "./data/jac_5k_train"
valid: "./data/jac_5k_valid"
test: "./data/jac_5k_test"

# LoRA configuration
lora_layers: 16            # Last 16 layers
lora_parameters:
  rank: 16                 # Same rank as Gemma for fair comparison
  alpha: 32                # 2x rank
  dropout: 0.05            # Same regularization
  scale: 10.0              # LoRA scaling

# Training hyperparameters
learning_rate: 2.0e-5      # Same LR as Gemma for controlled comparison
lr_schedule: cosine
batch_size: 1              # Memory constrained
grad_accumulation_steps: 8 # Effective batch size = 8
iters: 1875                # Same iteration count as Gemma (5k * 3 epochs / 8)
warmup_steps: 100
weight_decay: 0.01

# Sequence length
max_seq_length: 2048

# Checkpointing
save_every: 250
adapter_path: "./adapters/qwen3-coder-30b-jac"

# Evaluation
val_batches: 50
steps_per_eval: 100

# Seed
seed: 42
```

### Qwen-specific LoRA target modules

Qwen3-Coder uses a different attention module naming convention than Gemma. The LoRA target modules should be verified from the model architecture:

```python
# Inspect Qwen3-Coder's architecture to confirm LoRA targets
import mlx.nn as nn
from mlx_lm import load

model, tokenizer = load("./models/qwen3-coder-30b-q4")

# Print layer names to identify attention projections
for name, module in model.named_modules():
    if "attn" in name.lower() or "self_attn" in name.lower():
        print(name, type(module))
```

Typical Qwen attention modules:
- `self_attn.q_proj` — query projection
- `self_attn.k_proj` — key projection
- `self_attn.v_proj` — value projection
- `self_attn.o_proj` — output projection

MLX's LoRA implementation should automatically detect these, but if manual specification is needed, add to the config:

```yaml
lora_parameters:
  keys: ["self_attn.q_proj", "self_attn.v_proj"]
```

### Training command

```bash
mlx_lm.lora \
  --config configs/qwen3_coder_lora.yaml
```

### Expected training time

Approximately 4-8 hours for the full 1,875 steps on the M5 Pro. The slightly larger model (30B vs. 26B) may result in marginally longer per-step times, but since only ~3B parameters are active per token (vs. Gemma's 3.8B), the difference should be minimal. MoE routing overhead is the wildcard — Qwen's routing mechanism may have different computational characteristics than Gemma's.

### Merging LoRA adapters

```bash
mlx_lm.fuse \
  --model ./models/qwen3-coder-30b-q8 \
  --adapter-path ./adapters/qwen3-coder-30b-jac \
  --save-path ./models/qwen3-coder-30b-jac-fused-q8 \
  --de-quantize
```

---

## Known strengths for Jac finetuning

### Code-specialized pre-training

The single biggest advantage of Qwen3-Coder over the other candidates is its code-specialized training. The model was not just exposed to code during general pre-training — it was specifically optimized for code tasks. This means:

- **Stronger code syntax priors**: the model has a more refined internal representation of programming language syntax in general. Learning a new syntax (Jac) is easier when the model already has strong priors about what valid syntax looks like across many languages.
- **Better code-structure understanding**: the model understands control flow, scoping, type systems, and function signatures at a deeper level. This is directly relevant for Jac's unique constructs like walkers, nodes, and abilities.
- **More robust error understanding**: the model has been trained on debugging tasks and can reason about compiler errors. This should help with the debugging component of the test data (800 examples).
- **Code completion instincts**: when generating Jac code, the model's default behavior is to produce syntactically complete, structurally sound code. General-purpose models sometimes produce code-like text that is not actually valid code.

### Strong HumanEval and MBPP performance

While these benchmarks test Python code generation, they serve as proxies for general code generation capability. Strong performance indicates the model can translate natural language task descriptions into working code — the same capability needed for NL-to-Jac generation, just with a different target language.

### Efficient MoE routing

The ~3B active parameters per token (slightly less than Gemma's 3.8B) means Qwen3-Coder may actually be slightly faster for both training and inference on the M5 Pro. Each forward and backward pass processes fewer active parameters, reducing compute time per step. Over 1,875 training steps and hundreds of evaluation tasks, this adds up.

---

## Known risks for Jac finetuning

### Less agentic training than Gemma

Qwen3-Coder is optimized for code tasks, but it has not undergone the same agentic training that gives Gemma 4 its tau2-bench score of 86.4%. For the multi-turn conversation and debugging trajectory components of the test data, Qwen3-Coder may be at a disadvantage. Agentic behavior — maintaining context across turns, recovering from errors, planning multi-step actions — is a distinct capability from code generation, and Qwen3-Coder may not have it to the same degree.

This is a particular concern for the 500 multi-turn conversation examples in the test data and for the debugging examples (800) that require the model to reason through compiler error messages and propose iterative fixes.

### Tokenizer efficiency for Jac syntax

Qwen3-Coder uses a different tokenizer than Gemma 4, with a different vocabulary and different subword segmentation. The tokenizer's efficiency on Jac syntax — how many tokens it needs to represent Jac keywords, operators, and constructs — directly affects both training cost and inference speed. If the Qwen tokenizer fragments Jac keywords into many subword tokens, each example requires more tokens to represent, reducing the effective information per training step.

To assess this before training:

```python
from mlx_lm import load

_, tokenizer = load("./models/qwen3-coder-30b-q4")

# Test tokenization of Jac-specific constructs
jac_snippets = [
    "walker visit_all :ability: {",
    "node MyNode :has: name: str, value: int;",
    "edge connects :has: weight: float;",
    "can do_something with entry {",
    "disengage;",
    "visit [-->];",
]

for snippet in jac_snippets:
    tokens = tokenizer.encode(snippet)
    print(f"{snippet}")
    print(f"  Tokens: {len(tokens)} -> {tokenizer.convert_ids_to_tokens(tokens)}")
    print()
```

Compare the token counts with Gemma's tokenizer on the same snippets. If Qwen consistently uses 20-30% more tokens for the same Jac code, this is a meaningful disadvantage.

### Chat template differences

Qwen3-Coder uses its own chat template format, which differs from Gemma's. The training data must be formatted to match the Qwen template. This is handled automatically by MLX's data loading when using the standard conversation format, but it is worth verifying:

```python
from mlx_lm import load

_, tokenizer = load("./models/qwen3-coder-30b-q4")

# Verify the chat template
messages = [
    {"role": "system", "content": "You are a Jac programming expert."},
    {"role": "user", "content": "Write a walker that visits all nodes."},
    {"role": "assistant", "content": "Here is a Jac walker..."}
]

formatted = tokenizer.apply_chat_template(messages, tokenize=False)
print(formatted)
```

Ensure the formatted output includes proper role markers and separation tokens. If the template does not support a system role, the system prompt must be prepended to the first user message.

### Potential for different MoE routing behavior

Qwen's MoE implementation may use a different routing algorithm than Gemma's (top-k routing with different k values, different load balancing, different expert capacity). These differences can affect how the model responds to LoRA finetuning. If Qwen's routing is more rigid (less willing to shift which experts handle which tokens), the LoRA updates may have less impact on the model's behavior.

This is difficult to predict before training. The main diagnostic is whether the loss curve behaves differently from Gemma's: if Qwen's loss plateau is higher (indicating the model is not learning as effectively from the same data), routing limitations may be contributing.

---

## What to watch during training

### Same monitoring as Gemma, plus Qwen-specific checks

All the monitoring described in the Gemma guide applies equally to Qwen3-Coder:
- Loss curves (training and validation)
- Generated samples at each checkpoint
- Memory usage and throughput

### Additional Qwen-specific monitoring

**Token efficiency per example**: because Qwen's tokenizer may tokenize Jac differently, log the average tokens per training example and compare with Gemma's tokenization of the same data. If Qwen needs significantly more tokens per example, the effective training is on less semantic content per step.

**Thinking mode behavior**: some Qwen3 variants support a "thinking" mode where the model generates internal reasoning before producing the final answer. If this mode is activated by the training data's format, the model may produce thinking traces in its evaluation outputs. This is not necessarily bad (the reasoning traces could improve code quality), but it needs to be accounted for in evaluation: the compiler should only see the final code output, not the thinking trace.

To disable thinking mode explicitly:

```python
# When formatting evaluation prompts, explicitly add:
# /no_think
# to the end of the system prompt, or use the tokenizer's
# enable_thinking=False parameter if supported
```

**Output formatting consistency**: Qwen models sometimes produce markdown-formatted code blocks even when the prompt does not request them. If evaluation outputs are wrapped in triple backticks, the compiler will fail on the backticks, not the Jac code. Ensure the evaluation harness strips markdown formatting before compiler verification.

---

## Post-training evaluation

After training and adapter merging, evaluate the fused Q8 model using the full evaluation suite from [`evaluation.md`](evaluation.md). Qwen-specific evaluation considerations:

1. **Use Qwen's chat template for all evaluation prompts**: mismatched templates will produce garbled output and misleadingly low scores.

2. **Strip thinking traces if present**: if the model outputs `<think>...</think>` blocks before the code, extract only the content after the thinking block for compilation and testing.

3. **Handle markdown code blocks**: strip any ````jac` or ``` ``` wrappers from code outputs before compilation.

4. **Compare tokenizer efficiency**: record the total tokens generated per evaluation task and compare with Gemma. A model that produces correct code in fewer tokens is more efficient for deployment.

5. **Test at temperature 0.0 and 0.7**: temperature 0.0 for deterministic evaluation, 0.7 for diversity assessment.

---

## Qwen3-Coder-30B-A3B — quick reference

```
Download:     huggingface-cli download Qwen/Qwen3-Coder-30B-A3B
Convert Q4:   mlx_lm.convert --hf-path Qwen/Qwen3-Coder-30B-A3B --mlx-path ./models/qwen3-coder-30b-q4 --quantize --q-bits 4
Convert Q8:   mlx_lm.convert --hf-path Qwen/Qwen3-Coder-30B-A3B --mlx-path ./models/qwen3-coder-30b-q8 --quantize --q-bits 8
Train:        mlx_lm.lora --config configs/qwen3_coder_lora.yaml
Fuse:         mlx_lm.fuse --model ./models/qwen3-coder-30b-q8 --adapter-path ./adapters/qwen3-coder-30b-jac --save-path ./models/qwen3-coder-30b-jac-fused-q8
Generate:     mlx_lm.generate --model ./models/qwen3-coder-30b-jac-fused-q8 --prompt "..." --max-tokens 512
```
