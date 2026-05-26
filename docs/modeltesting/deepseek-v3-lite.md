# DeepSeek-V3-Lite — Model Testing Guide

*Third candidate for Jac finetuning. Smaller MoE variant of DeepSeek-V3, strong code capabilities.*

---

## Model specifications

| Attribute | Value |
|---|---|
| **Full name** | DeepSeek-V3-Lite |
| **Developer** | DeepSeek AI |
| **Total parameters** | ~16-37 billion (variant dependent) |
| **Active parameters per token** | ~2-4 billion (MoE routing) |
| **Architecture** | Mixture-of-Experts Transformer (Multi-head Latent Attention + DeepSeekMoE) |
| **Context length** | 64K-128K tokens |
| **License** | DeepSeek License (permissive, with some restrictions — verify latest terms) |
| **Release date** | 2026 |
| **Code benchmarks** | Competitive with similarly-sized code models |
| **Key innovation** | Multi-head Latent Attention (MLA) for efficient KV cache |
| **Hugging Face ID** | `deepseek-ai/DeepSeek-V3-Lite` (verify exact path) |

> **Note on specifications**: DeepSeek-V3-Lite is a smaller variant of the full DeepSeek-V3 model. Exact parameter counts, architecture details, and benchmark numbers should be verified from the official model card at the time of download. The specifications above are based on available information and may need updating.

---

## Why this model

DeepSeek-V3-Lite earns its place in the comparison because it represents a different lineage of MoE model engineering than Gemma or Qwen, with innovations that may be advantageous for Jac finetuning.

**DeepSeek's code training pedigree.** DeepSeek has established itself as one of the strongest open-weight code model providers. DeepSeek-Coder and DeepSeek-V2 demonstrated that the DeepSeek team's training recipes produce models with exceptional code understanding. DeepSeek-V3-Lite inherits this training lineage. Even as a smaller variant, it benefits from the architectural innovations and training methodology that made DeepSeek-V3 competitive with much larger models.

**Multi-head Latent Attention (MLA).** DeepSeek's MLA architecture compresses the key-value cache, significantly reducing memory usage during both training and inference. For the M5 Pro's 48 GB memory budget, MLA means the KV cache for a 128K context window may fit where standard multi-head attention would overflow. During training on 2048-token sequences, MLA reduces the activation memory footprint, potentially allowing larger effective batch sizes or longer sequences without hitting memory limits.

**DeepSeekMoE architecture.** DeepSeek's MoE implementation uses fine-grained experts with shared expert isolation. This means the model has both shared experts (always active) and routed experts (conditionally active). The shared experts provide a stable base of knowledge while the routed experts specialize. For Jac finetuning, this architecture could be advantageous: the LoRA updates can more effectively target the shared experts (which are always active) to instill Jac knowledge that applies across all inputs.

**Inference efficiency.** The combination of MLA and efficient MoE routing makes DeepSeek-V3-Lite potentially the fastest of the three candidates for inference. If all three models achieve similar accuracy after finetuning, inference speed becomes a deciding factor for deployment — a faster model can process more evaluation tasks in less time and provides a better user experience in the eventual Jac coding assistant.

**Diversity of the comparison.** Including a model from a third provider (Google, Alibaba, DeepSeek) ensures the comparison covers different training data distributions, different tokenizer designs, and different architectural choices. If one architectural approach turns out to be significantly better or worse for Jac finetuning, this diversity reveals it.

---

## Mac M5 Pro feasibility

### Quantization options

| Quantization | Approximate size | Fits in 48GB? | Use case |
|---|---|---|---|
| Q4 (4-bit) | ~8-18 GB (variant dependent) | Yes | Training (LoRA finetuning) |
| Q8 (8-bit) | ~16-37 GB (variant dependent) | Likely yes | Inference evaluation |
| BF16 (full precision) | ~32-74 GB | Depends on variant | May not fit |
| Q6 (6-bit) | ~12-28 GB | Yes | Alternative precision |

**The size range depends on the exact variant of DeepSeek-V3-Lite.** Before committing to this model, download the model card and verify the total parameter count. If the model is on the smaller end (~16B total), it fits easily at all quantization levels. If it is on the larger end (~37B total), Q8 inference becomes tight (similar to Qwen3-Coder).

**Recommended configuration**: Q4 for training, Q8 for evaluation (consistent with the other models). If Q8 is too tight for the specific variant, Q6 provides a good fallback.

### Memory analysis for Q4 training (assuming ~24B total params)

| Component | Memory estimate | Notes |
|---|---|---|
| Q4 model weights | ~12 GB | MoE weights plus shared experts |
| LoRA adapter weights | ~200-400 MB | Rank 16-32 |
| Optimizer states | ~400 MB - 1 GB | AdamW |
| Activations (batch size 1) | ~2-4 GB | MLA reduces this vs standard attention |
| Gradients | ~2-4 GB | Transient |
| Data pipeline | ~1-2 GB | Buffers |
| MLX overhead | ~1-2 GB | Framework |
| **Total** | **~19-26 GB** | |
| **Headroom** | **~22-29 GB** | Comfortable |

### MLA advantage for memory

DeepSeek's Multi-head Latent Attention compresses the KV cache by projecting keys and values into a lower-dimensional latent space before caching. The practical effect is that the KV cache for a 2048-token training sequence may use 2-4x less memory than standard multi-head attention. This is especially beneficial on the M5 Pro where memory is shared between model weights, training state, and KV cache.

During inference evaluation (where longer contexts are used), the MLA advantage is even more pronounced. A standard 8K context KV cache might need 2-4 GB; with MLA, this drops to 0.5-1 GB. This could be the difference between Q8 inference fitting or not fitting in 48 GB.

---

## Setup instructions

### Step 1: Install dependencies

```bash
# Same MLX stack
pip install mlx mlx-lm huggingface-hub

# Verify GPU
python -c "import mlx.core as mx; print(mx.default_device())"
```

### Step 2: Verify model availability and architecture

Before downloading, verify the exact model path and check community MLX compatibility:

```bash
# Search for the model
huggingface-cli search deepseek-v3-lite

# Check the model card for architecture details
# Look for: total params, active params, context length, license terms
```

If the model is not directly available as a single Hugging Face checkpoint, check for community-converted versions:

```bash
# Search for MLX-compatible versions
huggingface-cli search deepseek-v3-lite mlx
```

### Step 3: Download the model

```bash
# Download (adjust path based on actual Hugging Face ID)
huggingface-cli download deepseek-ai/DeepSeek-V3-Lite \
  --local-dir ./models/deepseek-v3-lite-hf \
  --local-dir-use-symlinks False
```

### Step 4: Convert to MLX format and quantize

```bash
# Q4 for training
mlx_lm.convert \
  --hf-path deepseek-ai/DeepSeek-V3-Lite \
  --mlx-path ./models/deepseek-v3-lite-q4 \
  --quantize \
  --q-bits 4

# Q8 for evaluation
mlx_lm.convert \
  --hf-path deepseek-ai/DeepSeek-V3-Lite \
  --mlx-path ./models/deepseek-v3-lite-q8 \
  --quantize \
  --q-bits 8
```

**Important**: DeepSeek's MLA and MoE architecture may require a custom conversion script if `mlx_lm.convert` does not support it out of the box. If conversion fails with an architecture error:

1. Check the MLX-LM GitHub repository for DeepSeek-specific support
2. Search for community conversion scripts: `pip install mlx-lm --upgrade` to get the latest model support
3. Look for pre-converted MLX models on Hugging Face (search `deepseek-v3-lite mlx`)
4. If no conversion path exists, this model is disqualified from the comparison

### Step 5: Verify the model loads and generates

```bash
mlx_lm.generate \
  --model ./models/deepseek-v3-lite-q4 \
  --prompt "Write a simple Jac walker that visits all nodes in a graph." \
  --max-tokens 256
```

---

## LoRA finetuning configuration

### Training configuration file

Create the file `configs/deepseek_v3_lite_lora.yaml`:

```yaml
# DeepSeek-V3-Lite - LoRA finetuning config for MLX
model: "./models/deepseek-v3-lite-q4"
train: true
data: "./data/jac_5k_train"
valid: "./data/jac_5k_valid"
test: "./data/jac_5k_test"

# LoRA configuration
lora_layers: 16            # Last 16 layers
lora_parameters:
  rank: 16                 # Same rank as other models
  alpha: 32                # 2x rank
  dropout: 0.05            # Consistent regularization
  scale: 10.0              # LoRA scaling

# Training hyperparameters
learning_rate: 2.0e-5      # Same LR for fair comparison
lr_schedule: cosine
batch_size: 1              # Memory constrained
grad_accumulation_steps: 8 # Effective batch size = 8
iters: 1875                # Same step count
warmup_steps: 100
weight_decay: 0.01

# Sequence length
max_seq_length: 2048

# Checkpointing
save_every: 250
adapter_path: "./adapters/deepseek-v3-lite-jac"

# Evaluation
val_batches: 50
steps_per_eval: 100

# Seed
seed: 42
```

### DeepSeek-specific LoRA target modules

DeepSeek's MLA architecture uses different attention projection names. Inspect the model to confirm:

```python
import mlx.nn as nn
from mlx_lm import load

model, tokenizer = load("./models/deepseek-v3-lite-q4")

# Print layer names to identify attention projections
for name, module in model.named_modules():
    if any(key in name.lower() for key in ["attn", "mla", "proj"]):
        print(name, type(module))
```

DeepSeek MLA typically uses:
- `self_attn.q_a_proj` and `self_attn.q_b_proj` — two-stage query projection (MLA)
- `self_attn.kv_a_proj_with_mqa` — compressed key-value projection
- `self_attn.kv_b_proj` — key-value expansion
- `self_attn.o_proj` — output projection

For LoRA, targeting `q_b_proj` and `kv_b_proj` is likely the most effective approach, as these are the expansion projections that carry the most representational capacity:

```yaml
lora_parameters:
  keys: ["self_attn.q_b_proj", "self_attn.kv_b_proj"]
```

If MLX does not support custom LoRA targets for DeepSeek's MLA modules, fall back to the default attention projection targeting and verify that the LoRA is actually modifying the correct weights by checking the adapter parameter count.

### Training command

```bash
mlx_lm.lora \
  --config configs/deepseek_v3_lite_lora.yaml
```

### Expected training time

Training time depends on the exact variant size. For a ~24B total / ~3B active parameter variant:

- **Q4 training (5k examples, 3 epochs)**: approximately 3-7 hours
- **Q8 inference evaluation (500 tasks)**: approximately 1.5-3 hours

DeepSeek's MLA should make the per-step forward pass slightly faster than standard attention, as the compressed KV operations involve fewer multiplications. This advantage compounds over 1,875 training steps.

### Merging LoRA adapters

```bash
mlx_lm.fuse \
  --model ./models/deepseek-v3-lite-q8 \
  --adapter-path ./adapters/deepseek-v3-lite-jac \
  --save-path ./models/deepseek-v3-lite-jac-fused-q8 \
  --de-quantize
```

---

## Known strengths for Jac finetuning

### DeepSeek's code training quality

DeepSeek has consistently produced some of the strongest open-weight code models. DeepSeek-Coder, DeepSeek-Coder-V2, and the code capabilities of DeepSeek-V3 have demonstrated that DeepSeek's training methodology — including their approach to code data curation, training curriculum, and loss formulation — produces models that understand code deeply, not just superficially. DeepSeek-V3-Lite inherits this training quality.

For Jac finetuning specifically, the depth of code understanding matters more than breadth. A model that deeply understands programming concepts (type systems, scoping, control flow, data structures) will learn Jac faster than a model with only surface-level code pattern matching, because Jac's novel constructs (walkers, nodes, edges, abilities) are extensions of fundamental programming concepts, not entirely alien features.

### Efficient MoE with fine-grained experts

DeepSeek's MoE implementation uses more experts with smaller individual capacity, compared to the coarser-grained expert designs in Gemma and Qwen. Fine-grained routing means the model can more precisely allocate compute to different input patterns. For Jac finetuning, this could mean that specific experts naturally specialize in Jac-relevant patterns without requiring changes to the routing weights — the LoRA updates to the expert weights are enough to shift the experts' behavior.

The shared expert architecture is particularly interesting: shared experts that are always active provide a stable foundation of knowledge, while routed experts specialize. If the LoRA updates primarily affect the shared experts, the Jac knowledge is consistently available regardless of which routed experts are selected. If the LoRA updates affect the routed experts, the model develops Jac-specialized routing patterns.

### Potential inference speed advantage

The combination of MLA (smaller KV cache, fewer memory accesses) and efficient MoE routing could make DeepSeek-V3-Lite the fastest of the three candidates for inference. On Apple Silicon, where memory bandwidth is the primary bottleneck for large language model inference, MLA's reduced memory footprint directly translates to higher tokens-per-second throughput.

This matters for the eventual deployment: a faster model provides a better developer experience in the Jac coding assistant. If DeepSeek-V3-Lite matches Gemma and Qwen on accuracy but is 30-50% faster for inference, that is a significant practical advantage.

### Multi-head Latent Attention for long-context Jac

MLA's memory efficiency is especially valuable for long-context tasks. In the evaluation suite, some tasks involve multi-file Jac programs or long multi-turn conversations that require understanding code across thousands of tokens of context. MLA allows the model to maintain a full KV cache for these long contexts without exhausting memory, potentially giving DeepSeek-V3-Lite an accuracy advantage on long-context tasks specifically.

---

## Known risks for Jac finetuning

### License considerations

DeepSeek models have historically used various license terms. While recent DeepSeek models have moved toward more permissive licensing, the exact terms for DeepSeek-V3-Lite must be verified before committing to it as the base model. If the license restricts commercial use or requires attribution in ways that are incompatible with the Jaseci ecosystem's distribution model, this model is disqualified regardless of its technical performance.

Check the license by:

```bash
# Download and read the license file
huggingface-cli download deepseek-ai/DeepSeek-V3-Lite LICENSE --local-dir ./models/
cat ./models/LICENSE
```

Key terms to verify:
- Commercial use: is it permitted without restriction?
- Distribution of finetuned weights: is it permitted?
- Attribution requirements: what is required?
- Use restrictions: are there any prohibited use cases?

If the license is not Apache 2.0 or MIT equivalent, carefully assess whether the restrictions are acceptable for the project's needs.

### MLA compatibility with MLX LoRA

DeepSeek's Multi-head Latent Attention is architecturally different from standard multi-head attention. MLX's LoRA implementation may not support applying LoRA adapters to MLA's compressed projections (q_a_proj, q_b_proj, kv_a_proj, kv_b_proj) out of the box. If LoRA can only be applied to the output projection and not the MLA-specific projections, the effective adaptation capacity is reduced, and the model may not learn Jac as effectively as models where LoRA can target all attention projections.

This is the highest-risk technical issue for DeepSeek-V3-Lite. If the dry run on Day 2 reveals that MLX cannot apply LoRA to MLA modules, the options are:

1. Apply LoRA only to the MLP/FFN layers (which are standard and always supported) — this reduces adaptation capacity but may still be sufficient for the 5k-example test
2. Write a custom LoRA implementation for MLA modules — this is feasible but adds development time
3. Disqualify DeepSeek-V3-Lite from the comparison

### Potentially different tokenizer efficiency

DeepSeek uses a different tokenizer (likely based on a custom BPE vocabulary) that may tokenize Jac syntax differently than Gemma's or Qwen's tokenizers. The tokenizer's treatment of Jac-specific keywords (walker, node, edge, ability, can, has, visit, disengage, spawn, report) directly affects training efficiency. If these keywords are consistently split into multiple subword tokens, each training example carries less semantic density.

To assess tokenizer efficiency:

```python
from mlx_lm import load

_, tokenizer = load("./models/deepseek-v3-lite-q4")

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

Compare results with Gemma and Qwen tokenizers. Document the comparison as it feeds into the token efficiency metric in the evaluation.

### Architecture novelty risk

DeepSeek-V3-Lite's architecture (MLA + fine-grained MoE + shared experts) is more novel than Gemma's or Qwen's relatively standard MoE designs. Novelty cuts both ways: it may provide advantages (as described in the strengths section), but it also increases the risk of encountering unexpected behavior during finetuning. Gradient flow through MLA's compressed projections, expert load balancing during LoRA updates, and the interaction between shared and routed experts under finetuning are all less well-studied than standard LoRA-on-MoE.

The 100-example dry run on Day 2 is critical for this model. It will reveal whether the training loop is stable, whether loss decreases as expected, and whether the model's architecture introduces any training pathologies (gradient explosions, expert collapse, routing instability).

### Less community tooling for MLX

While Gemma and Qwen both have extensive MLX community support (conversion scripts, quantization recipes, training guides), DeepSeek model support in MLX is less mature. This means:
- Conversion may require debugging
- Quantization may not use optimal group sizes for DeepSeek's architecture
- LoRA target module selection may require manual investigation
- Community-reported issues and workarounds may be scarce

This is a practical risk, not a fundamental one. If the conversion and dry run succeed, the model is as viable as the others. But the higher setup friction must be accounted for in the timeline.

---

## What to watch during training

### Standard monitoring (same as Gemma and Qwen)

- Loss curves (training and validation) every 100 steps
- Generated samples at each 250-step checkpoint
- Memory usage and throughput

### DeepSeek-specific monitoring

**Expert load balance**: if MLX exposes MoE routing statistics, monitor whether expert loads remain balanced during finetuning. DeepSeek's MoE uses auxiliary losses to maintain load balance during pre-training; LoRA finetuning does not update the routing weights, so the load balancing may drift. If one or two experts become dominant (processing >50% of tokens), the model is effectively collapsing from MoE to a smaller dense model, losing capacity.

**MLA projection behavior**: the compressed KV projections in MLA mean that the model's attention patterns are mediated by a bottleneck. If LoRA is applied to the expansion projections (q_b_proj, kv_b_proj), monitor whether the loss reduction is faster or slower than expected. Faster reduction suggests the bottleneck concentrates the learning signal effectively. Slower reduction suggests the bottleneck is limiting the adapter's ability to shift the attention patterns.

**Shared expert vs. routed expert adaptation**: if the adapter weights can be inspected after training, compare the magnitude of LoRA updates in shared experts vs. routed experts. Larger updates in shared experts indicate the model is encoding Jac knowledge in its always-active pathway. Larger updates in routed experts indicate the model is developing Jac-specialized routing. Either pattern is acceptable, but the information is useful for understanding the model's internal representation.

**Inference output format**: DeepSeek models may produce outputs with specific formatting conventions (e.g., step-by-step reasoning blocks, code blocks with language tags). Document any formatting patterns that need to be stripped for compiler evaluation.

---

## Post-training evaluation

After training and adapter merging, evaluate using the full suite from [`evaluation.md`](evaluation.md). DeepSeek-specific considerations:

1. **Verify the chat template is correctly applied for all evaluation prompts.** DeepSeek's template format differs from Gemma and Qwen; mismatched templates produce meaningless outputs.

2. **Handle any reasoning/thinking output.** If the model produces `<think>` blocks or similar reasoning markers, strip them before compiler evaluation.

3. **Measure inference tokens per second.** MLA should give DeepSeek a speed advantage; quantify it.

4. **Test with both short and long contexts.** MLA's advantage grows with context length. Include evaluation tasks at 512, 2048, and 8192 token context lengths to see if DeepSeek outperforms the others on longer contexts.

5. **Report license compatibility determination.** Before including DeepSeek-V3-Lite in the final decision matrix, confirm that its license is compatible with the project's requirements. If the license is disqualifying, note this prominently in the results.

---

## DeepSeek-V3-Lite — quick reference

```
Download:     huggingface-cli download deepseek-ai/DeepSeek-V3-Lite
Convert Q4:   mlx_lm.convert --hf-path deepseek-ai/DeepSeek-V3-Lite --mlx-path ./models/deepseek-v3-lite-q4 --quantize --q-bits 4
Convert Q8:   mlx_lm.convert --hf-path deepseek-ai/DeepSeek-V3-Lite --mlx-path ./models/deepseek-v3-lite-q8 --quantize --q-bits 8
Train:        mlx_lm.lora --config configs/deepseek_v3_lite_lora.yaml
Fuse:         mlx_lm.fuse --model ./models/deepseek-v3-lite-q8 --adapter-path ./adapters/deepseek-v3-lite-jac --save-path ./models/deepseek-v3-lite-jac-fused-q8
Generate:     mlx_lm.generate --model ./models/deepseek-v3-lite-jac-fused-q8 --prompt "..." --max-tokens 512
```
