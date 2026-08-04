# spectrum-plan.md — Spectrum SNR Layer-Selection SFT Probe

Status: **DONE (2026-08-03).** Both arms trained, evaluated, and compared —
see `reports/2026-08-spectrum-vs-stock-comparison.md` for the full result
(fresh arm: Spectrum significantly beats stock at every stage; cptv2 arm: no
significant difference at SFT, mixed/mostly-not-significant at DPO) and two
real engineering incidents found and fixed along the way (an MLX in-place-merge
zero-weight bug, and a DPO OOM resolved via a shorter max sequence length).

Companion to `spec.md` (umbrella architecture) and `workflow.md` (the three-arm
CPT protocol), sibling to `dpo-plan.md` in role: a probe spec that extends this
phase with one more measurement rather than reopening its umbrella design.
Nothing in `spec.md` §3's locked-decision table, `datagen/spec.md`'s task
catalog, or the frozen dataset/holdout is re-derived or re-litigated here.
Runbook: `spectrum-workflow.md`. Approved design: `docs/superpowers/specs/2026-08-02-spectrum-layer-sft-probe-design.md`.

Prior results this builds directly on: `RESULTS.md` and
`reports/2026-07-cpt-vs-fresh-comparison.md`.

## 1. What this probe tests

The phase's finished result splits cleanly into a behavioral verdict and a
structural one:

| Finding | Source | Number |
|---|---|---|
| CPT-v2 helps the raw base a lot | comparison report, headline table | 47.3% (404/855) vs 10.5% (90/855), +37.0pp, z≈16.75 |
| SFT absorbs almost all of it | comparison report §"Reading the three deltas", RESULTS.md §4.1 | 72.6% (621/855) vs 69.8% (597/855), +2.8pp, z≈1.28, p≈0.20 |
| DPO at this recipe caps at SFT parity | comparison report §3.3/§3.4 | fresh best 69.8% (597/855, step 20) = SFT exactly; cptv2 best 71.7% (613/855, step 40), 0.9pp below its own SFT |
| But CPT-v2's *shape* survives SFT | comparison report §4 (`lora_svd_qproj.py`) | q_proj stable rank at layer 47: CPT-v2 2.70, CPT+SFT 2.83, fresh-SFT 4.42; rank-1 magnitude 1.10 / 1.13 / 0.53 |

§4 is the hook. The CPT-v2-base SFT adapter and the fresh-base SFT adapter reach
statistically indistinguishable pass rates from visibly different q_proj updates
— one rank-1-dominated and high-magnitude, one more spread out. So *where and how
the LoRA capacity lands* is not fixed by the recipe: it is a free variable that
the arms already differ on by accident of lineage, and that the eval cannot
currently see.

This probe makes that variable explicit and controls it. **Layer history**
(CPT-v2 lineage) demonstrably changes adapter geometry without changing measured
capability. Does **layer selection** — choosing *which* 16 of the 48 decoder
blocks receive LoRA, instead of always taking the trailing 16 — do the same
thing, or does it move the functional number too?

Both existing arms hardcode `num_layers: 16`, which `mlx_lm.tuner.utils.
linear_to_lora_layers` resolves as `model.layers[-16:]` — decoder blocks 32-47,
confirmed directly from the trained artifacts: `sft-on-fresh/adapters.safetensors`
holds 256 tensors spanning exactly layers 32-47, and `03-cpt-only/adapters/cpt-v2/
adapters.safetensors` holds the same 256-tensor / layers-32-47 footprint. That
trailing block was never a measured choice; it is `mlx_lm`'s default slice.

Two informative outcomes:

- **Spectrum's picks beat trailing-16 outside noise** → the trailing default was
  leaving capability on the table across this whole phase, and the 69.8%/72.6%
  numbers are a floor set partly by an unexamined default.
- **No difference outside noise** → trailing-16 is vindicated as a selection
  policy on this model/dataset, and the phase closes with one fewer unexamined
  default. Combined with §4's structural finding, a null here would also say
  something sharper: adapter geometry (which §4 shows *does* vary) is not
  predictive of functional outcome on this task distribution, from either
  direction — neither lineage nor placement moves the pass rate.

Either way this is one variable, one arm, and one eval pass against a frozen
holdout that has already scored six other measurements.

## 2. What Spectrum actually is

`cognitivecomputations/spectrum` (co-developed and published by Arcee AI). It is
a *layer-selection* method, not a training method: it ranks a model's weight
matrices by how much of their spectral energy is distinguishable from random
noise, and emits a list of parameters worth training.

Mechanically:

1. For each real-valued 2-D weight matrix `W` of shape `(m, n)`, take the
   singular-value spectrum (equivalently the eigenvalue spectrum of the
   normalized `WWᵀ`).
2. Marchenko-Pastur random matrix theory gives the limiting eigenvalue
   distribution of a matrix whose entries are i.i.d. with variance `σ²`: a bulk
   supported on `[σ²(1−√γ)², σ²(1+√γ)²]` with aspect ratio `γ = m/n`. Everything
   inside that bulk is what a *structureless* matrix of the same shape and scale
   would produce on its own.
3. Singular values above the upper bulk edge are therefore not explainable as
   noise — they are learned structure. The ratio of above-edge energy to
   in-bulk energy is the per-matrix signal-to-noise ratio.
4. `generate_snr_results.py` runs this over a model's weight matrices and writes
   (a) a per-module SNR ranking and (b) an `unfrozen_parameters` YAML naming the
   top-N% of modules.

The published claim (Arcee's own numbers, not reproduced here and not assumed by
this spec): training only the top-SNR parameters is competitive with full
fine-tuning, beats QLoRA on benchmark performance, and cuts roughly 36% of
memory and 42% of training time, with reduced catastrophic forgetting — all
measured against training every layer.

Note the deviation this probe deliberately accepts: Spectrum's published results
come from **full-parameter unfreezing** of high-SNR modules. This probe reuses
only the *selection signal* and applies the existing LoRA recipe to the selected
layers, because changing the tuning method as well would break the one-variable
discipline every other arm in this phase held to. Carried as a named risk in §11.

## 3. What changes, and what does not

**Changes — exactly one thing:** the set of decoder-block indices that receive
LoRA. `model.layers[-16:]` (32-47) becomes an explicit, possibly non-contiguous
16-element index list produced by the SNR scan.

**Stays byte-identical**, read from `sft_fresh_probe/configs/sft.yaml` and
`sft_cptv2_probe/configs/sft.yaml`:

| Key | Value | Note |
|---|---|---|
| `model` | `models/qwen-q4` | 16GB, 4 shards, 4-bit affine `group_size: 64` |
| `data` | `.../sft_{fresh,cptv2}_probe/dataset/sft` | 8100 train / 1428 valid rows; the two arms' `train.jsonl` are MD5-identical (`a85b9ed2cec29686abb6491c7a7065eb`) |
| `fine_tune_type` | `lora` | |
| `num_layers` | `16` | **count** unchanged; only membership changes |
| `lora_parameters.rank` | `16` | |
| `lora_parameters.scale` | `2.0` | |
| `lora_parameters.dropout` | `0.05` | |
| `batch_size` | `1` | |
| `iters` | `8200` | |
| `learning_rate` | `2.0e-5` | |
| `lr_schedule` | `cosine_decay`, `warmup: 820`, `arguments: [2.0e-5, 8200, 1.0e-6]` | |
| `max_seq_length` | `3072` | |
| `save_every` | `820` | |
| `steps_per_eval` | `500` | |
| `steps_per_report` | `50` | |
| `val_batches` | `8` | |
| `seed` | `42` | |
| `mask_prompt` | `true` | |
| `grad_checkpoint` | `true` | |
| `optimizer` | `adam` (mlx_lm default, recorded in `adapters/sft-on-fresh/adapter_config.json`) | |
| `resume_adapter_file` | `null` (fresh arm) / `03-cpt-only/adapters/cpt-v2/adapters.safetensors` (cptv2 arm) | Phase 2 only; see §8.3 |

**Also unchanged:** the per-layer module inventory LoRA touches. Every one of the
48 blocks carries the same 8 LoRA-eligible modules — `self_attn.{q,k,v,o}_proj`,
`mlp.gate`, `mlp.switch_mlp.{gate,up,down}_proj` — because `config.json` sets
`mlp_only_layers: []` and `decoder_sparse_step: 1`, i.e. every block is an MoE
block. `linear_to_lora_layers` builds its `keys` set by walking *all* of
`model.layers`, so the key set is layer-independent and any 16-layer selection
produces the same footprint: 256 tensors, **281.838M trainable parameters
(0.923% of 30532.123M)**, a 1127MB F32 adapter file — exactly what
`results/sft/train.log` reports for the existing runs. This is a hard invariant
the driver self-test asserts (§6.4), and it is what makes the comparison fair:
Spectrum changes placement at constant capacity, not capacity.

**Also unchanged:** the eval holdout (`dataset/sft/valid.jsonl`, 1428 rows / 855
code-graded), the harness (`sft_cptv2_probe/jacgen/eval_functional.jac`), the
toolchain (mlx-lm 0.31.3, mlx 0.31.2, jaclang 0.16.1).

## 4. The SNR scan

### 4.1 Input: the bf16 snapshot, not `models/qwen-q4`

Random-matrix SNR needs real-valued weights. `models/qwen-q4` is 4-bit affine
packed at `group_size: 64`; its per-group quantization grid is itself a
structure-destroying transform on the spectrum, and this repo has already been
burned once by assuming a re-quantized weight preserves a fine-grained signal
(comparison report §3.2: `mlx_lm.fuse` re-quantization erased the SFT LoRA delta
almost entirely, ~15% of packed 4-bit elements changed and the model behaved like
raw base). The scan therefore runs against the bf16 HF snapshot, confirmed
present locally, no download needed:

```
~/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/
  snapshots/b2cff646eb4bb1d68355c01b18ae02e7cf42d120/
    model-00001-of-00016.safetensors … model-00016-of-00016.safetensors
    model.safetensors.index.json      (18,867 tensors)
    config.json, tokenizer.json, chat_template.jinja, …
```

57GB on disk, 16 shards. `models/qwen-q4` is the 4-bit quantization of this same
checkpoint (identical `Qwen3MoeForCausalLM` config: `num_hidden_layers: 48`,
`hidden_size: 2048`, `num_attention_heads: 32`, `head_dim: 128`,
`num_key_value_heads: 4`, `num_experts: 128`, `num_experts_per_tok: 8`,
`moe_intermediate_size: 768`), so a ranking computed on the bf16 weights is a
ranking of the same layers the q4 model trains.

### 4.2 Output

Spectrum emits a per-module SNR ranking plus an `unfrozen_parameters` YAML for a
chosen top-N%. Both are consumed as *data*: this probe never uses the YAML to
drive an unfreeze, only to read the ranking. Exact output filenames and the CLI
flag spelling are TBD, confirmed at scan time against the checked-out revision of
`generate_snr_results.py` — the workflow's Phase 1 checklist requires recording
them verbatim rather than guessing.

Everything from the scan lands under `sft_fresh_probe/spectrum/snr/`, committed:
raw ranking, the YAML, the scan's own stdout log, and the resolved
`spectrum_layers.json` (§5.3). The bf16 snapshot is a cache artifact and is not
copied into the repo.

### 4.3 The MoE problem — unresolved, flagged, not papered over

Spectrum's published design and evaluation are on **dense** decoder models. Its
unit of analysis is a 2-D weight matrix, and its ranking is organized by module
name so that comparable matrices are compared against comparable ones. Qwen3-Coder-30B-A3B
breaks two of those assumptions at once, and the break is concrete and verifiable
from the shard index:

- **The expert matrices dominate the namespace by count.** Of the 18,867 tensors
  in `model.safetensors.index.json`, 18,432 are expert projections —
  `model.layers.<L>.mlp.experts.<E>.{gate,up,down}_proj.weight`, 6,144 of each
  (128 experts × 48 layers). The dense per-layer projections are a rounding
  error next to them: 48 each of `self_attn.{q,k,v,o}_proj.weight` and
  `mlp.gate.weight`.
- **A "layer" here is heterogeneous.** One block contains 4 dense attention
  projections (`q_proj` 4096×2048, `k_proj`/`v_proj` 512×2048, `o_proj`
  2048×4096), one dense router (`mlp.gate` 128×2048), and 384 small expert
  matrices (768×2048 and 2048×768). Their aspect ratios `γ` differ by more than
  an order of magnitude, and only 8 of 128 experts fire per token
  (`num_experts_per_tok: 8`), so an expert's weights are exercised on roughly
  1/16 of the token stream. A per-layer SNR that averages a router, four
  attention projections, and 384 sparsely-activated expert matrices is a
  quantity with no established meaning.

**Open question, explicitly unresolved by this spec:** whether Spectrum documents
any MoE-aware handling (per-expert grouping, activation-frequency weighting, or
an exclusion rule for switched experts), or whether its per-module ranking is
simply applied to whatever 2-D matrices it finds. Nothing read for this spec
resolves it, and it cannot be resolved without reading the checked-out script.
Resolving it is the **first** item in `spectrum-workflow.md` Phase 1's checklist,
gating everything downstream — and §5 specifies a selection rule that is robust to
the answer.

There is also a **namespace mismatch** to handle regardless of the above: the
scan reads HF names (`mlp.experts.<E>.down_proj`), while LoRA targets MLX's
stacked names (`mlp.switch_mlp.down_proj`, one `LoRASwitchLinear` covering all
128 experts — confirmed from the trained adapter's key list). The mapping is
mechanical (all 128 HF `experts.<E>.<proj>` matrices in layer `L` correspond to
the single MLX `layers.L.mlp.switch_mlp.<proj>`), and the selection rule must
apply it explicitly rather than assume name compatibility.

## 5. Layer selection

### 5.1 The aggregation rule

Spectrum ranks *modules*; the LoRA slice is chosen by *layer*. An explicit
aggregation is required, and it is a real design decision, not a formality.

**Primary rule — dense-only, z-scored, mean-aggregated.** For each decoder block
`L ∈ [0, 47]`, take the SNR of its five dense matrices only —
`self_attn.{q,k,v,o}_proj` and `mlp.gate`. Z-score each module type's 48 values
across layers independently (so `q_proj` is compared only against other layers'
`q_proj`, never against `o_proj` at a different aspect ratio), then average the
five z-scores into one per-layer score. Rank descending, take the top 16.

Rationale for excluding the expert matrices from the aggregate:

- Their SNR is the part §4.3 cannot interpret. Including 384 uninterpretable
  numbers per layer and 5 interpretable ones would let the uninterpretable part
  decide the ranking outright.
- The dense projections are exactly what this phase has already probed
  structurally: §4 of the comparison report ran its SVD diagnostic on
  `self_attn.q_proj`, so the selection signal and the existing structural
  instrument are measured on the same matrices.
- LoRA still gets applied to *all 8* module types in every selected layer (§3) —
  exclusion is from the *ranking*, not from training. Capacity is untouched.

**Recorded but not used for selection:** the per-layer mean/median expert SNR,
computed and written into `snr/layer_scores.json` alongside the dense scores. If
the two rankings disagree sharply, that is itself a reportable finding about
Spectrum-on-MoE and belongs in the write-up (§9.3), but it does not change the
picks mid-probe.

**Fallback, if §4.3's investigation shows Spectrum's output is not usable
per-module in a way that supports the above** (e.g. it emits only a
top-N% parameter-name list with no numeric ranking): fall back to
`q_proj`-only ranking — one number per layer, no aggregation, directly
comparable to §4's SVD probe. Decide once, in Phase 1, record the decision and
the reason in `snr/SELECTION.md`, and do not revisit it after seeing any
training result.

### 5.2 Ties and degenerate outcomes

- Exact ties at the 16th position: break by lower layer index, and note it.
- If the top-16 comes out **equal to {32..47}**, the probe is answered before any
  training: Spectrum independently endorses the existing default. Record it, skip
  Phase 1 training, and close (`spectrum-workflow.md` Phase 2's exit).
- If the overlap with {32..47} is large (say ≥13/16), the probe still runs but
  the expected effect size shrinks proportionally; the write-up must report
  overlap alongside the delta so a null is not over-read.

### 5.3 The selection artifact

```json
{
  "layers": [3, 7, 11, 22, 31, 33, 35, 38, 40, 41, 43, 44, 45, 46, 47, 12],
  "count": 16,
  "rule": "dense-z-mean | q_proj-only",
  "source_snapshot": "b2cff646eb4bb1d68355c01b18ae02e7cf42d120",
  "spectrum_rev": "<git sha of cognitivecomputations/spectrum at scan time>",
  "generated": "<timestamp>",
  "overlap_with_trailing16": 11
}
```

(Illustrative `layers` value — the real one comes from the scan.) Written to
`sft_fresh_probe/spectrum/configs/spectrum_layers.json`, committed, and treated as
frozen once Phase 1 training starts. The driver reads this file and nothing else.

## 6. Driver script design — `spectrum_lora_layers.py`

Same pattern as `sft_fresh_probe/dpo_fixed_train.py` and
`03-cpt-only/cpt_train/run_cpt_leg.py`: **compose the installed package's public
API from our own script; never edit `.venv/`.** It survives `pip install
--upgrade` and venv rebuilds by construction and leaves every other recipe in the
repo on the stock code path.

Location (per the approved design spec §5):
`sft_fresh_probe/spectrum/spectrum_lora_layers.py`.

### 6.1 What it replaces, and what it reuses verbatim

`mlx_lm/tuner/utils.py:38-110` is the whole target. Of that function, exactly one
statement changes:

```python
for l in model.layers[-max(num_layers, 0) :]:      # upstream, line 103
```

Everything else is reused *verbatim*, copied into the driver so that a future
upstream change surfaces as a diff here rather than being silently overridden
(the same discipline `FixedDPODataset`'s docstring states):

- the inner `to_lora(layer)` closure, including its `hasattr(layer, "to_lora")`
  fast path and its full type dispatch — `nn.Linear`/`nn.QuantizedLinear` →
  `LoRALinear`, `SwitchLinear`/`QuantizedSwitchLinear` → `LoRASwitchLinear`,
  `nn.Embedding`/`nn.QuantizedEmbedding` → `LoRAEmbedding`, DoRA variants, and
  the `ValueError` for anything else. **The `SwitchLinear` branch is load-bearing
  here**: every one of this model's 48 blocks carries three
  `QuantizedSwitchLinear` expert projections, and they are 6 of the 8 LoRA
  tensors per layer.
- the `config.get("keys", None)` handling and the `get_keys_for_lora(p, m)`
  walker that populates `keys` from *all* of `model.layers` when the config
  supplies none (which is this recipe's case — the YAMLs set no `keys`).
- the trailing `model.named_modules()` pass that LoRA-izes any top-level matching
  modules outside `model.layers`.
- `r`/`scale`/`dropout` sourced from `config` exactly as upstream does.

The replacement loop iterates the explicit index list:

```
for i in LAYER_IDS:            # sorted, deduped, validated 0 <= i < len(model.layers)
    l = model.layers[i]
    ... identical body: named_modules() ∩ keys -> to_lora -> update_modules(tree_unflatten(...))
```

Signature is kept identical — `(model, num_layers, config, use_dora=False)` — so
it is a drop-in for every call site. `num_layers` is not ignored: it is asserted
equal to `len(LAYER_IDS)`, so a config/CLI mismatch fails loudly instead of
silently training a different capacity than the YAML claims.

### 6.2 Where the rebind is applied

Two module attributes, not one. `mlx_lm/lora.py:20` does `from .tuner.utils
import linear_to_lora_layers`, so `train_model` (line 238) resolves the name from
`mlx_lm.lora`'s own globals; rebinding only `mlx_lm.tuner.utils` would leave the
training path on the stock slice while *looking* patched.

| Attribute | Why | In scope |
|---|---|---|
| `mlx_lm.tuner.utils.linear_to_lora_layers` | used by `load_adapters` (line 131), which every `mlx_lm.load(..., adapter_path=)` goes through | yes |
| `mlx_lm.lora.linear_to_lora_layers` | used by `train_model` (line 238) — the SFT training path | yes |
| `mlx_lm_lora.utils.linear_to_lora_layers` (`mlx_lm_lora/utils.py:15,197`) | the DPO package's own conversion path | no — DPO is out of scope (§12); the driver asserts it is *not* being used rather than rebinding it |

After rebinding, delegate: `from mlx_lm.lora import main as _upstream_main;
_upstream_main()`, which parses `sys.argv` exactly as `mlx_lm.lora` does. Every
other flag, the whole YAML, the LR schedule, checkpointing, and
`--resume-adapter-file` behave stock. `run_sft.sh`'s watchdog loop, its
`--iters`/`--adapter-path`/`--resume-adapter-file` calls, and its checkpoint
archiving all work unchanged against the driver as a drop-in for `mlx_lm.lora`.

### 6.3 Upstream guard

Analogous to `dpo_fixed_train.py`'s `_assert_upstream_still_buggy()`, with the
same "fail at import time, never silently" property — but inverted in intent:
that guard checks a bug is *still present*; this one checks the code we are
composing against is *still shaped the way we copied it*. Name it
`_assert_upstream_still_matches()`. It runs before any rebind and raises with an
actionable message on any of:

1. `mlx_lm.tuner.utils.linear_to_lora_layers` is missing → target moved.
2. `inspect.signature(...)` parameter set ≠ `{model, num_layers, config,
   use_dora}` → call contract changed.
3. Whitespace-stripped `inspect.getsource(...)` does not contain
   `formodel.layers[-max(num_layers,0):]` → upstream no longer takes a trailing
   slice; the premise of this driver (and possibly of the whole probe's framing)
   changed.
4. Whitespace-stripped source hash ≠ a **pinned sha256 constant** recorded in the
   driver next to the mlx-lm version it was taken from (0.31.3) → any other edit
   to the function body, including changes to `to_lora`'s type dispatch or
   `get_keys_for_lora`'s type tuple, which the verbatim copy would then be
   silently stale against.
5. `mlx_lm.lora.linear_to_lora_layers is not mlx_lm.tuner.utils.linear_to_lora_layers`
   before rebinding → the import structure changed and §6.2's two-attribute list
   is no longer the complete set.

Failure mode on all five: raise, print the installed mlx-lm version and what to
re-read, exit non-zero. Never warn-and-continue.

### 6.4 Self-test mode (`--verify-layers`)

Mirrors `dpo_fixed_train.py --verify-tokenization`: loads the real model, applies
the rebind, converts, and asserts — printing a PASS/FAIL line and exiting
non-zero on any failure — before any multi-hour run is allowed to start:

| Assertion | Expected |
|---|---|
| set of block indices containing a `LoRALinear`/`LoRASwitchLinear` | exactly `spectrum_layers.json`'s `layers` |
| number of LoRA tensors | 256 (16 layers × 8 modules × {`lora_a`, `lora_b`}) |
| trainable parameter count | 281.838M, 0.923% of 30532.123M — bit-identical to the trailing-16 runs' `train.log`, since every block carries the same module inventory (§3) |
| per-layer module suffixes | the 8 names in the existing adapter: `self_attn.{q,k,v,o}_proj`, `mlp.gate`, `mlp.switch_mlp.{gate,up,down}_proj` |
| `LoRASwitchLinear` count | 3 per selected layer (48 total) — proves the MoE branch fired, not just the dense branch |
| stock control run | with the rebind disabled, the same assertions reproduce `{32..47}` — proves the harness measures what it claims |

A trainable-parameter count that differs from 281.838M means the selection
changed capacity, not just placement, and invalidates the comparison. That single
number is the probe's cheapest correctness check; the workflow gates training on it.

## 7. Eval-time layer reconstruction — the silent-drop trap

This is the highest-risk implementation detail in the probe and it must be
handled before Phase 1 eval, not discovered during it.

`eval_functional.jac:35-38` loads via `mlx_lm.load(mp, adapter_path=adapter)` →
`mlx_lm/tuner/utils.py:113` `load_adapters` → `linear_to_lora_layers(model,
config.num_layers, config.lora_parameters, ...)`, reading `num_layers` straight
out of the adapter's own `adapter_config.json`. With `num_layers: 16` written by
training, **the eval process rebuilds LoRA on layers 32-47 regardless of where
the weights actually came from**, and then line 137 calls
`model.load_weights(..., strict=False)`. `strict=False` silently ignores keys
that match nothing. A Spectrum adapter naming layer 7 would therefore have layer
7 dropped without a warning, and the arm would be scored as a partially-trained
model.

That is the same failure class as the `mlx_lm.fuse` bug in comparison report §3 —
a silent weight loss producing a plausible-looking, completely wrong number — and
this phase already paid for that lesson once.

**Primary fix (no rebind at eval time).** After training, rewrite the spectrum
adapter's `adapter_config.json` so the stock trailing slice covers a *superset*
of the picks:

```
num_layers := 48 - min(spectrum_layers)      # e.g. min=7 -> 41 -> covers layers 7..47
spectrum_layers := [...]                     # extra key, provenance only
```

This is exact, not an approximation. `load_adapters` will convert every layer in
`[min(picks), 47]`; the picks receive their trained weights, and every covered
layer with no saved weights stays at LoRA init, where the delta is **identically
zero** — `mlx_lm/tuner/lora.py` initializes `self.lora_b = mx.zeros(shape=(r,
output_dims))` for `LoRALinear` and `mx.zeros(shape=(num_experts, output_dims,
r))` for `LoRASwitchLinear`, and both `__call__`s compute `y + scale * (x @ a) @ b`,
so `b = 0` contributes nothing. Dropout is inert (`mlx_lm/utils.py` calls
`model.eval()` at load). Extra keys in `adapter_config.json` are harmless —
`load_adapters` builds a `types.SimpleNamespace` and reads only `fine_tune_type`,
`num_layers`, `lora_parameters`.

Cost: instantiating LoRA on the covered-but-untrained layers. Per layer that is
281.838M/16 = **17.615M F32 parameters ≈ 70MB**; the worst case (`min(picks) = 0`
→ all 48 layers) is 845M params ≈ 3.4GB on top of the 16GB q4 base. Training peak
was 27.857GB on this 48GB machine, so eval has headroom, but the number is real
and belongs in the run log.

**Mandatory assertion, regardless of fix:** before scoring, verify that every key
in `adapters.safetensors` matches a parameter that exists in the loaded model —
i.e. the loaded-key count equals the file's 256. `strict=False` will not tell you;
the check must be explicit and must fail the eval, not warn.

**Fallback (rejected as primary).** A wrapper process that rebinds
`mlx_lm.tuner.utils.linear_to_lora_layers` before `mlx_lm.load`. It works, but
`eval_functional.jac` imports `mlx_lm` itself under `jac run`, so applying the
rebind means changing how eval is invoked — weakening comparability with the six
existing measurements for no benefit over the config rewrite. Keep it documented
as the escape hatch if the config rewrite ever fails its assertion.

## 8. Sequencing and the gate

### 8.1 Phase 1 — fresh arm only

Train `sft-on-fresh-spectrum`, compare against the existing `sft-on-fresh`
(69.8%, 597/855). Cheapest possible signal on whether layer selection matters at
all: one training run (~1.9-2.6h at the 0.889-1.521 it/s logged for the existing
fresh SFT run, TBD confirmed at run time) plus one full-holdout eval. No new
dataset, no new holdout, no new harness.

### 8.2 The gate — quantitative, reusing this project's existing bar

`workflow.md` §6 already pre-registers the standard: a paired test over identical
holdout items (McNemar / paired bootstrap), clearing 95% significance. The
comparison report applies the unpaired two-proportion z-test at n=855 and declined
to call **+2.8pp (z≈1.28, p≈0.20)** a real effect. Both are reused verbatim; no
new significance bar is invented for this probe.

Phase 2 opens only if **both** hold on the 855-row holdout:

1. **Paired McNemar** on `sft-on-fresh-spectrum` vs `sft-on-fresh` over the
   identical items clears p < 0.05. Pairing is available because both arms answer
   the same items, and per `workflow.md` §6 it roughly halves the detectable
   effect versus comparing marginal rates.
2. **|Δ| ≥ 2.8pp.** This is not an arbitrary threshold: it is exactly the gap
   this project already looked at and declined to call real. An effect smaller
   than the one already rejected as noise cannot be promoted to a real effect here.

For calibration: at n=855 around a 70% pass rate, the unpaired two-proportion
standard error is `sqrt(2 × 0.70 × 0.30 / 855) ≈ 0.0222`, so an *unpaired*
significant difference needs roughly **≥4.3pp**. Pairing brings the practical
detectable effect down toward the 2.8pp floor above — which is why McNemar, not
the marginal z-test, is the deciding test, with the z-test reported alongside for
continuity with RESULTS.md §4.1's table.

**Gray zone (significant but < 4.3pp):** report it, and only then consider a
second seed. The comparison report's first caveat already names the limitation —
every number in this phase is a single holdout measurement per arm, not a
repeated-seed estimate. A single extra seed on the spectrum arm is the cheapest
way to bound run-to-run LoRA-init variance, and it is worth paying *only* in the
gray zone, not by default.

**On a null:** stop. Do not run Phase 2. This matches the phase's own discipline —
comparison report §3.5 tested the β lever once, ruled it out, and did not
re-run it blind at a fourth setting.

### 8.3 Phase 2 — cptv2 arm, and its own unresolved base-composition problem

Gated on §8.2, and carrying a design problem Phase 1 does not have.
`sft_cptv2_probe/configs/sft.yaml` seeds SFT from
`03-cpt-only/adapters/cpt-v2/adapters.safetensors` via `resume_adapter_file`,
which `mlx_lm/lora.py:250` applies as `model.load_weights(file, strict=False)`
**after** conversion. That CPT-v2 adapter covers exactly layers 32-47 (verified:
256 tensors, same 8 module suffixes, rank 16 / scale 2.0). So if the Spectrum
picks are not a superset of {32..47} — and by construction they cannot be, since
both sets have exactly 16 members and the picks are only interesting when they
differ — then every non-picked CPT-v2 layer is silently dropped, and the "cptv2
arm" would be running on a base that has *lost* part of its CPT-v2 delta. That is
not the same treatment `sft-on-cptv2` received, and comparing them would be
invalid.

Recommended resolution, to be finalized when (and only when) Phase 2 opens:

1. Convert the **union** `picks ∪ {32..47}` to LoRA.
2. Load the CPT-v2 weights (all 256 keys now land — assert the count).
3. **Freeze** the LoRA modules in `{32..47} \ picks` so they act as a frozen part
   of the base rather than trainable parameters; the picks stay trainable, and
   picks inside 32-47 are initialized from CPT-v2 exactly as the incumbent arm's
   were.
4. Assert trainable params still equal 281.838M — the frozen union members must
   not count.
5. **Post-training merge:** `mlx_lm/tuner/trainer.py:372,385` saves only
   `dict(tree_flatten(model.trainable_parameters()))`, so the frozen CPT-v2 keys
   would be absent from the saved adapter and the delta would vanish again at eval.
   The frozen keys must be merged back into the saved `adapters.safetensors`, and
   `num_layers` set to cover the union per §7.

This is specified now so it is not improvised under time pressure later, but it is
explicitly a Phase-2 item. If Phase 1 nulls, none of it gets built.

## 9. Eval plan

### 9.1 Same holdout, same harness, same invocation

No new eval is written. Reuse `sft_cptv2_probe/jacgen/eval_functional.jac`
(env-var driven, `mlx` mode, `mlx_lm.batch_generate`, `max_tokens=768`) against
the same `dataset/sft/valid.jsonl` — 1428 rows, 855 code-graded — that produced
every number in RESULTS.md. The invocation pattern is copied from
`sft_fresh_probe/eval_sft_sweep.sh`:

```
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL=models/qwen-q4 \
JAC_EVAL_ADAPTER=<adapter dir> JAC_HOLDOUT=<.../dataset/sft/valid.jsonl> \
JAC_EVAL_METRICS_OUT=<results dir>/metrics_functional.jsonl JAC_EVAL_STEP=<step> \
  jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac
```

with `JAC_EVAL_LIMIT=$SUBSET` (100) for the interim per-checkpoint sweep and no
limit for the full 855-row runs. `eval_sft_sweep.sh`'s structure carries over
directly: base row, per-checkpoint subset sweep over
`adapters/<name>/checkpoints/*_adapters.safetensors` (true-global-step named by
`run_sft.sh`'s watchdog, no offset correction needed), then the final full-holdout
run.

### 9.2 Rows reported

| Row | Comparator | Purpose |
|---|---|---|
| `sft-on-fresh-spectrum` final (8200), full holdout | `sft-on-fresh` 69.8% (597/855) | the headline delta and the §8.2 gate |
| per-checkpoint subset curve (10 checkpoints, `save_every: 820`) | fresh arm's own curve (820:43 … 4100:72 … 8200:72, comparison report §2.1) | did placement change the *shape* of learning, not just the endpoint |
| paired McNemar contingency table | item-level, both arms | the deciding statistic |
| trainable-param count + LoRA'd layer set from `--verify-layers` | 281.838M / the picks | provenance that the arm trained what it claims |

### 9.3 Structural read-out (cheap, reuses an existing script)

Run `lora_svd_qproj.py` on `sft-on-fresh-spectrum` and add it as a fourth arm
alongside the three already reported in comparison report §4 (`CPT-v2 (no SFT)`,
`Base+SFT (fresh)`, `CPT+SFT (cptv2)`). The existing q_proj layer sample there is
32/38/44/47; the intersection with the Spectrum picks is whatever it is, and only
the intersecting layers are directly comparable — say so rather than aligning the
plots by rank position. This costs one script run and directly extends §4's open
follow-up ("not yet checked: whether this holds for the other 7 projection types").

## 10. Directory and file layout

Follows the approved design spec §5, with names matching the conventions already
on disk (`sft-on-fresh`, `sft-on-cptv2`, `dpo-on-sft-nofuse`, `-best` suffixes;
`results/{sft,dpo,dpo-fixed,dpo-nofuse}/`):

```
model-experiments/04-cpt-sft/
  docs/
    spectrum-plan.md                     # this file
    spectrum-workflow.md                 # runbook
  sft_fresh_probe/
    spectrum/
      spectrum_lora_layers.py            # driver (§6)
      configs/
        sft_spectrum.yaml                # sft.yaml verbatim, adapter_path retargeted
        spectrum_layers.json             # §5.3, frozen once training starts
      snr/
        SELECTION.md                     # aggregation rule chosen + why (§5.1)
        layer_scores.json                # dense + expert per-layer scores
        snr_raw.<ext>                    # Spectrum's own ranking output, verbatim
        unfrozen_parameters.<ext>        # Spectrum's YAML, kept for provenance only
        scan.log
    run_sft_spectrum.sh                  # run_sft.sh with CFG/ADAPTER/RDIR retargeted
    eval_sft_spectrum.sh                 # eval_sft_sweep.sh, same retarget
    adapters/sft-on-fresh-spectrum/      # + checkpoints/, same as sft-on-fresh
    results/sft-spectrum/                # train.log, metrics_functional.jsonl, images/
  sft_cptv2_probe/
    spectrum/                            # Phase 2 only, built if §8.2 opens
    adapters/sft-on-cptv2-spectrum/
    results/sft-spectrum/
```

`sft_spectrum.yaml` differs from `sft.yaml` in exactly one key — `adapter_path`.
`num_layers` stays `16` (the driver cross-checks it against the picks, §6.1).
Nothing in the existing arms' directories is modified.

## 11. Risks and open questions

| # | Risk | Assessment / mitigation |
|---|---|---|
| 1 | **MoE SNR is uninterpretable** (§4.3) | Unresolved. 18,432 of 18,867 tensors are expert projections; a per-layer aggregate over a router + 4 attention projections + 384 sparsely-activated experts has no established meaning. Mitigated by ranking on dense matrices only (§5.1) and recording the expert scores separately. If Phase 1 documents MoE-awareness in Spectrum, revisit and say so. **This is the probe's largest interpretive caveat and must appear in the write-up regardless of outcome.** |
| 2 | **Top-16 ≈ trailing-16** | Informative but cheap: if identical, no training runs at all (§5.2). If overlap is high, effect size shrinks and the write-up must report overlap next to the delta so a null is not over-read as "selection doesn't matter." |
| 3 | **Top-16 disjoint from trailing-16** | Also informative, and the harder case: the eval-time reconstruction (§7) and Phase 2's base composition (§8.3) both get maximally awkward. Both are specified for exactly this case. |
| 4 | **SNR scan memory on a 48GB machine** | Real constraint, unmeasured. The bf16 snapshot is 57GB — larger than RAM. If `generate_snr_results.py` materializes the model via a standard full-precision load, it cannot run here as-is. Mitigation: stream per-tensor with `safetensors.safe_open`, which never holds more than one matrix — the largest single tensor is `model.embed_tokens.weight` at 151936×2048 bf16 ≈ 622MB, and every matrix that matters for §5.1 is ≤17MB (`q_proj` 4096×2048 bf16). **Check the script's load path before running; wall-clock and peak RSS are TBD, measured at scan time.** |
| 5 | **Silent weight drop at eval** (§7) | Highest-severity implementation risk; same failure class as the `mlx_lm.fuse` bug (comparison report §3). Mitigated by the `num_layers` superset rewrite plus a mandatory loaded-key-count assertion. Never rely on `strict=False` being benign. |
| 6 | **Spectrum's published wins are for full-parameter unfreezing, not LoRA** | Accepted deviation (§2). This probe tests whether Spectrum's *selection signal* transfers to a LoRA budget; a null here does not refute Spectrum's own published results, and the write-up must not claim it does. |
| 7 | **Single-run measurement** | Inherited from the phase (comparison report, caveat 1). No seed repeats by default; one extra seed only in the §8.2 gray zone. |
| 8 | **Upstream drift** | `linear_to_lora_layers` is copied verbatim into the driver. The pinned-source-hash guard (§6.3) turns a silent staleness into a loud import-time failure. |
| 9 | **Spectrum output format** | Exact filenames, CLI flags, and whether a numeric ranking (vs. only a top-N% name list) is emitted are TBD — recorded verbatim in Phase 1, with §5.1's fallback rule covering the name-list-only case. |

## 12. Out of scope

Mirrors the approved design spec §6:

- Running the SNR scan or any training — this document and `spectrum-workflow.md`
  are the deliverable for this pass.
- Phase 2 (cptv2 arm) implementation, until Phase 1 clears §8.2's gate.
- Re-litigating dataset, recipe, or holdout choices already locked for 04
  (`spec.md` §3).
- A DPO stage on the spectrum adapter. `mlx_lm_lora`'s own conversion path
  (`mlx_lm_lora/utils.py:197`) would need the same rebind, and comparison report
  §3.4 already establishes DPO at this recipe caps out at SFT parity — there is
  nothing to learn there until the recipe itself changes.
- Varying the LoRA *budget* (`num_layers`, rank, scale). Placement only, at
  constant capacity.
- Full-parameter Spectrum training (unfreezing the selected layers outright),
  which is what Spectrum's own published numbers measure.
