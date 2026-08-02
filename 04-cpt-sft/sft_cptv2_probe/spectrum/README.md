# spectrum/ — Phase-2 scaffolding for the cptv2 arm

Specs: `../../docs/spectrum-plan.md` §8.3 (the design problem this directory
exists to solve), `../../docs/spectrum-workflow.md` Phases 7–8 (runbook).

**This arm is CONDITIONAL.** `spectrum-plan.md` §12 lists it as out of scope
"until Phase 1 clears §8.2's gate" — paired McNemar p < 0.05 **and** |Δ| ≥ 2.8pp
for `sft-on-fresh-spectrum` vs `sft-on-fresh` (597/855, 69.8%). The scaffolding is
built and self-tested so that decision is not made under time pressure; it does
not authorize the run. On a Phase-1 null, nothing here is used.

## What is NOT duplicated here, and why

`snr_scan.py`, `layer_select.py`, `spectrum_lora_layers.py` and
`adapter_config_fix.py` live once, in `../../sft_fresh_probe/spectrum/`, and are
imported from there.

The SNR ranking is a property of the **base model's weights**, which both arms
share — the arms differ only in what they *resume from*. Two copies of the
ranking code, or two copies of the selection, would create two sources of truth
for the one thing that must be provably identical across arms. So:

- `configs/spectrum_layers.json` is a **relative symlink** to the fresh arm's
  frozen selection. A copy could drift; a symlink cannot. There is a test for it.
- `cptv2_spectrum_lora_layers.py` imports the shared driver and reuses its
  upstream-composition guard, its verbatim `to_lora` dispatch, and its reporting
  helpers rather than re-deriving them.

Nothing in `sft_fresh_probe/` is modified by this arm.

## What IS arm-specific

| File | What it does |
|---|---|
| `cptv2_spectrum_lora_layers.py` | §8.3 steps 1–4: convert the **union** `picks ∪ {32..47}`, load CPT-v2 into it, **freeze** `{32..47} \ picks`, assert trainable == 281,837,568 exactly. `--verify-layers` self-test. |
| `merge_frozen_keys.py` | §8.3 step 5: `mlx_lm/tuner/trainer.py` saves only `trainable_parameters()`, so the frozen CPT-v2 blocks are absent from every file training writes. This puts them back, and verifies the result by reading it off disk. |
| `configs/sft_spectrum.yaml` | `../configs/sft.yaml` with `adapter_path` retargeted. **One key differs**, same as the fresh arm's — there is a test asserting exactly that, so the arm-vs-arm contrast stays placement-only. |
| `configs/spectrum_layers.json` | symlink → the fresh arm's frozen picks. |
| `../run_sft_spectrum.sh` | fresh arm's runner + the §8.3 verify gate + the step-5 merge + a merged-resume path (see below). |
| `../eval_sft_spectrum.sh` | fresh arm's eval + the §7 rewrite driven with the **union**, not the picks. |

Tests: `.venv/bin/python -m pytest model-experiments/04-cpt-sft/sft_cptv2_probe/spectrum/ -v`

## DPO stage — added 2026-08-02

Scope reversal, same as the fresh arm's README records: DPO now continues the
*same* Spectrum-picked layers as SFT. On this arm that means the union/freeze/
merge discipline carries straight through, because the lineage is **21 blocks**,
not 16.

| File | What it does |
|---|---|
| `cptv2_dpo_spectrum_train.py` | Injects `make_union_linear_to_lora_layers` into `dpo_spectrum_train.apply_patches(replacement=…)`, so the three rebind sites, both upstream guards, the chat-template fix and the post-condition are shared code. Adds `assert_frozen_blocks_match_cpt_v2` and `--verify-resume`. |
| `../run_dpo_spectrum.sh` | The no-fuse DPO recipe + the union merge on the resume file, on every snapshot *before it is scored*, and on the final adapter; §7 rewrite driven with the **union**. Outputs `adapters/dpo-on-sft-cptv2-spectrum{,-best}`, `results/dpo-spectrum/`. |
| `../eval_dpo_spectrum.sh` | §7 rewrite over the union + the all-keys-present assertion + a frozen-blocks-bit-match assertion on both adapters. |

**Why DPO needs the union too.** The SFT-spectrum adapter it resumes from holds
336 keys over `picks ∪ {32..47}` (16 trained + 5 frozen CPT-v2 blocks merged back
by §8.3 step 5). `mlx_lm_lora/train.py:527` applies `--resume-adapter-file` with
`load_weights(..., strict=False)` **after** conversion, so a picks-only DPO
conversion would silently drop those 80 tensors — the same bug the SFT side
already fixed, one stage later. And `mlx_lm_lora/trainer/dpo_trainer.py:672-685`
saves only `trainable_parameters()`, so every DPO artifact needs the step-5 merge
before anything reads it.

## The problem, in one paragraph

`configs/sft.yaml` seeds SFT from `03-cpt-only/adapters/cpt-v2/adapters.safetensors`
via `resume_adapter_file`, which `mlx_lm/lora.py:250` applies as
`model.load_weights(file, strict=False)` **after** LoRA conversion. That
checkpoint covers exactly blocks 32–47 (verified at runtime, not assumed: 256
tensors, 8 module suffixes, rank 16 / scale 2.0). The Spectrum picks also have
exactly 16 members, so they can never be a superset unless the two sets are
identical — in which case the probe is answered by construction and no training
happens. So a naive "mirror the fresh arm with a different `resume_adapter_file`"
would drop every non-picked CPT-v2 block silently, and the arm would train on a
base that has lost part of its CPT-v2 delta. Same failure class as the
`mlx_lm.fuse` bug in the 2026-07 comparison report §3.

With the real selection, the concrete numbers are:

```
picks   (16) [0, 22, 23, 27, 30, 34, 36, 37, 38, 39, 41, 42, 43, 44, 45, 47]
cpt-v2  (16) [32..47]
union   (21) [0, 22, 23, 27, 30, 32..47]
frozen   (5) [32, 33, 35, 40, 46]     <- converted so CPT-v2 lands, no gradients
```

5 of the 16 CPT-v2 blocks would have been dropped. 80 of its 256 tensors.

## Three places the delta could still be lost, and what stops each

1. **At conversion** — picks-only conversion → `strict=False` drops 80 keys.
   Stopped by converting the union, and by `assert_adapter_keys_present` in
   `--verify-layers`, which raises rather than warns.
2. **At mid-run resume** — `run_sft_spectrum.sh` passes `--resume-adapter-file`,
   which **overrides** the YAML's `resume_adapter_file`. Mid-training the arm's
   own `adapters.safetensors` holds only the trainable picks, so resuming
   straight from it after a crash would silently revert the 5 frozen blocks to
   LoRA init (zero delta) for the rest of the run. Stopped by merging first and
   resuming from the merged file.
3. **At save/eval** — the trainer saves only trainable keys. Stopped by the
   step-5 merge, by `eval_sft_spectrum.sh` refusing to run without
   `results/sft-spectrum/.merge.done`, and by the §7 `num_layers` rewrite being
   driven with the **union** (`min(union) = 0` here, so `num_layers := 48`).

## Launch sequence (do not run before the Phase-1 gate opens)

```bash
# 1. self-test on the real model (loads models/qwen-q4 twice, a few minutes)
.venv/bin/python model-experiments/04-cpt-sft/sft_cptv2_probe/spectrum/cptv2_spectrum_lora_layers.py \
  --verify-layers --model models/qwen-q4 \
  --spectrum-layers model-experiments/04-cpt-sft/sft_cptv2_probe/spectrum/configs/spectrum_layers.json

# 2. training (~2-2.6h). run_sft_spectrum.sh runs the self-test itself and
#    refuses to train unless it PASSes; it then merges step 5 automatically.
CONFIRM_FULL_RUN=1 model-experiments/04-cpt-sft/sft_cptv2_probe/run_sft_spectrum.sh

# 3. eval
model-experiments/04-cpt-sft/sft_cptv2_probe/eval_sft_spectrum.sh
```

Then the four-way table (spectrum-workflow.md Phase 8): fresh trailing 69.8%,
fresh spectrum, cptv2 trailing 72.6%, cptv2 spectrum — paired McNemar vs
`sft-on-cptv2` (621/855), and whether the spectrum effect is the same size in
both arms or interacts with CPT-v2 lineage.

## Standing caveat

Everything in `../../sft_fresh_probe/spectrum/snr/SELECTION.md` applies here
unchanged — same picks, same MoE interpretive caveat, same thin selection
boundary. This arm adds one of its own: with 11/16 picks already inside
`{32..47}`, this arm's *base* differs from the incumbent `sft-on-cptv2`'s only in
that 5 blocks carry CPT-v2 frozen rather than trainable. That is a smaller
perturbation than the fresh arm's, and the write-up should not expect the two
arms' spectrum effects to be the same size for that reason alone.
