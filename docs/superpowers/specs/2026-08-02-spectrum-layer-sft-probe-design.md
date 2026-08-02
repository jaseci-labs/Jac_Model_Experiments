# 04-cpt-sft — Spectrum-layer SFT probe — Design Spec

Status: approved, pre-implementation (spec-only phase).
Date: 2026-08-02.

## 1. Purpose

Test whether Arcee AI's **Spectrum** method (SNR-based layer selection via Marchenko-Pastur
random matrix theory, `cognitivecomputations/spectrum`) changes SFT outcome on this project's
actual recipe — a real, checkable version of "Arcee/Qwen models react better to SFT" (research
found no size-matched Arcee-Qwen model exists to swap in; Spectrum is Arcee's actual relevant
contribution, a technique, not a model). One last addition to `04-cpt-sft` before that phase
closes.

## 2. What changes vs. the existing SFT recipe

Only **which 16 layers** get LoRA'd. `sft_fresh_probe/configs/sft.yaml` and
`sft_cptv2_probe/configs/sft.yaml` both hardcode `num_layers: 16`, which `mlx_lm.tuner.utils.
linear_to_lora_layers` resolves as `model.layers[-16:]` (trailing contiguous block, 32-47).
Spectrum picks the top-16-by-SNR layers instead, which need not be contiguous. Dataset, iters,
LR schedule, rank/scale/dropout, holdout — everything else stays byte-identical to isolate this
one variable, matching every prior arm in this phase.

## 3. Pipeline

1. **SNR scan**: run `cognitivecomputations/spectrum`'s `generate_snr_results.py` against the
   cached bf16 HF snapshot (`~/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct`,
   57GB, all 16 safetensors shards present — confirmed, no download needed) → per-layer SNR
   ranking + an `unfrozen_parameters` YAML. Not the quantized `models/qwen-q4` — SNR/random-
   matrix-theory needs real-valued weights, 4-bit packing would corrupt the signal.
2. **Layer selection**: take Spectrum's top-16-by-SNR layers, holding LoRA capacity constant
   (same budget as the existing recipe — 16 layers — so the only changed variable is *which*
   16, not *how many*).
3. **Driver script** (`spectrum_lora_layers.py`, same compose-the-public-API-not-.venv/ pattern
   as `sft_fresh_probe/dpo_fixed_train.py` and `03-cpt-only/cpt_train/run_cpt_leg.py`):
   replicates `mlx_lm.tuner.utils.linear_to_lora_layers`'s loop but iterates an explicit layer
   index list (Spectrum's picks) instead of the trailing slice `model.layers[-num_layers:]`,
   then delegates to `mlx_lm.lora`'s normal training path so every other CLI flag/config
   behaves exactly as stock.
4. **Train**: identical dataset (`sft_fresh_probe/dataset/sft/` or `sft_cptv2_probe/dataset/
   sft/`), iters (8200), LR schedule, rank 16 / scale 2.0 / dropout 0.05 as the existing
   `sft-on-fresh` / `sft-on-cptv2` adapters — only the layer set differs.
5. **Eval**: same 855-row functional holdout, same harness, as every other 04 stage.

## 4. Sequencing (gated, not both arms at once)

- **Phase 1 — fresh arm only.** New adapter `sft-on-fresh-spectrum` vs. existing `sft-on-fresh`.
  Cheapest signal on whether layer selection matters at all.
- **Phase 2 — cptv2 arm, gated on Phase 1.** Only run `sft-on-cptv2-spectrum` if Phase 1 shows a
  real, outside-noise effect (not just run-to-run LoRA-init variance). No point paying 2x
  compute to confirm a null twice — matches this project's general discipline (e.g. 04's own
  DPO §3.5 stopped at a tested-and-ruled-out lever rather than re-running blind).

## 5. Directory layout

Lives inside `04-cpt-sft` (per explicit user scope — not a new numbered experiment):

```
04-cpt-sft/
  docs/
    spectrum-plan.md        # umbrella spec (this design, expanded to full detail)
    spectrum-workflow.md     # phased runbook: SNR scan -> layer selection -> driver -> train -> eval
  sft_fresh_probe/
    spectrum/
      spectrum_lora_layers.py
      configs/sft_spectrum.yaml
      (adapters/results land under the arm's existing adapters//results/ dirs, tagged
       -spectrum, matching existing -best/-nofuse suffix conventions)
  sft_cptv2_probe/
    spectrum/                # Phase 2 only, built if/when gated open
```

## 6. Out of scope / deferred

- Actually running the SNR scan or any training (spec-only phase, per explicit user scope).
- Phase 2 (cptv2 arm) build, until Phase 1 result is in.
- Re-litigating dataset/recipe choices already locked for 04.

## 7. Provenance

Approved via `superpowers:brainstorming` in-session (2026-08-02) — web research ruled out an
Arcee-Qwen model swap (no size-matched candidate: Arcee-Agent is Qwen2-7B, Virtuoso-Large is
Qwen2.5-72B/too-large, Homunculus is Mistral-Nemo-backbone, AFM-4.5B isn't Qwen-based), then 2
sequential clarifying questions (technique vs. model-swap direction, arm scope/sequencing).
Builds on `04-cpt-sft/RESULTS.md`, `04-cpt-sft/docs/spec.md`, `04-cpt-sft/docs/dpo-plan.md`
(sibling probe-spec convention this mirrors).
