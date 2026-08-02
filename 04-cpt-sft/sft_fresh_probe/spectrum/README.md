# spectrum/ — Phase-1 scaffolding for the Spectrum layer-selection SFT probe

Specs: `../../docs/spectrum-plan.md` (design), `../../docs/spectrum-workflow.md`
(runbook). Plan: `docs/superpowers/plans/2026-08-02-spectrum-layer-sft-probe.md`.

## Built and self-tested

| File | What it does |
|---|---|
| `snr_scan.py` | Marchenko-Pastur bulk-edge SNR per weight matrix, streaming one tensor at a time out of the bf16 snapshot (hand-rolled safetensors header + mmap, BF16→F32 by bit-shift — no numpy bf16 dtype exists and both `safe_open` backends raise on this file). |
| `layer_select.py` | spectrum-plan.md §5.1's primary rule: z-score each dense module type across its 48 layers, mean the five, rank, take 16. Expert scores recorded, not used. `--rule q_proj-only` is the §5.1 fallback. |
| `spectrum_lora_layers.py` | `mlx_lm.lora` drop-in that LoRA-izes an explicit layer list. Five-check upstream guard, two-attribute rebind, `--verify-layers` self-test. |
| `adapter_config_fix.py` | spectrum-plan.md §7: `num_layers := 48 - min(picks)` rewrite + the mandatory all-keys-present assertion `strict=False` would otherwise swallow. |
| `configs/sft_spectrum.yaml` | `../configs/sft.yaml` with `adapter_path` retargeted. Nothing else differs. |
| `../run_sft_spectrum.sh` | `run_sft.sh` retargeted; adds a `--verify-layers` gate ahead of the existing `CONFIRM_FULL_RUN=1` gate. |
| `../eval_sft_spectrum.sh` | `eval_sft_sweep.sh` retargeted; runs the §7 rewrite and the key assertion before any scoring. |

Tests: `.venv/bin/python -m pytest model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/ -v`

## NOT run — these are yours, they need the GPU / hours

`configs/spectrum_layers.json` does not exist yet. It is the scan's output and is
frozen once training starts; nothing in this directory invents one.

1. **SNR scan** (spectrum-workflow.md Phase 1). 57GB snapshot, 18,867 tensors;
   wall-clock and peak RSS are unmeasured and belong in the write-up.
   ```bash
   SNAP=~/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/b2cff646eb4bb1d68355c01b18ae02e7cf42d120
   .venv/bin/python model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/snr_scan.py \
     --snapshot "$SNAP" \
     --out model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/snr/snr_raw.json \
     2>&1 | tee model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/snr/scan.log
   ```
   `--expert-sample 8` scores 8 of 128 experts per layer instead of all of them
   if the full expert pass is too slow; the choice is recorded in the output JSON.
   `--no-experts` skips them entirely (the ranking never uses them). Peak RSS
   measured on one real shard during development: **+84MB** while reading a
   4.00GB shard, so the 48GB constraint is not a factor for the reader itself.

2. **Layer selection** (Phase 2), then write `snr/SELECTION.md` by hand with the
   rule used, the MoE finding, ties, and the overlap:
   ```bash
   .venv/bin/python model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/layer_select.py \
     --snr model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/snr/snr_raw.json \
     --layer-scores-out model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/snr/layer_scores.json \
     --selection-out model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/configs/spectrum_layers.json
   ```
   If it prints `picks == {32..47}`, stop: the probe is answered by construction.

3. **Self-test** (Phase 3) — loads `models/qwen-q4` twice, a few minutes:
   ```bash
   .venv/bin/python model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/spectrum_lora_layers.py \
     --verify-layers --model models/qwen-q4 \
     --spectrum-layers model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/configs/spectrum_layers.json
   ```
   Must print 281.838M / 0.923%, 256 tensors, 48 LoRASwitchLinear, and a control
   run reproducing `{32..47}`. A different trainable count means the selection
   changed capacity, not placement — stop and fix before spending compute.
   `run_sft_spectrum.sh` runs this itself and refuses to train unless it PASSes.

4. **Training** (Phase 4, ~2-2.6h, one continuous process):
   ```bash
   CONFIRM_FULL_RUN=1 model-experiments/04-cpt-sft/sft_fresh_probe/run_sft_spectrum.sh
   ```

5. **Eval** (Phase 5):
   ```bash
   model-experiments/04-cpt-sft/sft_fresh_probe/eval_sft_spectrum.sh
   ```

6. **Gate** (Phase 6): paired McNemar vs `sft-on-fresh` (597/855, 69.8%),
   p < 0.05 AND |Δ| ≥ 2.8pp, decision written to `results/sft-spectrum/GATE.md`
   before any Phase-2 work. On a null: stop.

## Standing caveat

Spectrum's published wins are for full-parameter unfreezing of high-SNR modules,
not LoRA on selected layers, and its ranking is designed for dense decoders —
18,432 of this model's 18,867 tensors are sparsely-activated expert projections
whose SNR has no established interpretation (spectrum-plan.md §4.3, §11 risks 1
and 6). Both belong in the write-up regardless of the result.
