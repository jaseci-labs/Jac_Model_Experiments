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

## Phases 1-3: RUN, 2026-08-02

| Phase | Result |
|---|---|
| 1. SNR scan | **DONE** — 240 dense matrices, `--no-experts`, **88s**, **290MB peak RSS**. `snr/snr_raw.json`, `snr/scan.log`. |
| 2. Layer selection | **DONE** — `configs/spectrum_layers.json` is **frozen**. Picks `[0, 22, 23, 27, 30, 34, 36, 37, 38, 39, 41, 42, 43, 44, 45, 47]`, **11/16 overlap** with the stock `{32..47}`, no boundary tie. Read `snr/SELECTION.md` — it is real analysis of this output, including why the marginal picks are soft and how the MoE caveat bites. |
| 3. `--verify-layers` self-test | **PASS** on the real 30B — 256 LoRA tensors, 48 LoRASwitchLinear, **281.838M** trainable (exactly 281,837,568), control reproduces `{32..47}` at the identical count. |

One bug the real self-test caught: `EXPECTED_TRAINABLE_PARAMS` was pinned at
`281_838_080`, an off-by-512 read of mlx_lm's rounded "281.838M" banner. The
exact integer is **281,837,568** (= 17,614,848/block × 16, derived in a test).
The gate could never have PASSed with the old constant.

Note the two percentages are both correct and are not the same number: mlx_lm's
own banner prints `0.923% (281.838M/30532.123M)` against the *dequantized* model,
while `--verify-layers` prints `5.576%` because it divides by
`tree_flatten(model.parameters())` on the q4 model, which counts packed uint32
words. Same numerator.

Because `min(picks) == 0`, the §7 eval-time rewrite lands on its documented worst
case: `num_layers := 48`, i.e. 32 covered-but-untrained blocks, ~564M extra F32
params ≈ 2.2GB on top of the 16GB q4 base. It fits, but it is the maximum.

## NOT run — these are yours, they need the GPU / hours

1. **SNR scan** — already run (above). Reproduce with:
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

2. **Layer selection** — already run (above); `snr/SELECTION.md` is written.
   Reproduce with:
   ```bash
   .venv/bin/python model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/layer_select.py \
     --snr model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/snr/snr_raw.json \
     --layer-scores-out model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/snr/layer_scores.json \
     --selection-out model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/configs/spectrum_layers.json
   ```
   It did not print `picks == {32..47}` (11/16 overlap), so the probe is live.

3. **Self-test** — already run and PASSing (above). Re-run any time; loads
   `models/qwen-q4` twice, a few minutes:
   ```bash
   .venv/bin/python model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/spectrum_lora_layers.py \
     --verify-layers --model models/qwen-q4 \
     --spectrum-layers model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/configs/spectrum_layers.json
   ```
   Must print 281.838M (281,837,568) trainable, 256 tensors, 48
   LoRASwitchLinear, and a control run reproducing `{32..47}` at the same count. A different trainable count means the selection
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

7. **Phase 2 (cptv2 arm)** — scaffolded and self-tested in
   `../../sft_cptv2_probe/spectrum/`, but CONDITIONAL on step 6. It shares this
   directory's modules and symlinks this directory's `spectrum_layers.json`; it
   adds the §8.3 union+freeze base composition and the step-5 merge, because
   `resume_adapter_file` would otherwise drop 80 of CPT-v2's 256 tensors
   silently. See that README before launching anything there.

## Standing caveat

Spectrum's published wins are for full-parameter unfreezing of high-SNR modules,
not LoRA on selected layers, and its ranking is designed for dense decoders —
18,432 of this model's 18,867 tensors are sparsely-activated expert projections
whose SNR has no established interpretation (spectrum-plan.md §4.3, §11 risks 1
and 6). Both belong in the write-up regardless of the result.
