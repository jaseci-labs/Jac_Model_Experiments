# Spectrum results — summary

Distilled from `2026-08-spectrum-vs-stock-comparison.md`. Question: does Arcee's
Spectrum (SNR/Marchenko-Pastur layer selection, 16 non-contiguous blocks) beat
`mlx_lm`'s default trailing-16-block LoRA target? Same recipe, same dataset,
same 855-row holdout, only the target block set changes.

## Headline

| Arm | Stage | Spectrum | Stock (trailing-16) | Δ | p | Significant? |
|---|---|---|---|---|---|---|
| fresh | SFT | 74.7% (639/855) | 69.8% (597/855) | **+4.9pp** | 0.023 | **Yes** |
| fresh | DPO-best | 74.2% (634/855) | 69.8% (597/855) | **+4.3pp** | 0.046 | **Yes** |
| fresh | DPO-final | 72.7% (622/855) | 62.1% (531/855) | **+10.6pp** | <0.0001 | **Yes** |
| cptv2 | SFT | 70.5% (603/855) | 72.6% (621/855) | −2.1pp | 0.335 | No |
| cptv2 | DPO-best | 71.7% (613/855) | 71.7% (613/855) | 0.0pp | 1.000 | No (exact tie) |
| cptv2 | DPO-final | 69.0% (590/855) | 64.9% (555/855) | +4.1pp | 0.072 | Marginal |

**Verdict:** Spectrum wins clean on the **fresh** arm, every stage, growing margin.
On the **cptv2** arm it's a wash — no stage clears p<0.05.

## Why the asymmetry (hypothesis, not proven)

Fresh arm: Spectrum's 16 picks are the only adaptation happening — clean room to
help. Cptv2 arm: CPT-v2 already reshaped blocks 32–47 with a structurally real
(SVD-confirmed) fingerprint that survives SFT; Spectrum's non-contiguous picks
have to fight that pre-existing adaptation instead of starting fresh, which may
blunt the SNR-selection advantage. Flagged as a hypothesis for a future probe.

## Two real bugs found running this

1. **Merge-corruption (cptv2 SFT):** `merge_frozen_keys.py` did `mx.load(--in)` then
   wrote `--out` to the *same path* — MLX's lazy mmap load means unread arrays get
   truncated by the in-place overwrite before they're materialized, silently
   zeroing every trained LoRA key (frozen keys, loaded from a separate file, were
   unaffected — that asymmetry is what pinned the cause). Fix: `mx.eval()` the
   loaded dict before writing. Recovered by re-merging from the untouched raw
   per-checkpoint files.
2. **DPO OOM (cptv2 arm, `DPO_MAXLEN=512`):** Metal GPU wired-memory ceiling hit
   holding both policy+reference models plus this arm's extra ~1.9GB
   union-conversion overhead — not a real RAM shortage (37GB+ free). Fixed by
   dropping to `DPO_MAXLEN=384` (same value the real run's own OOM-recovery
   ladder already uses). **Caveat:** cptv2 DPO numbers above are at maxlen 384,
   not the fresh arm's 512 — some longer pairs may have been truncated,
   unverified pair-by-pair.

## Bottom line

Spectrum layer selection is a real, useful lever **only when there's no prior
CPT adaptation to work around**. Not a reliable lever once CPT has already
reshaped part of the same layer range.

Full detail, per-stage checkpoint sweep tables, and incident writeups:
`model-experiments/04-cpt-sft/docs/reports/2026-08-spectrum-vs-stock-comparison.md`.
