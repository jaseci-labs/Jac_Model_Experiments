# Spectrum layer selection — Qwen3-Coder-30B-A3B-Instruct

Written by hand from the real scan output in this directory. Rule, provenance,
and every number below come from `snr_raw.json` / `layer_scores.json`, not from a
template.

## What was run

```
snr_scan.py --snapshot .../snapshots/b2cff646eb4bb1d68355c01b18ae02e7cf42d120 --no-experts
layer_select.py --k 16 --rule dense-z-mean
```

| | |
|---|---|
| Snapshot revision | `b2cff646eb4bb1d68355c01b18ae02e7cf42d120` (bf16 HF, 16 shards, 57GB) |
| Matrices scored | **240** — 5 dense projections × 48 blocks |
| Matrices skipped | **18,432** expert projections (`--no-experts`) |
| Wall clock | **88s** |
| Peak RSS | **290MB** (streaming reader; the 48GB ceiling was never in play) |
| Rule | `dense-z-mean` — spectrum-plan.md §5.1 PRIMARY, not the q_proj-only fallback |
| Tie at the k-th position | `false` (see "the boundary is thin" below — *not* an exact tie, but close enough to matter) |

## The picks

```
ascending   [0, 22, 23, 27, 30, 34, 36, 37, 38, 39, 41, 42, 43, 44, 45, 47]
by rank     [45, 0, 39, 43, 41, 34, 37, 44, 38, 42, 47, 30, 22, 36, 27, 23]
```

## Versus the stock trailing-16 (`{32..47}`)

**Overlap: 11/16.** The two sets are neither identical (which would have ended the
probe by construction, spectrum-workflow.md Phase 2) nor disjoint.

| | Layers | Their dense z-mean |
|---|---|---|
| Dropped from the default | `32, 33, 35, 40, 46` | −0.069, −0.244, −0.058, −0.077, **−0.498** |
| Added by Spectrum | `0, 22, 23, 27, 30` | **+1.036**, +0.282, +0.037, +0.122, +0.289 |

Mean z-mean of the picks **+0.598** vs the trailing-16's **+0.429**.

Two things are worth stating plainly:

- **Layer 46 ranks 44th of 48** (z = −0.498) — its k_proj is high (1.052) but its
  o_proj is the second-lowest in the model (0.257) and its q_proj is the single
  lowest (0.623). The stock recipe has been spending 1/16th of its LoRA budget on
  one of this model's least-structured blocks. That is the clearest single result
  of the scan, and it is independent of every judgement call below.
- **Layer 0 ranks 2nd** (z = +1.036), on the highest o_proj in the model (1.356)
  and the 3rd-highest router score (2.299). A trailing slice can never reach it.
  This has a real downstream cost: `min(picks) = 0` puts the §7 eval-time
  reconstruction at its documented worst case — `num_layers := 48 − 0 = 48`, i.e.
  every block gets LoRA rebuilt at load, ~845M extra F32 params ≈ 3.4GB on top of
  the 16GB q4 base. That fits on this machine, but it is the maximum, not a
  typical case, and both arms' eval scripts inherit it.

## The boundary is thin — 3 of the 16 picks are effectively coin flips

Ranked z-means around the cut:

```
 15. layer 27   +0.122
 16. layer 23   +0.037   <- last pick
 ------------------------ cut
 17. layer 31   +0.035   <- displaced by 0.002
 18. layer 28   +0.020
 19. layer 25   −0.034
 20. layer 35   −0.058   <- a trailing-16 member
```

`tie_at_boundary` is `false` only because the floats are not bit-identical. Layers
23, 27 and 30 are separated from the layers they displaced by less than 0.17 z,
and layer 23 by 0.002. The *top* of the ranking (45, 0, 39, 43, 41) and the
*bottom* (46, 33) are decisive; the marginal picks are not. If this probe returns
a small effect, that is the first thing to check — a re-run with the q_proj-only
fallback rule would very likely reshuffle the last three slots and nothing else.

## Does the MoE caveat materially affect confidence in the picks? Yes — more than §4.3 alone implies

spectrum-plan.md §4.3 / §11 risk 1 flags one problem: 18,432 of 18,867 tensors are
sparsely-activated expert projections with no established SNR interpretation, so
they are excluded from the ranking. That exclusion is the right call and is not in
question. But the scan surfaces two further, concrete effects that the plan could
not have known before the numbers existed:

**1. The ranking signal and the training target barely overlap.** Selection is
decided by 5 matrices per block. LoRA is then applied to all 8 module types in the
selected block, three of which (`mlp.switch_mlp.{gate,up,down}_proj`) are the
stacked expert projections that carry the overwhelming majority of the block's
parameters and were *never scored*. We are ranking blocks by the structure of
their attention and router weights and then training mostly their experts. This
is a coherent thing to do — it is the plan's stated design — but it means a null
result would not be evidence that "SNR-based selection doesn't work", only that
*dense-projection* SNR does not predict where *expert* LoRA capacity is best
spent. The write-up must not collapse those two claims.

**2. The one MoE-adjacent matrix that IS in the ranking is both the most
influential and the least stable term.** `mlp.gate` is the MoE router, 128×2048.
It has the widest raw SNR range of any module type (1.504 → 2.908, median 1.938)
and, because it only has 128 eigenvalues, only **5–10** of them sit above the
Marchenko-Pastur edge. A single eigenvalue crossing the edge moves that layer's
router SNR by ~10–20%. Layer 47's pick is materially a router result: its q_proj
(0.874) and k_proj (0.656) are both below median, and it makes the cut mostly on
the highest router score in the model (2.908). Layer 0 is helped the same way
(2.299). So the two most "unusual" picks both lean on the term with the fewest
degrees of freedom, and that term is MoE machinery.

**3. Unrelated to MoE but visible in the same table: `v_proj` is close to
degenerate as a ranking input.** Median SNR 0.060, and 5 of 48 layers have
*exactly zero* eigenvalues above the bulk edge (n_signal = 0 for layers 15, 26,
27, 28, 40). For most of the model, v_proj's z-score is discriminating between
different flavours of noise floor. It is nevertheless weighted 1/5, equal to
q_proj which has 108–184 signal eigenvalues. The one place it dominates is layer
**45**, whose v_proj (1.777, 58 signal eigenvalues) and k_proj (2.160) are both
extreme outliers — and layer 45 is the #1 pick by a wide margin (z = 2.196 vs
+1.036 for #2). The top pick therefore rests largely on two small (512×2048)
matrices being unusual in one block.

**Net read.** The picks are trustworthy at the extremes and soft in the middle.
Two claims survive every caveat above: layer 46 does not deserve its slot in the
default recipe, and the default cannot reach layer 0. The rest of the 5-layer swap
is a marginal reshuffle. Report the 11/16 overlap next to any delta so a null is
not over-read — with 11 of 16 layers shared, the treatment contrast is only 5
blocks (31% of the budget), and the effect size available to be measured is
correspondingly capped.

## Provenance for both arms

`configs/spectrum_layers.json` is the frozen selection. `sft_cptv2_probe/spectrum/
configs/spectrum_layers.json` is a **relative symlink to this file**, not a copy:
the SNR ranking is a property of the base model's weights, which both arms share,
so the picks are identical by construction and cannot drift. Only the weights each
arm *resumes from* differ (nothing vs. `03-cpt-only/adapters/cpt-v2`); see
spectrum-plan.md §8.3 and `sft_cptv2_probe/spectrum/README.md` for the union +
freeze composition that entails.

Frozen as of the first training launch. Do not regenerate.
