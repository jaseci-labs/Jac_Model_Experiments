# SFT-on-CPT-v2 Results

**Question:** Does the rejected CPT-v2 checkpoint provide real downstream value once an SFT stage trains on top of it?

**Setup:** `models/qwen-q4` + CPT-v2 adapter (rank16/layers16, resumed via `--resume-adapter-file`) → SFT LoRA, fresh 85/15 split of `sft_train.jsonl` (8100 train / 1428 holdout after filtering out-of-range-length rows), 8200 iters (~1 epoch), `max_seq_length: 3072`, `grad_checkpoint: true`, batch_size 1.

## Result

Functional pass rate ("runs" = generated code jac-compiles-and-executes) on the full 1428-row holdout, 855 code-graded rows (573 rows are `prose_lexical` — explanation/documentation categories with no code to grade):

| Stage | Overall runs % | n |
|---|---|---|
| CPT-v2 base (no SFT) | **47%** (404/855) | 855 |
| +SFT (final, 8200 iters) | **72%** (621/855) | 855 |

**+25 percentage points, absolute.** A real, large improvement.

## Per-category breakdown

| Category | base | +SFT | Δ |
|---|---|---|---|
| code_gen | 9.6% | 64.5% | +54.9 |
| conversion | 76.4% | 86.6% | +10.2 |
| debug | 69.4% | 86.1% | +16.7 |
| migration | 100.0% | 100.0% | 0.0 |
| trajectory | 55.7% | 59.0% | +3.3 |

The gain is not uniform — `code_gen` (the largest category by row count, 313/855) improves the most by far, going from barely-functional to solidly majority-passing. `migration` was already saturated at 100% pre-SFT (a known ceiling — only 11 real deprecated-syntax examples exist in the whole corpus) so has no room to move. Every other category improves.

## Training

- Loss: train loss dropped smoothly across the full run (see `images/train_loss.png`), val loss tracked alongside it without diverging (`images/val_loss.png`) — no overfitting signal, unlike the DPO stage that followed.
- Peak memory: stable ~27-28GB throughout (well under the 48GB ceiling) once the real config was tuned (`max_seq_length` 4096→3072, `val_batches` 25→8, `grad_checkpoint` enabled after an initial OOM).
- Real incidents during this run (full detail in `.superpowers/sdd/progress.md`): an OOM at the first eval→train transition (fixed via the config changes above), and a NaN loss from `mask_prompt` truncation colliding with a small number of very-long-tail training examples (fixed by filtering any row whose full tokenized length exceeded `max_seq_length`, dropping 67/8167 train rows and 13/1441 valid rows, ~0.8-0.9%). Training resumed cleanly from the last good checkpoint after each fix; final run completed 8200/8200 iters with no further issues.

## Verdict

CPT-v2, despite being rejected on its own Track A/B convergence eval (see `model-experiments/03-cpt-only/results/cpt-v2/results.md`), is **not dead weight**: SFT on top of it produces a substantially better model than doing nothing, with the improvement concentrated in exactly the category (`code_gen`) that most directly exercises whatever CPT-v2 actually learned. This answers the underlying hypothesis question on its own, independent of what happens in any later stage.

**Graphs:** `images/train_loss.png`, `images/val_loss.png`, `images/learning_rate.png`, `images/tokens_per_sec.png`, `images/iters_per_sec.png`, `images/trained_tokens.png`, `images/peak_mem.png`, `images/functional_pass_rate.png` (all in `model-experiments/04-cpt-sft/sft_cptv2_probe/results/sft/`).

*For the full pipeline including the DPO stage (a follow-up that catastrophically regressed this result, and the v2 retry currently in progress with a corrected config), see `results.md` in this same directory.*
