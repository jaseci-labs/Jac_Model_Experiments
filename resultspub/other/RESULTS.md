# Python→Jac Conversion Probe — Results

Finetune two small open MoE models on synthetic Python→Jac conversion data (Apple
Silicon, 48 GB, MLX LoRA) and measure base-vs-finetuned on decontaminated holdouts.
Primary metric: **cross-compiled behavioral test-pass rate** (generated Jac compiles,
runs, and matches recorded test cases). Idiom metric: **transpile-similarity** =
`rouge_l(model_output, py2jac(python))` — high = Python-shaped (learned to transpile),
low = idiomatic (rewrote into walker/node/edge Jac).

Models: `Qwen/Qwen3-Coder-30B-A3B-Instruct` · `google/gemma-4-26B-A4B-it`. Same data,
same config (`configs/lora.yaml`: LoRA r16, 600 iters, lr 2e-5, batch 2).

---

## 1. Headline — model comparison

| metric | Qwen3-Coder-30B-A3B | Gemma-4-26B-A4B | winner |
|---|---|---|---|
| function holdout — base | 0% | 0% | tie |
| function holdout — finetuned | **94%** | 93% | tie |
| graph holdout — SFT correct | **46%** | 15% | **Qwen** |
| graph holdout — DPO correct | **61%** | 15% | **Qwen** |
| graph — of-correct idiomatic (SFT→DPO) | 83% → **100%** | 100%* | — |
| graph — transpile-similarity (SFT→DPO) | 0.457 → **0.338** | 0.667 (flat) | **Qwen** |
| graph — constructs/output (SFT→DPO) | 4.5 → **6.75** | 0.0 | **Qwen** |

\* Gemma's 100% idiomatic is only 2 correct of 13 — too few for the rate to mean much.

**Verdict:** functions are a tie; **Qwen wins graph idiom decisively**. DPO only pays off
where SFT already produces idiom (Qwen), not where it doesn't (Gemma). **Pick: Qwen3-Coder.**

---

## 2. Function holdout (150 tasks) — full, with token metrics

| model | stage | runs % | test-pass % | gen tokens (total) | eval tok/s | tokens-to-correct |
|---|---|---|---|---|---|---|
| Qwen | base | 0% (0/150) | **0%** (0/150) | 34,753 | 67 | — |
| Qwen | SFT finetuned | 97% (146/150) | **94%** (141/150) | 16,314 | 63 | 106 |
| Qwen | DPO | 96% (145/150) | 93% (140/150) | 16,168 | 66 | 104 |
| Gemma | base | 0% (1/150) | **0%** (0/150) | 76,800 | 50 | — |
| Gemma | SFT finetuned | 96% (145/150) | **93%** (140/150) | 19,182 | 43 | 126 |
| Gemma | DPO | 96% (145/150) | 93% (140/150) | 19,190 | 51 | 126 |

Notes:
- Both base models produce **zero** runnable Jac → finetuning is what creates the capability.
- Finetuning also makes generation far more concise: Qwen base 34.7k → 16.3k tokens; Gemma
  base **76.8k** → 19.2k (stock Gemma rambled heavily before finetuning).
- DPO does **not** change function behavior (no idiom headroom in function tasks — see §5).

---

## 3. Graph holdout (13 tasks) — the idiom axis

| model | stage | correct % | of-correct idiomatic | py-shaped | avg sim | constructs/output |
|---|---|---|---|---|---|---|
| Qwen | base | 0% (0/13) | — | — | — | 0.0 |
| Qwen | SFT | **46%** (6/13) | 83% (5) | 16% (1) | 0.457 | 4.5 |
| Qwen | **DPO** | **61%** (8/13) | **100%** (8) | 0% (0) | **0.338** | **6.75** |
| Gemma | SFT | 15% (2/13) | 100% (2) | 0% | 0.667 | 0.0 |
| Gemma | DPO | 15% (2/13) | 100% (2) | 0% | 0.667 | 0.0 |

- **Qwen progression**: cannot do graph (0%) → SFT produces it (46%, mostly idiomatic) →
  **DPO lifts both correctness (61%) and idiom (100% idiomatic, sim 0.457→0.338)**.
- **Gemma plateaus at 15%** — struggles to learn the walker/node/edge structure; DPO can't
  push what SFT didn't produce.
- Reference: a fully-idiomatic answer scores sim ≈ **0.26**; a mechanical transpile ≈ **1.0**.

---

## 4. Learning curves (post-training checkpoint sweep, 30-task subset)

One point per saved checkpoint (`save_every: 100`, so 6 points). Function holdout test-pass%.

| iter | Qwen runs% | Qwen pass% | Qwen tok/s | Gemma runs% | Gemma pass% | Gemma tok/s |
|---|---|---|---|---|---|---|
| 100 | 96 | 93 | 40 | 90 | 90 | 58 |
| 200 | 100 | 96 | 49 | 100 | 96 | 58 |
| 300 | 100 | 96 | 74 | 100 | 96 | 55 |
| 400 | 100 | 96 | 49 | 100 | 96 | 47 |
| 500 | 100 | 96 | 55 | 100 | 96 | 58 |
| 600 | 100 | 96 | 79 | 100 | 96 | 59 |

Both models learn function conversion **almost immediately** (≥93% by iter 100) and plateau
— more iters don't help. PNGs: `results/<model>/learning_curve.png`.

---

## 5. Why functions show no idiom gain (the headroom result)

| holdout | idiomatic-vs-transpile similarity | meaning |
|---|---|---|
| function tasks | **0.97** | idiomatic Jac ≈ the transpile (a pure `def factorial` IS its transpile) → no room to improve idiom → SFT can't move it, DPO is a no-op |
| graph tasks | **0.26** | idiomatic Jac (walker/node/edge) diverges hard from the dict-loop transpile → real room → SFT learns it, DPO sharpens it |

The graph tier was built specifically to create an idiom axis the function tier lacks. It
worked: it's where every idiom gain in this report shows up.

---

## 6. DPO training signal (Qwen, the one that moved)

| iter | loss | chosen_r | rejected_r | accuracy | margin |
|---|---|---|---|---|---|
| 140 | 0.011 | +2.86 | −2.99 | 1.000 | 5.85 |
| 150 | 0.004 | +3.14 | −3.71 | 1.000 | 6.85 |
| 200 | 0.003 | +3.29 | −3.46 | 1.000 | 6.74 |

DPO cleanly learned to prefer chosen (idiomatic) over rejected (transpile): accuracy 1.0,
growing margin. On Qwen this translated to better generation; on Gemma the same training
signal did not (too few correct graph outputs to begin with).

---

## 7. Dataset (gitignored; regenerable from the Jac builders)

| artifact | count |
|---|---|
| SFT total | **1,647** |
| — idiomatic core | 147 (24 seed + 84 batch + 8 mined + **31 graph**) |
| — transpile volume (py2jac, behaviorally gated) | 1,500 |
| DPO pairs (incl 31 graph w/ real divergence) | **147** |
| Balanced manifest (1:3 idiom:transpile) | 588 |
| mlx-lm train / valid split | 529 / 59 |
| Function eval holdout | 150 |
| Graph eval holdout (3 idioms, disjoint train/test) | 13 |

SFT difficulty mix: atomic 41 / idiomatic 37 / composed 69. Source corpus:
`Vezora/Tested-22k-Python-Alpaca`. Graph idioms: accumulator-walker over adjacency,
typed/weighted edges (`edge Road{has w}`), linked-list chains.

---

## 8. Setup / cost

| item | value |
|---|---|
| hardware | Apple Silicon, 48 GB unified memory, MLX |
| quant | Q4 (train) + Q8 (eval/fuse) |
| LoRA | rank 16, 600 iters, lr 2e-5, batch 2, num_layers 16 |
| DPO | mlx-lm-lora, LoRA-DPO (reference=frozen base), beta 0.1, 150 iters, num_layers 8, grad-checkpoint, max-seq 384 (fits 48 GB) |
| train peak mem | ~27 GB (Qwen) / ~37 GB (Gemma) |
| per-model wall-clock | ~2–4 hr (download + quantize + train + sweep + fuse + evals) |

---

## 9. Graph index (PNGs)

Per model, in `results/qwen/` and `results/gemma/`:

| file | shows |
|---|---|
| `learning_curve.png` | holdout test-pass % per checkpoint |
| `train_loss.png` / `val_loss.png` | LoRA loss curves |
| `learning_rate.png` | LR schedule |
| `tokens_per_sec.png` / `iters_per_sec.png` | throughput |
| `trained_tokens.png` | cumulative tokens |
| `peak_mem.png` | peak GPU memory |

---

## 10. Bottom line

1. **Our data teaches plain Python→Jac to either model** (0% → ~94%).
2. **Idiomatic (graph-spatial) Jac needs data with idiom headroom** — the graph tier
   delivers it; Qwen learns it (46%), Gemma mostly doesn't (15%).
3. **DPO sharpens existing idiom, can't manufacture it** — Qwen 46%→61% & 100% idiomatic;
   Gemma flat.
4. **Use Qwen3-Coder-30B-A3B for Jac graph idiom.**

Next levers: more graph examples per idiom (lift both models' graph %), denser learning
curve (`save_every: 50`), more structural idioms (trees, weighted-edge traversal). Full
engineering handoff: `docs/modeltesting/HANDOFF.md`.
