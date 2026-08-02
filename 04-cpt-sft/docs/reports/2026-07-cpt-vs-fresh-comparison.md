# Cross-Arm Comparison — Does CPT-v2 Help Once SFT/DPO Trains On Top Of It?

**Question:** Two arms were trained with the identical SFT+DPO recipe and byte-identical dataset/holdout, differing in exactly one variable: whether a CPT-v2 LoRA adapter sits underneath the base Qwen model before SFT/DPO trains on top. Does that pre-existing CPT-v2 adapter still matter once real task-specific training (SFT, then DPO) lands on top of it?

**Method:** functional pass rate ("runs" = generated code jac-compiles-and-executes) on the identical 855-row code-graded holdout at every stage, same eval harness (`eval_functional.jac`), same jaclang version, for both arms. `sft_cptv2_probe/` (base already carries the CPT-v2 adapter) vs. `sft_fresh_probe/` (base is the unmodified pretrained model) — everything else (dataset, split, SFT hyperparameters, DPO v1 hyperparameters) held identical.

---

## Headline result

| Stage | fresh (no CPT) | CPT-v2 | Δ (cptv2 − fresh) |
|---|---|---|---|
| base (no SFT) | **10.5%** (90/855) | **47.3%** (404/855) | **+37.0 pp** |
| +SFT (final, 8200 iters) | **69.8%** (597/855) | **72.6%** (621/855) | **+2.8 pp** |
| +SFT+DPO v1 (last checkpoint)¹ | **12.0%** (103/855) | **12.3%** (105/855) | **+0.2 pp** |

¹ Iteration counts differ: fresh ran only 40/250 iters (the early-stop collapse gate fired after 2 consecutive failing snapshots); cptv2 ran the full 250. See "DPO stage" below — this doesn't change the reading (both floors are ~12% regardless of how far each got), but the row isn't a like-for-like iteration count.

![CPT-v2 vs fresh, 3-stage comparison](../../results/images/cpt_vs_fresh_overall.png)

**Computed deltas (from `compare_arms.jac`, rounded `runs_pct` values as printed):**

```json
{
  "base_delta_cptv2_minus_fresh": 37.0,
  "sft_delta_cptv2_minus_fresh": 3.0,
  "dpo_delta_cptv2_minus_fresh": 0.0,
  "fresh_base_pct": 10.0,
  "cptv2_base_pct": 47.0,
  "fresh_sft_pct": 69.0,
  "cptv2_sft_pct": 72.0,
  "fresh_dpo_pct": 12.0,
  "cptv2_dpo_pct": 12.0
}
```

(Note: the fresh-arm DPO number picked here — 103/855, 12.0% — is the DPO run's *last checkpoint*, selected by the same "max step wins" logic used for every other stage in this comparison. The fresh arm's *best* snapshot during that run reached 107/855 (12.5%), which would make the DPO delta −0.2 pp instead of +0.2 pp — either way, effectively zero.)

---

## Reading the three deltas honestly

These three numbers are not the same kind of fact, and conflating them would misstate the result:

**1. Base stage: +37.0 pp — real, large, and not remotely close to noise.** 404/855 vs 90/855 on an identical holdout is an enormous, obvious gap. CPT-v2 continued-pretraining clearly did teach the base model *something* that shows up directly in raw functional generation ability, before any task-specific SFT/DPO training touches it. This is the one place in this whole experiment where CPT-v2's own standalone value is undeniable.

**2. SFT stage: +2.8 pp (621/855 = 72.63% vs 597/855 = 69.82%) — real-looking, but does not clear conventional statistical significance at this sample size.** A two-proportion z-test on n=855 vs n=855 gives z ≈ 1.28, two-tailed p ≈ 0.20. That is well short of the conventional p < 0.05 threshold — this gap is consistent with sampling noise on a holdout of this size. It is *directionally* in CPT-v2's favor (both the raw count and the base-stage advantage point the same way), but it would be dishonest to call a p ≈ 0.20 gap a confirmed win. The huge 37-point base-stage gap does **not** survive into a statistically distinguishable post-SFT gap — SFT appears to close nearly all of the pre-training gap by itself, leaving at most a small, statistically unconfirmed residual.

**3. DPO stage: ~0 pp (105/855 vs 103/855, or 105 vs 107 using fresh's best snapshot) — both arms collapsed to the same catastrophic floor, independent of CPT-v2.** Both DPO runs wrecked the SFT model's functional capability (as documented separately in each arm's own results — `sft_cptv2_probe/results/FULL-RESULTS.md` and the fresh-probe's own report). Whatever CPT-v2 contributed upstream is fully swamped by DPO's collapse in both arms; there is no CPT-related signal left to read at this stage.

---

## Verdict

**CPT-v2's contribution is statistically indistinguishable from a fresh base once SFT lands on top of it.** The base-model gap between the two arms is real and large (+37.0 pp, 47.3% vs 10.5%) — CPT-v2 clearly does *something* to the raw pretrained model. But that advantage does not survive as a confirmed post-SFT edge: the +2.8 pp gap after SFT (72.6% vs 69.8%) fails a two-proportion significance test at n=855 (z ≈ 1.28, p ≈ 0.20), and by the DPO stage both arms have collapsed to the same catastrophic floor with a ~0 pp difference between them. Practically: if the downstream product is "base model → SFT (→ DPO)", CPT-v2 is not earning its keep — a fresh base put through the identical SFT recipe lands within noise of the CPT-v2 arm's final functional pass rate, at a fraction of the upstream training cost.

---

## Caveats / limits of this comparison

- The SFT-stage +2.8 pp gap is a single holdout measurement per arm, not a repeated-seed estimate — the significance test above says it's *consistent with* noise, not that it's *proven* to be noise. A genuinely small real advantage that happens to fall short of p<0.05 at n=855 cannot be fully ruled out from this single run each; only a repeated-seed comparison could tighten that further, and that was explicitly out of scope for this experiment.
- Both arms' DPO runs are independently documented as catastrophic collapses (see each arm's own DPO section) — this comparison's DPO-stage delta is a comparison of two failure floors, not a meaningful measure of CPT-v2's marginal value at that stage.
- This comparison is functional-pass-rate only (`runs_pct` on the 855 code-graded rows); it does not re-examine category-level breakdowns or the 573 prose/lexical rows excluded from grading in either arm's original reports.

## Artifacts

- Script: `model-experiments/04-cpt-sft/jacgen/compare_arms.jac`
- Image: `model-experiments/04-cpt-sft/results/images/cpt_vs_fresh_overall.png`
- Raw deltas: `model-experiments/04-cpt-sft/results/cpt_vs_fresh_deltas.json`
- Source data: `sft_cptv2_probe/results/{sft,dpo,dpo-v3}/metrics_functional.jsonl`, `sft_fresh_probe/results/{sft,dpo}/metrics_functional.jsonl`

---

## Phase 2 Addendum — full curves, no early stopping, sweet-spot search

Phase 1's DPO numbers used an early-stop gate (fresh: stopped at 40/250 iters; cptv2's own v1 ran the full 250 blind, with no per-checkpoint eval). This addendum removes the early stop, runs every stage to its full budget on **both** arms, and searches the complete curve for a better stopping point than the endpoint — closing the gap between "what we measured" and "the best that recipe could have produced."

### 2.1 — SFT: was there a better stopping point than the final checkpoint?

Re-read the full interim-checkpoint curve (subset=100, code_gen-only per `eval_functional.jac`'s category-sorted-holdout limitation — same caveat as Phase 1, doesn't change the shape of this finding) already logged from Phase 1's training run for both arms — no retraining needed.

| Arm | Curve (step: pct) | Peak | Final (8200) |
|---|---|---|---|
| fresh | 820:43, 1640:62, 2460:68, 3280:66, 4100:**72**, 4920:65, 5740:70, 6560:70, 7380:69, 8200:**72** | 72% @ step 4100 | 72% (tied) |
| cptv2 | 1640:63, 2460:74, 3280:66, 4100:72, 4920:70, 5740:73, 6560:77, 7380:**79**, 8200:**79** | 79% @ step 7380 | 79% (tied) |

**Finding: in both arms, the peak interim score exactly ties the final checkpoint.** Neither SFT run would have benefited from early stopping — full training already lands on (or matches) its own optimum. This is a clean, unambiguous result: no sweet-spot correction needed for either arm's SFT stage.

### 2.2 — DPO: full 250-iter curves, no early stop, both arms

Reran DPO on both arms from scratch/resume to the full 250-iter budget (`DPO_DISABLE_EARLY_STOP=1`), with per-20-iter snapshot + subset functional eval throughout (cptv2's arm got this instrumentation for the first time — its original v1/v2 runs only ever measured 1-2 points on the curve).

| Step | fresh (subset %) | cptv2 (subset %) |
|---|---|---|
| 20 | 2 | 1 |
| 40 | 2 | 1 |
| 60 | 2 | 1 |
| 80 | 2 | 1 |
| 100 | 2 | 1 |
| 120 | 2 | 1 |
| 140 | 2 | 1 |
| 160 | 2 | 1 |
| 180 | 2 | 1 |
| 200 | 2 | 1 |
| 220 | 2 | 1 |
| 240 | 2 | 1 |
| 250 | 2 | 1 |

**Finding: total, permanent collapse in both arms, with zero variation across all 13 checkpoints.** Every single measurement from step 20 through step 250 is identical within each arm (2% for fresh, 1% for cptv2) — there is no dip-and-recover pattern, no late-training rescue, and critically **no sweet spot to find**: the earliest checkpoint tested (step 20) is exactly as good (bad) as the last (step 250). This confirms Phase 1's early-stop gate made the right call in substance — it just fired two segments later (step 40) than the theoretical earliest point (step 20), costing nothing real either way. Full curves and diagnostics: `sft_fresh_probe/results/dpo/{train.log,metrics_functional.jsonl,WOULD_HAVE_STOPPED.md}`, `sft_cptv2_probe/results/dpo-v3/{train.log,metrics_functional.jsonl,WOULD_HAVE_STOPPED.md}`.

### 2.3 — DPO: full-holdout (855-row) confirmation at both final and "best" checkpoint

Since every subset score tied, "best" reduces to "the first checkpoint" (step 20, the only one that could set the initial max via strict `>`) in both arms. Ran the full 855-row holdout eval at both the true final checkpoint (step 250) and this nominal best (step 20) for both arms:

| Arm | checkpoint | Full-holdout result |
|---|---|---|
| fresh | last (step 250/40)¹ | 12.0% (103/855) |
| fresh | best (step 20) | 12.5% (107/855) |
| cptv2 | last (step 250) | 12.5% (107/855) |
| cptv2 | best (step 20) | 13.5% (115/855) |

¹ fresh's "last" full-holdout eval (103/855) was measured in Phase 1 at the original early-stop point (step 40) — re-confirmed identical by the full 250-iter rerun, since every checkpoint from step 20 onward scores the same on the subset metric; not re-measured on the full 855-row holdout a second time at step 250 specifically, since the subset curve gives no reason to expect a different number.

**Two-proportion z-tests, both cross-arm comparisons at this stage:**
- last vs last: 107/855 (cptv2) vs 103/855 (fresh) → diff = +0.47pp, pooled p = 0.1228, SE = 0.01587, **z ≈ 0.29, p ≈ 0.77**
- best vs best: 115/855 (cptv2) vs 107/855 (fresh) → diff = +0.94pp, pooled p = 0.1298, SE = 0.01626, **z ≈ 0.58, p ≈ 0.57**

Both comparisons are far short of significance — smaller in magnitude and weaker in signal than even Phase 1's own inconclusive SFT-stage gap (z≈1.28, p≈0.20). The striking fact here isn't the small directional edge toward cptv2 — it's that **four independent full-holdout measurements across two arms and two checkpoint choices all land in a tight 12.0–13.5% band**, which reads as noise around a single shared collapse floor, not as two arms behaving differently.

### 2.4 — Final conclusion (Phase 2, supersedes nothing in Phase 1 — reinforces it with the complete picture)

**The fuller, non-early-stopped picture reinforces Phase 1's verdict and closes the door on the one thing Phase 1 couldn't rule out: CPT-v2 provides a real, large, undeniable advantage to the raw base model (+37pp before any training), but that advantage does not survive as a statistically confirmed edge once SFT lands on top (+2.8pp, p≈0.20), and DPO — trained to its full budget with no early stopping in either arm, with every single one of 13 checkpoints searched for a better stopping point — collapses both arms to the same statistical noise band (12.0–13.5% across four full-holdout measurements, largest z≈0.58) regardless of whether CPT-v2 sits underneath. There is no hidden sweet spot in either training stage that changes this picture: SFT already converges to its own optimum by the final iteration in both arms, and DPO fails completely and identically from the earliest checkpoint tested onward. Practically: a fresh base run through the identical SFT (and, separately, DPO) recipe performs indistinguishably from the CPT-v2 arm at every post-training stage — CPT-v2's upstream training cost buys a real pretraining-stage improvement that task-specific training entirely absorbs.**

> **⚠ Superseded by Phase 3 below.** The 12.0–13.5% DPO collapse floor documented in 2.2/2.3 was a measurement artifact, not a property of DPO or of the data. §3 explains the real cause and gives corrected numbers. The CPT-vs-fresh verdict for the base and SFT stages above is unaffected (DPO was the only stage that used a fused checkpoint) — only the DPO-stage conclusion changes.

## Phase 3 — the DPO collapse was never DPO: `mlx_lm.fuse` was silently destroying the SFT weights

A second, independent reviewer flagged the same thing Phase 2 should have caught: a collapse this total (12%), this immediate (already at step 20), this uniform (zero variance across 13 checkpoints and two arms), and this immune to hyperparameter changes doesn't look like a hard optimization problem — it looks like the policy was broken *before* DPO ever touched it.

### 3.1 — First hypothesis (wrong, but real): the DPO chat-template bug

`mlx_lm_lora.trainer.datasets.DPODataset` calls `tokenizer.apply_chat_template(msgs, add_generation_prompt=True)` on already-complete assistant turns, appending a spurious `<|im_start|>assistant\n` after the real `<|im_end|>` on every chosen/rejected sequence. This is a real bug (verified at the token level, fixed via a driver script that rebinds the class — `sft_fresh_probe/dpo_fixed_train.py`, following the same "compose the public API, never patch `.venv/`" pattern as `03-cpt-only/cpt_train/run_cpt_leg.py`). But a full 250-iter confirming rerun with this fix applied **still collapsed to 2% on schedule** — ruling this out as the root cause of the collapse (it was worth fixing regardless, and stays fixed in the runs below, but it was never the reason for the 12% floor).

### 3.2 — Real root cause: fusing a LoRA delta into a 4-bit-quantized model silently drops it

Reading raw generations (not just pass/fail) from a "fixed" run's step-20 snapshot showed the model emitting plain React/JSX for a prompt whose SFT training example was genuine Jac (`def:pub`, `sv import`, `<div>` inside a `cl { }` block) — i.e. it wasn't writing *broken* Jac, it wasn't writing Jac at all. Direct comparison confirmed why:

| generation source | output on a held-out Jac-component prompt |
|---|---|
| `models/qwen-q4` + SFT adapter, unfused | correct Jac (`def:pub IngredientRow(...) { ... }`) |
| `mlx_lm.fuse`'d SFT checkpoint, no adapter | plain React/JSX — indistinguishable from raw base |
| `models/qwen-q4`, no adapter, no fuse at all | plain React/JSX — **near-identical to the fused checkpoint** |

The fused "SFT" model behaves like the untouched base model. `mlx_lm.fuse` dequantizes the int4 base, adds the LoRA delta, and re-quantizes — and re-quantization rounds the fine-grained SFT delta away almost entirely (confirmed at the weight level: ~15% of packed 4-bit elements changed bit-pattern between base and fused, i.e. not a literal no-op, but functionally close to one). Every DPO run before this point — both arms, every β/lr variant, the chat-template fix included — trained on top of `mlx_lm.fuse`'s output, meaning **DPO was training on top of an effectively un-SFT'd base the entire time.** Nothing about the collapse's shape (immediate, total, uniform, hyperparameter-invariant) needed a DPO-specific explanation once this was found — it explains all of it.

### 3.3 — The fix and corrected full-holdout results

Skip fusing entirely. Train DPO directly against `models/qwen-q4` (raw, quantized, untouched), seeding the DPO LoRA from the SFT adapter via `--resume-adapter-file` instead of baking it into weights first (`sft_fresh_probe/run_dpo_nofuse.sh`, `sft_cptv2_probe/run_dpo_v4_nofuse.sh` — both also carry the real chat-template fix from §3.1). Reran both arms to the full 250-iter budget, no early stop, same instrumentation as Phase 2:

**Subset curve (100-row code_gen-only, the training-time gate's own view):**

| step | fresh | cptv2 |
|---|---|---|
| 20 | 73% | 78% |
| 40 | 71% | **79%** (best) |
| 60 | 68% | 73% |
| 80 | 70% | 75% |
| 100 | 64% | 76% |
| 120 | 62% | 71% |
| 140 | 66% | 69% |
| 160 | 62% | 68% |
| 180 | 64% | 66% |
| 200 | 64% | 70% |
| 220 | 63% | 68% |
| 240 | 63% | 68% |
| 250 | 59% | 64% |

Peaks early (step 20 fresh / step 40 cptv2), then drifts down — mild overfitting to the 654-pair DPO training set, not collapse.

**Full 855-row holdout, at the SFT baseline, DPO's own-best checkpoint, and DPO's final (step 250) checkpoint:**

| Arm | SFT baseline | DPO best | DPO final (step 250) |
|---|---|---|---|
| fresh | 69.8% (597/855) | 69.8% (597/855, step 20) | 62.1% (531/855) |
| cptv2 | 72.6% (621/855) | 71.7% (613/855, step 40) | 64.9% (555/855) |

The subset gate's "+4 to +9pp win" was real on code_gen specifically but didn't generalize: on the full 7-category holdout, DPO's best checkpoint **ties SFT in the fresh arm exactly (identical 597/855) and lands 0.9pp below it in cptv2** — both differences are noise at n=855. Continuing DPO to its full 250-iter budget makes things measurably worse in both arms (−7.7pp fresh, −7.7pp cptv2 vs. their own SFT baseline).

### 3.4 — Corrected final conclusion (Phase 3, supersedes §2.2–2.4's DPO-stage verdict only)

**DPO does not collapse once trained against the real SFT policy instead of a re-quantization-destroyed stand-in — but at this recipe (β=0.1, lr=1e-6, sigmoid loss, 654 pairs) it does not beat plain SFT either. Its best checkpoint in either arm is statistically indistinguishable from SFT alone, and training it to completion actively regresses both arms by ~8pp. The base-stage and SFT-stage CPT-vs-fresh comparisons in §1–2 are unaffected (they never touched a fused-then-DPO'd checkpoint) — CPT-v2 still shows a real pretraining-stage edge (+37pp) that SFT alone absorbs to statistical noise (+2.8pp, p≈0.20). The corrected, practical recommendation: if using DPO on this recipe, stop at the earliest checkpoint (~20-40 iters) purely to avoid the later-training regression — but don't expect it to outperform SFT alone until the recipe itself changes (see §3.5 for what's actually worth trying next).**

### 3.5 — Where the next real gain is, if any (not yet run)

The full-holdout numbers say this DPO recipe caps out at SFT parity, not that DPO categorically can't help. Two candidate levers were identified; one was tested:

1. **Higher β (more conservative KL regularization) — tested, ruled out.** Ran a 60-iter β=0.3 (3x) confirming test on the fresh arm, same subset-eval methodology: 72% / 70% / 70% at steps 20/40/60, essentially tracking β=0.1's own curve (73% / 71% / 68%) within noise. Training loss dropped steadily each segment (4.234→2.955→2.151 per-segment mean) while `val_accuracy` sat pinned at 0.888 throughout — the model IS optimizing the preference loss, but that optimization isn't translating into more functional passes, at any β tested. **Beta is not the lever** — not worth a full 250-iter rerun at higher β.
2. **More/better preference data — untested, now the leading hypothesis.** 654 training pairs is thin for DPO. The early-peak-then-decay shape (steady loss improvement, flat-to-declining functional pass rate) is consistent with the model drifting on a small, quickly-exhausted dataset rather than a regularization-strength problem — reinforced by (1) showing regularization strength doesn't change the shape. Requires new preference-pair generation, not just a rerun — flagged for a follow-up, not run in this phase.

## Phase 4 — does CPT-v2's fingerprint survive SFT structurally, even though it washes out behaviorally?

§2.4 and §3.4 establish the behavioral verdict: CPT-v2's advantage is real at the base stage (+37pp) but statistically absorbed by SFT (+2.8pp, p≈0.20). That leaves an open structural question — does the SFT LoRA adapter trained *on top of* the CPT-v2 base end up looking like a fresh SFT adapter internally, or does it still carry a trace of CPT-v2 underneath the identical functional outcome? A same-style SVD probe to `03-cpt-only/cpt_train/eval_v2/lora_svd_analysis.py` (adapter-only, un-merged LoRA A-matrix singular-value spectrum; same stable-rank diagnostic), narrowed to `self_attn.q_proj` at layers 32-47, was run across three adapters: the standalone CPT-v2 adapter (`CPT-v2 (no SFT)`, final leg checkpoint), the fresh-base SFT adapter (`Base+SFT (fresh)`, final checkpoint), and the CPT-v2-base SFT adapter (`CPT+SFT (cptv2)`, final checkpoint), against a synthetic untrained-init reference (`Base`, n=50 mean of random rank-16 draws at mlx-lm's own A-init scale — see script docstring for exactly what this baseline is and isn't).

**Note on provenance:** an earlier draft of this analysis existed only as orphaned output (`results/lora_svd_qproj.json` + 2 PNGs) with no surviving generator script, and this section originally reported numbers read from that file. Rewriting the script from scratch (`model-experiments/04-cpt-sft/lora_svd_qproj.py`, committed, self-tests its own SVD trick against a direct `np.linalg.svd` on a small case before running) and rerunning against the same three real checkpoints produced a **materially different spectral shape** — the orphaned numbers are superseded below and should not be trusted. (Likely cause: the lost script omitted `adapter_config.json`'s `lora_parameters.scale` factor, or read the wrong checkpoint/tensor — impossible to say for certain since it's gone. The corrected numbers below come from applying mlx-lm's actual effective update, `scale * A @ B`, exactly as it gets added to the base weight.)

**What the corrected spectra show:** stable rank (`sum(s)^2 / sum(s^2)`) for the three *trained* adapters sits at **2.7-6.0 out of a 16-dim budget** — a genuinely rank-1-dominated update, the classic trained-LoRA signature (docstring: "near 1 means one direction dominates"). The untrained synthetic baseline, by contrast, sits at essentially full stable rank (~16, flat spectrum) as expected for random noise — visible directly in the line chart as the gray line staying flat around 0.43-0.52 across all 16 ranks while every trained arm's colored line falls off a cliff after rank 1.

| layer | stable rank: CPT-v2 (no SFT) | Base+SFT (fresh) | CPT+SFT (cptv2) | rank-1 magnitude: CPT-v2 | fresh | cptv2 |
|---|---|---|---|---|---|---|
| 32 | 3.06 | 4.91 | 2.96 | 1.14 | 0.66 | 1.37 |
| 38 | 3.35 | 5.18 | 3.25 | 0.98 | 0.53 | 1.11 |
| 44 | 4.11 | 6.04 | 4.29 | 0.87 | 0.47 | 0.93 |
| 47 | 2.70 | 4.42 | 2.83 | 1.10 | 0.53 | 1.13 |

Two consistent patterns across all 8 sampled layers: (1) the fresh-base SFT adapter's stable rank is **noticeably higher** (more spread across the rank budget, ~4.4-6.0) than either CPT-v2-involving adapter (~2.7-4.3) — its update is less concentrated in a single direction; (2) the CPT-v2-base SFT adapter's stable rank and rank-1 magnitude both **track the standalone CPT-v2 adapter closely** (e.g. layer 47: stable rank 2.70 vs 2.83, within noise) while the fresh-SFT adapter's numbers sit clearly apart from both (stable rank 4.42, rank-1 magnitude roughly half). Visible directly in the heatmap as the `CPT-v2` and `CPT+SFT` panels sharing a bright rank-1 column that the `Base+SFT (fresh)` panel's rank-1 column is visibly dimmer than.

**Reading:** SFT training on top of the CPT-v2 base produces a q_proj update whose shape (both magnitude and rank-concentration) resembles CPT-v2's own update far more than it resembles an SFT adapter trained from scratch — CPT-v2's structural fingerprint on q_proj is still there after SFT, even though §2's functional numbers say the two arms are statistically indistinguishable at that stage. This isn't a contradiction of §2.4's verdict (functional performance really does converge) — it says the convergence is behavioral, not mechanistic: SFT reaches the same pass rate from two differently-shaped q_proj updates, one of which visibly inherits CPT-v2's concentration/magnitude signature and one of which doesn't. Not yet checked: whether this holds for the other 7 projection types (o/k/v_proj, mlp gates, MoE projections), or whether it predicts anything at the DPO stage — flagged as a follow-up, not run here.
