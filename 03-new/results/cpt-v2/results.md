# CPT-v2 results — chart readout

Companion to [`03-new/docs/cpt-2/results.md`](../../docs/cpt-2/results.md) (the authoritative Task 19 narrative and verdict). This file walks through the 20 static matplotlib charts in this folder one at a time with the analysis each one supports. All charts are generated from the source-of-truth JSON in `json/` by `03-new/cpt_train/eval_v2/make_charts.py` — regenerate anytime with:

```
.venv/bin/python3 03-new/cpt_train/eval_v2/make_charts.py
```

An interactive version of the same data (11 linked charts, dark/light themes) also exists as a published artifact; static screenshots of it are `cptv2_dashboard_dark.png` / `cptv2_dashboard_light.png` in this folder.

**Verdict up front: REJECTED.** design.md §6.4's bar needs Track A to beat both base and cpt-v1 by ≥0.03 cosine, and Track B win-or-tie ≥0.50. Neither cleared — see `19_acceptance_gauges.png` and `20_summary_dashboard.png` for the one-glance version, or read on for the full breakdown.

---

## Corpus

### `01_corpus_composition.png` — token composition, v1 → v2

Grouped bars, real numbers pulled from both datasets' `manifest.json` (not the rounded projections in design.md's prose table). Docs dominate both corpora (1.96M → 2.06M tokens) and moved least; code was dropped entirely (0.99M → 0); rehearsal shrunk hard (0.75M → 0.23M, from ~20% of v1's total down to CF-insurance-only). Net corpus size: 3.80M tokens (v1) → 2.41M tokens (v2), a real reduction driven almost entirely by dropping code.

### `02_corpus_share_pie.png` — CPT-v2 share by source

Same v2 numbers as a share-of-total view: docs alone is 85.6% of the corpus. This is the "docs-dominant" design decision made visible — there's very little room in this corpus for anything except doc-derived semantics to move the model.

### `03_curation_verdicts.png` — Fable curation pass

10,157 chunks reviewed pre-pack: 70.3% kept as-is, 4.6% upweighted, 25.1% dropped (noise, boilerplate, near-duplicate, off-topic). A quarter of the raw corpus never made it into training at all — the curation pass was a real filter, not a rubber stamp.

---

## Training (12-leg epoch loop)

### `04_loss_curves.png` — train & val loss per leg

The core training result. Train loss falls monotonically and steeply (−82%, 1.756→0.319). Val loss falls too but with two upticks (legs 9 and 11) against the still-falling train curve — the classic shape that raises "is this starting to overfit" — except leg 12 recovers to *tie* leg 10's best val loss (0.783 vs 0.784) while posting the run's best train loss and a clean CF-check. Read together, that's oscillation around a floor, not a genuine turn toward overfitting past leg 10. Floor/target/ceiling of the stop-loss schedule are marked; training halted at the ceiling exactly as designed, not early and not extended.

### `05_val_loss_delta.png` — leg-over-leg val loss change

The same story from a different angle: bars below zero (green) are legs where val loss improved over the previous leg, above zero (red) are upticks. Legs 2, 4, 5, 6, 8, 10, 12 improve; legs 3, 7, 9, 11 regress slightly. No trend of accelerating regression toward the end — leg 12's delta is negative (an improvement), which is the strongest single piece of evidence that leg 9/11's upticks weren't the start of an overfitting slide.

### `06_lr_schedule.png` — learning rate decay

One coherent cosine schedule across all 12 legs, not 12 independent decay-and-restart cycles — this was a deliberate fix from CPT-v1, which hit a hard 3-epoch ceiling by resetting LR every epoch. Note the true peak is leg 2 (9.90e-6), not leg 1 (8.33e-6) — this is a short warmup before the cosine decay proper begins, worth knowing if eyeballing "leg 1 = peak" by habit.

### `07_leg_duration.png` — per-leg wall-clock time

12 legs, ~82 minutes each on average, ~16.4 hours of wall-clock training total on the M5 Pro. Duration is flat across legs (79–85 min, no leg is a runaway outlier) — the per-leg cost was predictable, which matters for anyone budgeting a similar run.

### `08_train_vs_val_scatter.png` — train loss vs. val loss, colored by leg

Plots each leg as a point in (train loss, val loss) space, colored by leg number (dark = early, bright = late). Early legs cluster top-right (both losses high); late legs cluster bottom-left (both losses low) — a clean, well-behaved training trajectory with no leg jumping off the diagonal band, which is another angle confirming nothing pathological happened at any single leg.

### `09_cf_check_strip.png` — CF-check per leg

16/16 on the general-Python coding regression benchmark, every single leg, no exceptions. This is the guardrail that would have stopped the run early if CPT-v2 was degrading general coding ability while it learned Jac-specific content — it never fired. All 12 `cf_check_*.json` snapshots (full generated code, not just pass/fail) were inspected by hand; all clean and idiomatic.

---

## Track A — cosine similarity to jac-gpt oracle (n=100)

### `10_track_a_means.png` — mean cosine-to-oracle, with required margin

base 0.8051, cpt-v1 0.8126, cpt-v2 0.8133. The dashed red lines mark what cpt-v2 needed to clear (mean + 0.03) to pass this track — it isn't close. cpt-v2 beats base by 0.0081 and cpt-v1 by essentially nothing (0.0007).

### `11_track_a_delta_hist_vs_v1.png` / `12_track_a_delta_hist_vs_base.png` — per-question delta histograms

The vs-cpt-v1 histogram (chart 11) is the more damning of the two: it's centered almost exactly on zero and roughly symmetric — visually, this is what a coin flip looks like, and the paired t-test agrees (t=0.22). The vs-base histogram (chart 12) skews very slightly positive — a real but small effect (t=2.07) — consistent with CPT moving the model's style/vocabulary toward Jac-ish text without CPT-v2 being any better at it than CPT-v1 was a training run ago.

### `13_track_a_win_loss.png` — win/loss counts

cpt-v2 beats base on 55/100 questions (barely over half) and beats cpt-v1 on only 47/100 (under half). The 50/50 reference line makes both bars easy to read against chance.

### `14_track_a_boxplot.png` — score distribution by model

All three models' interquartile ranges nearly overlap completely; medians differ by hundredths of a cosine point. If this were the only evidence, "no meaningful difference between the three models" would be the honest one-line summary — which is exactly what the paired stats confirm numerically.

### `15_track_a_scatter_v1_vs_v2.png` — per-question cpt-v1 vs cpt-v2

Every point is one question, x = cpt-v1's score, y = cpt-v2's score, diagonal = no difference. Points scatter evenly on both sides of the diagonal with no visible skew — the clearest single visual for "these two models are indistinguishable on this metric."

---

## Track B — blind pairwise judge vs. oracle (n=100)

### `16_track_b_outcome_bar.png` / `17_track_b_pie.png` — outcome breakdown

Oracle won 91, cpt-v2 won 7, tied 2. Win-or-tie rate 9% against a required 50%. Where Track A said "too close to call," Track B says "not competitive" — and Track B is the one that actually reads for factual correctness against the source passage rather than embedding-space proximity.

### `18_honest_gap_scatter.png` — where the two tracks disagree

The chart that explains why Track A and Track B tell such different-looking stories. Each point is one question's cpt-v2 cosine score, placed on the lane the blind judge actually decided. The ringed outlier, `b003-q-any-inference`, scored 0.9355 — 2nd-highest of all 100 questions — and still lost the blind judgment outright: cpt-v2 hedged ("may infer or leave as unknown") where the oracle correctly confirmed the passage's actual rule (untyped values are typed `any`). High stylistic similarity to the oracle's answer did not mean the answer was right. This is design.md §6.2's "cosine can reward verbose-but-wrong answers" warning, caught in the data rather than just asserted.

---

## Verdict

### `19_acceptance_gauges.png` — the three gates, side by side

Each panel is drawn to its own scale so the shortfall is legible rather than squashed by Track B's much larger required margin. All three read FAIL. The black vertical line is the threshold; the filled bar is where cpt-v2 actually landed.

### `20_summary_dashboard.png` — one-page capstone

Six panels combining loss curves, Track A means, Track B outcome, corpus composition, all three acceptance gates, and the verdict stamp — everything above compressed onto one image for a single glance.

**Bottom line:** the corpus rebuild, curation pass, and 12-leg epoch-loop training all executed cleanly and as designed (zero CF regression, real val-loss improvement, one coherent LR schedule). None of that changed the outcome — CPT-v2 does not clear design.md §6.4's acceptance bar. Per the same discipline as CPT-v1's original null, that's the honest read, not rounded up.
