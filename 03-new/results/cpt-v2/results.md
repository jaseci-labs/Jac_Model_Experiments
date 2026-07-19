# CPT-v2 analysis

Companion to [`03-new/docs/cpt-2/results.md`](../../docs/cpt-2/results.md) (the Task 19 acceptance narrative). This document is the analysis: what CPT-v2 was built to find out, what the data actually shows, why it shows that, and what to do next. All 22 charts in this folder are cited inline as evidence; regenerate them anytime from the source-of-truth JSON in `json/` via `.venv/bin/python3 03-new/cpt_train/eval_v2/make_charts.py`. An interactive version of the same data also exists as a published dashboard artifact (`cptv2_dashboard_dark.png` / `cptv2_dashboard_light.png` are static captures of it).

## Executive summary

CPT-v2 was a second, independent attempt to make continual pretraining move Jac's semantic ceiling, after CPT-v1 nulled (base and CPT-v1 gave byte-identical answers on a 20-question MCQ). CPT-v2 changed two things at once: the corpus (docs-dominant, Fable-curated, code dropped, rehearsal cut to CF-insurance-only) and the eval (dual-track against a RAG-grounded jac-gpt oracle instead of MCQ). Both changes were real and well-executed — the training mechanism worked cleanly (12 legs, zero CF regression, real loss improvement) — and the result is still a null: **cpt-v2 is statistically indistinguishable from cpt-v1** (Track A, t=0.22) and **loses to jac-gpt on 91 of 100 blind-judged questions** (Track B). Verdict: **REJECTED**, design.md §6.4's bar cleared on 0 of 3 gates.

The value of this run isn't the checkpoint — it's what a *second* null, with two major variables changed, rules out. See "What this actually tells us" below.

---

## 1. What CPT-v2 was testing

CPT-v1's null left two live, non-exclusive hypotheses for why nothing moved:

1. **Corpus dilution** — CPT-v1 mixed docs (1.96M tok) with a 17-repo code corpus (992K tok) and heavy rehearsal (750K tok, ~20% of total); the semantic signal from docs may have been drowned out.
2. **Instrument mismatch** — a 20-question MCQ (constrained choice) may simply be too blunt to detect real learning that open-ended generation, graded against a grounded reference, would show.

CPT-v2 was designed to kill or confirm both at once: drop code entirely, cut rehearsal to ~10%-of-total CF-insurance, add a Fable curation pass to remove noise (`03_curation_verdicts.png`), and replace MCQ with two open-ended tracks scored against a real RAG-grounded reference system (jac-gpt) instead of a fixed answer key.

---

## 2. Finding: corpus dilution wasn't it

`01_corpus_composition.png` and `02_corpus_share_pie.png` show the corpus change is real, not cosmetic — code (992K tok in v1) dropped to zero, rehearsal cut from 750K to 234K tok, docs now 85.6% of a smaller, more concentrated 2.41M-token corpus (down from 3.80M). The curation pass (`03_curation_verdicts.png`) removed a further 25.1% of raw chunks as noise/near-duplicate/off-topic before packing even started. If corpus dilution were the real bottleneck, this is close to the cleanest test of that hypothesis this project can run without a from-scratch data pipeline.

It didn't move the needle. `10_track_a_means.png`: cpt-v2 mean cosine-to-jac-gpt is 0.8133 vs cpt-v1's 0.8126 — a delta of +0.0007, t=0.22 (`11_track_a_delta_hist_vs_v1.png` is centered almost exactly on zero, 47 win / 53 loss, indistinguishable from a coin flip: `15_track_a_scatter_v1_vs_v2.png`, `13_track_a_win_loss.png`). `14_track_a_boxplot.png` shows the three models' score distributions nearly fully overlapping. **A docs-dominant, curated corpus produces the same result as CPT-v1's diluted one.** Corpus dilution is not the explanation for the semantic ceiling.

---

## 3. Finding: instrument mismatch wasn't it either

CPT-v2's eval is a strictly harder test than CPT-v1's MCQ: Track A grades 100 open-ended generations by embedding similarity to a real RAG-grounded system's answer; Track B has a blind judge (source-blind A/B order, no knowledge of which system produced which answer) read each pair against the actual ground-truth doc passage and decide which is *factually correct*, not just similar in style. If CPT-v1's MCQ was too blunt to see real learning, this setup should have found it.

It didn't. `16_track_b_outcome_bar.png` / `17_track_b_pie.png`: jac-gpt wins 91 of 100 blind judgments, cpt-v2 wins 7, ties 2 — win-or-tie rate 0.09 against a 0.50 bar. This is not a marginal miss the way Track A's numbers are; it's a rout, and it comes from an eval specifically built to be more sensitive than CPT-v1's. **A richer, open-ended, reference-grounded eval still finds nothing to reward.** Instrument mismatch is not the explanation either.

Two independent, well-executed nulls — different corpus, different eval — now sit at the same ceiling.

---

## 4. Finding: what the model actually does wrong

Track A and Track B don't just disagree in magnitude (a small real edge vs. a rout) — `18_honest_gap_scatter.png` shows why, and it's the most mechanistically informative chart in the set. `b003-q-any-inference` scored 0.9355 cosine — 2nd-highest of all 100 questions, near-maximal stylistic similarity to jac-gpt's answer — and still lost the blind judgment outright: cpt-v2 hedged ("may infer or leave as unknown") instead of confirming the passage's actual rule. High cosine did not mean correct.

Reading all 100 Track B justifications during judging surfaced a consistent pattern in cpt-v2's losing answers, not random noise: it **invents plausible, fluent, wrong-domain syntax** rather than saying it doesn't know. Examples pulled directly from the blind judgments:

- Asked about Jac's typed-edge syntax, cpt-v2 answered entirely in Neo4j/Cypher query syntax.
- Asked about Jac's file-based routing, cpt-v2 answered "I believe you're referring to... this is standard Next.js routing" — misidentified the language/framework entirely.
- Asked how the Jac toolchain is distributed, cpt-v2 described GitHub-release binaries, Docker containers, and `pip install -e .` — fabricated a plausible-sounding but entirely wrong distribution story where the real answer (a single self-contained `jac` binary, no pip) was directly in the training corpus's own breaking-changes docs.

This is the actual failure mode, not "the model didn't learn enough Jac." **Next-token prediction on doc prose teaches style and vocabulary association — it has no training signal that penalizes fluent, confident, wrong output**, because nothing in a next-token CPT objective distinguishes "plausible continuation" from "correct continuation." A model can absorb the *shape* of Jac documentation (which is why cosine similarity rose at all, marginally, vs. base) without absorbing enforced correctness. That gap is invisible to a similarity metric and glaring to a judge reading for truth — which is exactly the divergence in section 3.

---

## 5. Was the run wasted?

Two different questions, two different answers.

**As a deliverable: yes.** 12 legs (~16.4h wall-clock, `07_leg_duration.png`), a curation pass, and new eval infrastructure produced a checkpoint that clears 0 of 3 acceptance gates (`19_acceptance_gauges.png`) and isn't usable as a foundation for Phase 4. Nothing downstream builds on this checkpoint as-is.

**As information: no.** CPT-v2 killed two live, previously-unresolved hypotheses in one well-controlled run — corpus dilution and instrument mismatch — and did it cleanly enough (zero CF regression throughout, `09_cf_check_strip.png`; well-behaved loss curves with no pathological leg, `04_loss_curves.png`/`05_val_loss_delta.png`/`08_train_vs_val_scatter.png`) that the null can't be waved away as a broken run. That's a real, load-bearing result for deciding what *not* to spend the next attempt on.

---

## 6. What to try next

Given two independent CPT nulls that survived a corpus-mix change and an eval-instrument change, the honest read is that **continual pretraining via next-token prediction on doc prose is not the right lever** for teaching enforced syntactic/semantic correctness — the failure mode in section 4 is structural to the objective, not a tuning problem.

**Recommended: skip a third CPT attempt, move to Phase 4 (SFT/DPO) directly on base.** `workflow.md`'s existing plan already calls for DPO pairs contrasting semantically-correct vs. subtly-wrong-but-compiling OSP idiom — that objective directly penalizes the "confident, fluent, wrong" failure mode this analysis found, in a way next-token CPT structurally cannot. That plan was gated behind "CPT checkpoint accepted"; two nulls is enough to drop that gate and test SFT/DPO as its own falsifiable experiment rather than waiting on a third CPT success.

**If CPT is worth one more shot first**, the one real variable neither v1 nor v2 varied is corpus *shape*: both trained on doc *prose* (markdown describing Jac, code examples embedded but not dominant). Neither tried CPT on a corpus dominated by actual compilable Jac snippets rather than prose about Jac — closer in shape to what actually needs to be learned. That's a genuine, still-untested single-variable CPT hypothesis if you want to exhaust the lever before moving on. Not recommended as the first move given the section 4 finding, but noted as the one open door.

---

## Chart index

| # | file | what it shows |
|---|---|---|
| 1 | `01_corpus_composition.png` | token composition, v1 vs v2, by source |
| 2 | `02_corpus_share_pie.png` | v2 corpus share by source |
| 3 | `03_curation_verdicts.png` | Fable curation pass keep/upweight/drop |
| 4 | `04_loss_curves.png` | train & val loss per leg, floor/target/ceiling |
| 5 | `05_val_loss_delta.png` | leg-over-leg val loss change |
| 6 | `06_lr_schedule.png` | LR decay across all 12 legs |
| 7 | `07_leg_duration.png` | wall-clock time per leg |
| 8 | `08_train_vs_val_scatter.png` | train vs val loss, colored by leg |
| 9 | `09_cf_check_strip.png` | CF-check pass/fail per leg |
| 10 | `10_track_a_means.png` | mean cosine-to-jac-gpt per model, vs required margin |
| 11 | `11_track_a_delta_hist_vs_v1.png` | per-question delta, cpt-v2 − cpt-v1 |
| 12 | `12_track_a_delta_hist_vs_base.png` | per-question delta, cpt-v2 − base |
| 13 | `13_track_a_win_loss.png` | win/loss counts vs each baseline |
| 14 | `14_track_a_boxplot.png` | score distribution by model |
| 15 | `15_track_a_scatter_v1_vs_v2.png` | per-question cpt-v1 vs cpt-v2 |
| 16 | `16_track_b_outcome_bar.png` | Track B outcome, stacked bar |
| 17 | `17_track_b_pie.png` | Track B outcome, pie |
| 18 | `18_honest_gap_scatter.png` | cosine score vs blind-judge outcome |
| 19 | `19_acceptance_gauges.png` | all 3 acceptance gates vs threshold |
| 20 | `20_summary_dashboard.png` | one-page capstone summary |
| 21 | `21_gap_to_jacgpt.png` | mean distance from jac-gpt, by model |
| 22 | `22_cptv2_vs_jacgpt_head_to_head.png` | both tracks, cpt-v2 vs jac-gpt only |

---

**Bottom line:** the mechanism worked (clean training, zero regression, real infrastructure); the hypothesis didn't survive (corpus mix and eval instrument both ruled out; the actual failure mode is a model that fabricates plausible wrong syntax rather than admitting uncertainty, which next-token CPT can't fix). Per the same discipline as CPT-v1's null: stated plainly, not rounded up.
