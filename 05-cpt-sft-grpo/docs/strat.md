# 05-cpt-sft-grpo — Strategy

*Run the GRPO stage nobody ever ran, on a corpus and a grader the old RL harness never
had. Two questions, one 24-run matrix: does CPT-v2's surviving structural fingerprint
become a behavioral difference under RL, and was "GRPO ≡ SFT everywhere" ever about
GRPO at all?*

| | |
|---|---|
| Goal | Close Attempt03's four-stage architecture (`base → +CPT → +CPT+SFT/DPO → +CPT+SFT/DPO+GRPO`) by running the missing GRPO stage for both arms, and re-test the `02-rl-grpo` null on a harness that no longer has its two suspected artifacts. |
| Dataset | New multi-source RL corpus: HOLE-marker drivers mined from the 17-repo, already-decontaminated CPT-v1 code corpus (`03-cpt-only/dataset/cpt/manifest.json`) — `this_is_jac` plus 16 further repos. Replaces `02-rl-grpo`'s `this_is_jac`-only pool (116 drivers on disk, 84 of them `this_is_jac`-origin across 12 files). |
| Models | One base (`models/qwen-q4`, Qwen3-Coder-30B-A3B-Instruct q4), four LoRA lineages: fresh-SFT-warm, fresh-cold, cptv2-SFT-warm, cptv2-cold. |
| Verification | **Type-B AST-equivalence** on the spliced completion (α-normalized AST equality = pass), with `jac run` + byte-exact stdout kept as a second, independent pass route where the task is deterministically runnable. Replaces exact-stdout-only. No learned reward model. |
| Lever | GRPO on top of a real, non-fused SFT policy — the exact configuration 04's fuse bug made impossible to test and 02's corpus made unmeasurable. |

See [`spec.md`](spec.md) for the design of record and [`workflow.md`](workflow.md) for
the runbook. This doc is the *why*.

---

## The reset

Two prior threads both ended in a place that this experiment is the only way out of.

**`02-rl-grpo` ended at "GRPO ≡ SFT everywhere."** The finding reproduced three times
across two corpora and survived every escape hatch that was raised against it: GRPO was
re-run at 500 iterations / 10× learning rate (the tuned arm — identical output); the
σ=0 cold-start explanation was *refuted* rather than confirmed, because the dense
similarity term gave real within-group variance (σ=0.09–0.21) and GRPO still moved
nothing; and the raw-base GRPO control landed **exactly** on base. Corrected numbers
(`02-rl-grpo/RL_FINDINGS.md`, authoritative — the `docs/rl/` verdicts predate the
extractor-bug fix and are superseded): SFT lifts greedy 38.9% → 61.1% at rung-20, GRPO
adds nothing on top, and the deployable configuration is SFT + best-of-k with the Jac
compiler as verifier (~78–82%). What was *never* ruled out: that the whole null was a
property of the **harness** — an exact-stdout gate that gives zero credit to a
structurally correct body, a corpus mined from one repo with a ~84-task ceiling and one
32-task file dominating it, and a holdout too small and too concentrated to resolve
anything below a large swing.

**`04-cpt-sft` ended at "CPT-v2 is behaviorally absorbed but structurally alive."** The
two arms (fresh base vs CPT-v2 base, byte-identical SFT/DPO recipe and 855-row
holdout) gave: base +37.0pp for CPT-v2 (47.3% vs 10.5%, z≈16.75), post-SFT +2.8pp
(72.6% vs 69.8%, z≈1.28, p≈0.20 — not significant), post-DPO parity in both arms after
the `mlx_lm.fuse` bug was found and removed. But the Phase-4 q_proj SVD probe found the
CPT-v2-base SFT adapter's update still *looks* like CPT-v2's own: stable rank 2.7–4.3
and rank-1 magnitude 0.93–1.37, versus the fresh-base SFT adapter's 4.4–6.0 and
0.47–0.66, on an untrained reference of ~16 (flat spectrum). Two differently-shaped
q_proj updates reaching the same pass rate. That is a live question SFT cannot answer,
because SFT converged.

05 is the intersection: **RL is the stage where a differently-shaped policy has a
different amount of room to move**, and the new corpus + new grader are exactly the two
things `02-rl-grpo`'s null could have been an artifact of. Running them together is
deliberate — if the ladder moves, the corpus/grader change is the headline and the
CPT-vs-fresh comparison is secondary; if it doesn't, the null is confirmed a fourth
time on materially different inputs, which is a much stronger null than three
repetitions on the same one.

---

## Anchors

What keeps this from being "train four things and hope."

1. **The compiler + the AST are both free oracles.** Every completion is spliced into a
   real mined driver at `__HOLE__` and (a) parsed, its hole-unit sub-AST α-normalized
   and compared to the gold body's, and (b) where the driver is deterministically
   runnable, executed and stdout-compared. Neither oracle is learned, neither can be
   gamed by prose, and rejection is free — the limit is the model, not the grader.
   Adding the AST route is what makes non-runnable Jac (client components, fullstack
   files, library code with no deterministic entry point) minable at all; the
   stdout-only gate structurally excluded it.
2. **The ladder = controlled scaling.** Train-set size climbs 1 → 3 → 5 → 10 → 20 → all
   against a *fixed* holdout, each rung a strict superset of the last (front-slice of a
   stable, family-interleaved trainpool — `pick_rung.jac`). Only task count changes
   between rungs. The shape of the curve is the result; no single cell is.
3. **Four lines = the lineage probe.** Two warm (SFT-started) and two cold
   (no SFT) lines, differing only in whether CPT-v2 sits in the LoRA's history. Warm
   vs warm answers "does CPT change GRPO's ceiling"; cold vs cold answers "does CPT
   alone change GRPO's floor"; warm vs cold reproduces the cold-start control that
   `02-rl-grpo` ran and that this phase must not silently drop.
4. **One continuous LoRA lineage per line.** The `cptv2` lines are CPT-v2 → SFT → GRPO
   in the *same* A/B matrices — never fused into base weights, never stacked as two
   separate adapters. Confirmed by reading `sft_cptv2_probe/configs/sft.yaml`'s own
   `resume_adapter_file`. GRPO continues that lineage; it does not compose it.

---

## Research questions → hypotheses

Two questions (both, per user). Written here as falsifiable predictions with explicit
refutation conditions. **No verdicts — nothing in this phase has run.**

### RQ1 — Does CPT change GRPO's ceiling, or just SFT's?

**H1 (warm ceiling).** The q_proj fingerprint that survives SFT is not inert: a
policy whose q_proj update is rank-1-concentrated (stable rank 2.7–4.3) responds
differently to RL than one that is more spread (4.4–6.0), even though the two start at
statistically indistinguishable pass rates.

- *Prediction:* `cptv2-SFT-warm` final-checkpoint AST-equivalence pass exceeds
  `fresh-SFT-warm`'s at **≥3 of the 6 rungs**, with the paired per-item difference's
  95% interval excluding 0 at those rungs.
- *Refuted if:* the two warm curves stay inside each other's CI at every rung. That
  closes the CPT thread **for good** — the fingerprint would then be structurally real
  and behaviorally inert at every downstream stage measured (base → SFT → DPO → GRPO).
- *Note the asymmetry:* 04 already showed the two arms converge behaviorally after SFT,
  so H1 predicts a *divergence that only RL can produce*. Any gap must be argued
  against the +2.8pp / p≈0.20 SFT-stage prior, not against the +37pp base-stage one.

**H1b (cold floor).** CPT-v2 alone — no SFT anywhere in the lineage — changes what
GRPO can do from a cold start.

- *Prediction:* `cptv2-cold` ends above `fresh-cold` at ≥3 rungs, paired interval
  excluding 0. Motivated by the base-stage +37.0pp: the CPT-v2 base is a materially
  more capable starting policy on Jac, so if cold-start GRPO is capability-gated rather
  than method-gated, this is where it shows.
- *Refuted if:* both cold lines sit on their respective no-training anchors. That says
  GRPO's inability to move is about the method at this scale, not about how good the
  starting policy is — which would also make H1's warm result easier to interpret.

### RQ2 — Was "GRPO ≡ SFT everywhere" GRPO, or the old harness?

**H2 (harness artifact).** The `02-rl-grpo` null was produced by exact-stdout grading
on a single-repo, ~84-task corpus, not by GRPO/LoRA/30B.

- *Prediction:* at least one line's AST-equivalence pass rises above its own rung-0
  anchor (the un-GRPO'd checkpoint that line resumes from) by a margin whose paired
  95% interval excludes 0, at ≥1 rung. Any such move is **the headline of this
  phase** — CPT-vs-fresh becomes secondary in that case.
- *Refuted if:* all four lines stay flat against their anchors at every rung. That is
  the fourth independent confirmation of the null, now on a different corpus (17 repos
  vs 1), a different grader (AST-equivalence vs exact stdout), and a different starting
  policy family (a full 7-category SFT model at ~70% functional pass, versus
  `02-rl-grpo`'s rung-SFT models). A null that survives all three changes is a much
  stronger statement than the one currently on record.

**H3 (cold-start control).** GRPO cannot bootstrap from a policy with no relevant
skill — the reason both cold lines exist.

- *Prediction:* both cold lines end statistically indistinguishable from their
  no-training anchors, reproducing `02-rl-grpo`'s raw-base-GRPO control, which landed
  exactly on base.
- *Refuted if:* either cold line moves. Note this is **not** a σ=0 prediction: 02
  already refuted the σ=0 mechanism (the dense term gave σ=0.09–0.21 and GRPO still
  moved nothing), and 05's tiered reward keeps a similarity term in every tier for the
  same reason. If a cold line moves here, the difference is the corpus or the grader,
  not the variance.

---

## Carried scars (non-negotiable build requirements)

These cost real time to find in `02-rl-grpo` and `04-cpt-sft`. Re-breaking any of them
invalidates the phase silently rather than loudly, which is the expensive kind.

| # | Scar | Where it was learned | What 05 must do |
|---|---|---|---|
| 1 | **`unwrap_unit` splice** | `02-rl-grpo/docs/rl/strat.md` "Carried scars" §1 | Models emit the whole enclosing unit (`can name { body }`), not the bare body. Unwrap exactly one enclosing unit before splicing into `__HOLE__`, or the file nests `can {…can {…}…}` and never runs. This faked the entire first weekend run. Implemented in `reward_logic.jac:unwrap_unit`. |
| 2 | **Brace-matched hole-unit extraction** | `reward_logic.jac:unit_body` docstring | Models reproduce the *entire* driver, so the extractor must pull **that specific unit's** body by name (`hole_unit_name` → brace-match), not the first/last brace pair. Blind unwrapping grabbed the docstring and undercounted the reward ~4×. |
| 3 | **One shared extractor, independently tested** | `RL_FINDINGS.md` Era 2 | The eval script and the GRPO reward shared one helper; a bug in it capped every cell's ceiling regardless of training condition and undercounted 3.5–4×, making a flat ladder look like a real null for three weeks. The AST grader is the *same* shared surface. It gets its own self-test (`test_reward.jac` / `test_ladder.jac` convention), asserting that a gold body scores 1.0 against itself on every mined task, before any training runs. |
| 4 | **Variance in every reward tier** | `02-rl-grpo/docs/rl/strat.md` §2, `reward_logic.jac` docstring | The similarity term is computed for **every** completion including non-compiling ones — it is the only term not gated behind `runs`, and without it a group of all-failing rollouts has zero within-group variance → zero GRPO advantage. In 05 that term becomes `ast_sim` (with a difflib text fallback when the completion does not parse, so the term is *never* undefined). |
| 5 | **Monotone reward tiers, no additive formula** | `reward_logic.jac` docstring | The old dense v2 formula (`0.25·compiles + 0.25·runs + 0.25·output + 0.10·idiom + 0.15·body_sim`) let a wrong-output, high-idiom completion outscore a correct terse one. 05 keeps the tiered replacement: each tier's ceiling is strictly below the next tier's floor, so a pass always dominates a non-pass. Type-B grading changes *what counts as a pass*, not this ordering property. |
| 6 | **Idiom is a diagnostic, never a reward term** | `reward_logic.jac` docstring | Trivially gamed (stuff `-->` tokens), anti-correlates with terse gold bodies, and the headline metric never rewarded it. Report idiom density; do not pay for it. |
| 7 | **File-disjoint holdout** | `build_rl_splits.jac` docstring §1 | Whole *source files* are held out, so a held-out walker never shares its skeleton with a trained sibling. A family-stride split leaks structure. |
| 8 | **Family-interleaved trainpool** | `build_rl_splits.jac` docstring §2 | The old family-ordered pool meant `pick_rung`'s front-slice gave rungs 1–20 **zero** `social_graph` tasks while the holdout was 47% `social_graph` — the "more tasks" curve was really a "which family" curve. Round-robin interleave keeps every prefix-N family-balanced *and* a strict superset of the previous rung. With 17 repos the same failure mode returns at repo granularity: interleave across **repo × family**, not family alone. |
| 9 | **`valid` ≠ `holdout`** | `build_rl_splits.jac` docstring §3 | The old split wrote `valid.jsonl == holdout.jsonl`, so the trainer selected checkpoints on the test set. `valid` is carved from the trainpool tail, never from the holdout. |
| 10 | **Never `mlx_lm.fuse`** | `04-cpt-sft/docs/reports/2026-07-cpt-vs-fresh-comparison.md` §3.2 | Fusing a LoRA delta into an int4 base dequantizes, adds, and re-quantizes — rounding the delta away almost entirely (~15% of packed 4-bit elements changed bit-pattern; the fused "SFT" model generated plain React/JSX, indistinguishable from raw base). This produced a 12–13% "DPO collapse" that was not DPO at all. Every 05 line seeds via `--resume-adapter-file` against the raw `models/qwen-q4`. |
| 11 | **GRPO cannot start cold** | `02-rl-grpo/docs/rl/01-design.md` §3, `RL_FINDINGS.md` Era 2 | Hence the two cold controls in the matrix. Note the *mechanism* claim (σ=0) was refuted; the *observation* (raw-base GRPO ≡ base) was not. Keep the control, drop the explanation. |
| 12 | **Determinism guard on stdout-graded tasks** | `build_tasks.jac` (double-run check) | A driver whose stdout varies run-to-run is unusable as an exact-match target (`jid()`/time/unseeded RNG leak). Under Type-B grading such a task is still usable via the AST route — but it must be *tagged* as AST-only, never silently stdout-graded. |
| 13 | **Run the full ladder, no early stop** | `02-rl-grpo/docs/rl/strat.md` scope discipline | The sweet spot is read off the finished curve, not guessed mid-run. 04 reinforced this from the other side: its SFT peak *tied* its own final checkpoint in both arms, and its DPO curve had no recoverable sweet spot at all. |

---

## Scope discipline

- **Hole-fill only.** Whole-file regeneration stays deferred (`02-rl-grpo`'s Type-B
  *task type*), even though 05 borrows Type-B's *grading idea*. Mixing task types would
  muddy the ladder curve, which is the result.
- **One holdout, one metric.** New RL holdout, AST-equivalence pass rate. No dual-eval
  against 04's 855-row functional holdout — that measures a different objective and
  would invite cherry-picking whichever number moved.
- **Harness extended in place.** All code lives in `02-rl-grpo/rl/`. 05 owns data,
  adapters, results, docs. Duplicating the harness would fork the shared
  reward/eval surface, which is exactly how scar #3 happened.
- **Measure before scaling.** Corpus size, per-repo yield, and holdout composition are
  all TBD until mining runs. Report the real numbers; do not pad a thin repo's yield to
  hit a target.
