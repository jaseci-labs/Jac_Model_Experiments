# CPT-v1 Analysis

*Authoritative verdict doc for attempt 03's first training run — same role `02-rl-grpo/RL_FINDINGS.md` plays for the RL thread. Raw log-derived numbers live in [cpt-v1-training-results.md](cpt-v1-training-results.md); this doc is the interpretation.*

## Verdict

**Training itself: good, textbook-clean.** Loss dropped, tracked with validation, no divergence, no OOM, finished on schedule.

**Whether it's "good" for the actual goal — closing the semantic gap that SFT and RL both hit — is unknown.** Perplexity dropping on held-out CPT text is *necessary but not sufficient* evidence. It proves the model got better at predicting the next token in Jac docs/code/blogs. It does **not** prove the model understands OSP/Jac semantics any better than before — that could be surface pattern-fitting (markdown structure, syntax shape, formatting habits) as easily as real conceptual gain. The only way to tell the difference is Checkpoint 1 (semantic MCQ + human trust check), which has not run. Until it does, treat CPT-v1 as *plausible, not confirmed*.

## The numbers

| Metric | Start (iter 1) | End (avg, last 10 evals) | Change |
|---|---|---|---|
| Val loss | 1.495 | 1.079 | −27.8% |
| Val perplexity | 4.46 | 2.94 | **−34.0%** |
| Train loss (100-iter avg) | 1.205 (first 100) | 0.982 (last 100) | −18.5% |

Perplexity is the more readable number: at the start, the model was on average "confused" between about 4-5 plausible next tokens on held-out CPT text; by the end, about 3. That's a real, meaningful drop for a 3-epoch, 10M-token LoRA run — not huge in absolute CPT terms (production CPT runs are often 100-1000x this token count), but consistent with a small-corpus run fitting cleanly rather than memorizing badly or destabilizing.

**Signs this was a healthy run, not overfitting or a fluke:**
- Val loss and train loss moved down together at similar magnitude — overfitting would show val flattening or rising while train kept dropping. Didn't happen.
- Memory flat at 29.3–30.4GB the entire run — no leak, no gradual OOM creep.
- All 3 epochs completed (10.04M tokens trained ≈ 862 windows × 4096 × 3, matches the packed corpus size) — not a partial/truncated run.
- Throughput stable (423–490 tok/s) — no thermal throttling or degradation over 6h49m.

**Signs this is not yet evidence of the thing that actually matters:**
- No task-based accuracy exists for this checkpoint (CPT is raw language-modeling loss, not pass/fail — see the perplexity-vs-accuracy distinction below).
- No comparison against a proper eval-format holdout, only against the CPT corpus's own held-out split (which shares style/distribution with train, by construction — 85/15 split of the same packed corpus).
- No CF (catastrophic-forgetting) check — general coding ability could have moved in either direction and this run wouldn't show it.
- 3 epochs over one small corpus risks light memorization of *phrasing*, not just concepts — perplexity can't distinguish "learned the idiom" from "memorized this exact paragraph."

## Why there's no accuracy number

CPT trains next-token prediction on raw text — there's no right/wrong label per example, so there's no pass/fail rate to report. This is different from `01-sft-dpo`'s holdout, which grades generated code against a compiler + expected output and produces a real accuracy percentage. A CPT run's only native signal is loss/perplexity, which measures fit, not correctness. Getting an accuracy number for this checkpoint requires a downstream task-based eval — exactly what Checkpoint 1 (semantic MCQ, isolated from syntax) and the online-judge track (compiler/pass@k, reusing `02-rl-grpo`'s harness) are for, per `design.md`'s eval design. Neither has run yet.

## Head-to-head qualitative check (2026-07-14)

Loss curves don't answer "does it actually work" — so ran a real, direct comparison: `qwen-q4` (base) vs `qwen-cpt-v1` (fused) on 5 identical prompts (3 code-gen, 2 concept questions), same sampler (temp 0.2), outputs compiler-checked where applicable. Script: `03-new/cpt_train/eval_headtohead.py`, raw outputs: `03-new/results/cpt-v1/headtohead.json`. This is a fast sanity check, not a substitute for the real Checkpoint 1 (semantic MCQ + trust check, still unbuilt) — 5 prompts, temp 0.2, single sample each, not statistically rigorous.

**Code correctness: neither model produces compiler-valid Jac.** Both write `func add(a: int, b: int) -> int:` (colon, Python-style body) instead of `def add(...) -> int { return a + b; }`. This held on all 3 code prompts, both models, 0/3 compiled either way. **This is expected, not a failure** — CPT trains on raw text (docs/code/blogs), never on "here's a prompt, here's the exact correct syntax" instruction pairs. Teaching exact grammar is explicitly SFT's job in this project's own 3-stage design (`design.md`'s Problem section: SFT already took syntax-error rate 94%→4.2% on its own). This result is actually a small piece of evidence *for* the pipeline's reasoning — CPT alone genuinely doesn't fix syntax, matching the hypothesis, not contradicting it.

**Idiom vocabulary: CPT-v1 shows a real, visible difference.** On the OSP-walker prompt ("write a walker that counts nodes"), base invented a fully fabricated pseudo-DSL — `rule traverse`, `foreach edge in self.outgoing_edges`, `continue next_node` — none of which are real Jac constructs. CPT-v1's attempt, while still not compiler-correct, used genuine Jac-specific vocabulary throughout: `has root;` (real field-declaration idiom), `spawn here -> root;` (real spawn/edge syntax), `here.out_edges` (plausible node attribute). Still hallucinated some non-existent bits (`rule start`), but recognizably drawing from real Jac idiom instead of inventing an unrelated DSL from scratch. This is the kind of signal the design hoped CPT would produce — vocabulary/idiom uptake ahead of a grammar-teaching stage.

**Conceptual answers: no clear improvement.** Both models' explanations of `node` vs `edge` and `by llm()` are similar in quality and both share the same real inaccuracy — describing OSP nodes as having "spatial coordinates" / existing "at specific spatial locations," which mischaracterizes Jac's actual graph-topology semantics as literal physical space. CPT training on the docs + OSP paper corpus (which explicitly explains this) did not visibly correct this specific misconception in 3 epochs on this small a corpus. Worth noting for Checkpoint 1's MCQ design — this is exactly the kind of concept-level error a real semantic eval should be built to catch.

**Bottom line**: CPT-v1 nudges idiom vocabulary in code generation, doesn't fix syntax (as designed — that's SFT's job), and doesn't yet visibly fix conceptual misunderstandings from a small 3-epoch run. Consistent with "plausible, not confirmed" from the verdict above — now with a concrete, honest example of what "not confirmed" looks like in practice, not just a caveat.

## CF regression check (2026-07-14) — PASS

Built the check that was flagged as missing above: 16 classic Python coding tasks (fizzbuzz-adjacent — `is_prime`, `bubble_sort`, `binary_search`, `caesar_cipher`, etc.), deliberately **not Jac**, each graded by executing the model's generated function against 1-4 exact-match test cases in a subprocess with a timeout. Script: `03-new/cpt_train/cf_check/{tasks.py,run_cf_check.py}`. Same head-to-head method as before: identical prompts, both models, temp 0.2.

**Result: `qwen-q4` 16/16 (100%) vs `qwen-cpt-v1` 16/16 (100%) — zero regression, zero delta.** CPT did not measurably hurt general Python coding ability at this task difficulty. Raw outputs + per-task pass/fail: `03-new/results/cpt-v1/cf_check.json`.

**Honest caveat**: both models hit the ceiling (16/16), which means this task set has no headroom to detect a *small* regression — if CPT had caused a subtle 1-2 task drop, a harder task set would be needed to see it. What it does confirm: no *gross* catastrophic forgetting (the failure mode design.md's CF guard is actually worried about — a LoRA CPT run destabilizing basic capability) occurred. Good enough to clear this gate; not proof of zero forgetting at any granularity.

**This clears the CF gate.** Nothing blocks moving to Checkpoint 1.

## Checkpoint 1: semantic MCQ (2026-07-14) — NULL RESULT

Built the eval Checkpoint 1 calls for: 20 multiple-choice questions on Jac/OSP concepts (walker, node/edge, `can`/`def`, `spawn`/`visit`/`disengage`, `has` fields, `root`, OSP philosophy, `by llm()`, `sem` annotations), no compiler involved — pure concept recognition. Hand-authored and grounded against this project's own verified facts about Jac (not the full LLM+MCP-authored pipeline from the original deck — a scope call made for this first pass, see the earlier "what is semantic MCP" discussion). Question bank: `03-new/cpt_train/mcq_check/questions.py`. Runner: `run_mcq_check.py`, greedy decode (temp 0.0) so answers are deterministic, not sampled.

**Result: `qwen-q4` 18/20 (90%) vs `qwen-cpt-v1` 18/20 (90%) — zero delta. Not just the same score: the exact same 18 questions right and the exact same 2 questions wrong, for both models.** Raw per-question outputs: `03-new/results/cpt-v1/mcq_check.json`.

**This is a real null, not noise from too few questions landing on a coin flip** — an actual behavior change would show up as at least *some* disagreement between the two answer sets, even a small one. Getting byte-identical answer patterns across 20 independent questions is a strong signal that CPT-v1's LoRA delta did not move constrained-choice concept recognition at all on this instrument.

The two shared wrong answers are informative about what CPT did *not* fix:
- **`obj` vs `class`** (which keyword Jac prefers for archetypes) — both models confidently answered `type` (wrong; the docs corpus was upsampled 3x specifically because it's the highest-quality source, and this exact fact is a basic, frequently-stated one). CPT training did not correct this specific, learnable factual gap.
- **Which construct isn't real Jac** — both models picked `has` (a real, extremely common keyword used throughout this very codebase) instead of the actual fabricated one (`rule`, the same hallucinated construct base invented unprompted in the head-to-head walker test above). Neither model has reliable grounding on which keywords are real.

**How this fits with the head-to-head qualitative check above**: that test showed CPT-v1 using more real Jac vocabulary in *free-form generation* (`has`, `spawn`, `here.*`) than base's fully-invented pseudo-DSL. This MCQ result shows CPT did *not* move *constrained-choice recognition* of the same concepts. Read together, the honest interpretation is: CPT's effect, if any, shows up as a generation-style/vocabulary nudge, not as corrected factual understanding — the model picked up *how Jac code tends to look* more than *what Jac constructs actually mean*. That's a meaningfully different (and weaker) claim than "CPT taught the model Jac semantics," and it's the claim the evidence actually supports.

**Verdict on the original hypothesis** (did CPT move the semantic ceiling SFT+RL both hit): **no evidence of that yet, on this instrument.** This doesn't rule it out — 3 epochs on 3.8M tokens is a small CPT run, and 20 questions is a small eval — but it means CPT-v1 should not be treated as a confirmed win. Matches this doc's original "plausible, not confirmed" framing, now with a real null attached instead of just an absence of evidence.

## Recommendation (updated post-Checkpoint-1)

Both gates have now run. **CF: pass. Checkpoint 1: null.** CPT-v1 is confirmed not to have broken anything, and confirmed not (yet) to have delivered the thing this whole attempt exists to test.

1. **Do not accept `qwen-cpt-v1` as validated.** Keep it registered in Studio's chat picker (useful for manual poking, and it *did* show a real vocabulary-uptake difference in free-form generation), but the MCQ null means it hasn't earned promotion to any default model or downstream stage.
2. **Don't yet build Phase 4 (SFT/DPO redesign) on top of `qwen-cpt-v1`** — the checkpoint this stage exists to validate didn't pass its own validation. Building on top of an unconfirmed checkpoint compounds the uncertainty instead of resolving it.
3. **Before re-running CPT bigger/longer, get more signal on *why* the MCQ was flat.** Two live hypotheses, not mutually exclusive: (a) 3 epochs / 3.8M tokens is genuinely too small a CPT run to shift factual recall measurable at MCQ granularity — scale up token count/epochs and re-test; (b) the LoRA capacity (rank 16) or which layers it targets is limiting how much factual content can actually be absorbed vs. just style/vocabulary — the ladder work already established LoRA-scale limits show up this way in this project's RL experiments, worth checking if the same ceiling applies here.
4. **The MCQ bank itself is now a reusable instrument** — future CPT runs (bigger corpus, more epochs, different LoRA rank) can be scored against the same 20 questions for a real before/after comparison, not just base-vs-single-checkpoint.
5. This is a real, informative null, not a failure to hide — matches this project's track record (`RL_FINDINGS.md` is the precedent: take a null seriously, don't paper over it, don't just re-run blind hoping for a different answer).

## Reference

Full run config, resource usage, PNG graphs, wall-clock breakdown: [cpt-v1-training-results.md](cpt-v1-training-results.md). Roadmap/phase status: [workflow.md](workflow.md). Locked architecture: [design.md](design.md).
