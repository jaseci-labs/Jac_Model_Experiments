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

## Recommendation

1. **Do not promote `qwen-cpt-v1` to any default model or downstream stage yet.** It's registered in Studio's chat picker for manual poking, not accepted.
2. **Run Checkpoint 1** (semantic MCQ + human trust check) against `qwen-cpt-v1` vs base `qwen-q4` — this is what actually answers "is this good." Not built yet.
3. **Build the CF regression check** before or alongside Checkpoint 1 — currently zero signal on whether general coding ability moved.
4. If Checkpoint 1 shows a real semantic lift: accept the checkpoint, move to Phase 4 (SFT/DPO redesign spec, unwritten) on top of it. If it shows nothing: that's a real, informative null (matches this project's track record of nulls turning out to be measurement bugs *or* real findings — RL_FINDINGS.md is the precedent for taking a null seriously rather than re-running blind).

## Reference

Full run config, resource usage, PNG graphs, wall-clock breakdown: [cpt-v1-training-results.md](cpt-v1-training-results.md). Roadmap/phase status: [workflow.md](workflow.md). Locked architecture: [design.md](design.md).
