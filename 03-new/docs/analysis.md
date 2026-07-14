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

## Recommendation

1. **Do not promote `qwen-cpt-v1` to any default model or downstream stage yet.** It's registered in Studio's chat picker for manual poking, not accepted.
2. **Run Checkpoint 1** (semantic MCQ + human trust check) against `qwen-cpt-v1` vs base `qwen-q4` — this is what actually answers "is this good." Not built yet.
3. **Build the CF regression check** before or alongside Checkpoint 1 — currently zero signal on whether general coding ability moved.
4. If Checkpoint 1 shows a real semantic lift: accept the checkpoint, move to Phase 4 (SFT/DPO redesign spec, unwritten) on top of it. If it shows nothing: that's a real, informative null (matches this project's track record of nulls turning out to be measurement bugs *or* real findings — RL_FINDINGS.md is the precedent for taking a null seriously rather than re-running blind).

## Reference

Full run config, resource usage, PNG graphs, wall-clock breakdown: [cpt-v1-training-results.md](cpt-v1-training-results.md). Roadmap/phase status: [workflow.md](workflow.md). Locked architecture: [design.md](design.md).
