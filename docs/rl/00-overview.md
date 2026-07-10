# Jac RL — Overview & North Star

## Goal
Train models that can (1) **fill in parts of Jac code** and (2) eventually **generate new Jac codebases**. We will not get best-in-class results yet — the point of this phase is to get the *workflow* right: a slow, patient, measurable ladder we trust.

## Why a reset
The weekend GRPO run ([RL_WEEKEND_RESULTS.md](RL_WEEKEND_RESULTS.md)) proved the harness, reward, and eval are correct on real 30B — but also that LoRA-GRPO does not move a 30B's greedy decoding at this scale, and the only lever that moved the needle was supervised fine-tuning. So we restart from the ground up, slowly, and measure everything.

## Dataset
`this_is_jac/` only — 77 `.jac` files, deliberately diverse (graph walkers, libs, the littlex social graph, raylib, guestbook). Diversity is why it is a good seed corpus. No external task sources.

## The two models (same ladder, head to head)
- `jac-qwen3coder` — already SFT+DPO on jac (`models/jac-qwen3coder-q4`).
- fresh `qwen3coder` — untrained HF base.

The comparison answers: **does prior jac knowledge help the memorize→generalize curve, and by how much?**

> Qwen3.6 is removed entirely — dense 27B and 35B-A3B both OOM training on 48GB (inference-only). Only the fewer-expert 30B-A3B (Qwen3-Coder) trains here.

## Principles
1. **One variable at a time.** Map the curve before chasing a score.
2. **Honest hard bar.** Exact stdout match is the headline number; diagnostics sit beside it, never replace it.
3. **Memorize before generalize.** Rung 1 proves the plumbing can overfit a single task to 100%.
4. **Carry the scars forward.** The `unwrap_unit` splice fix and the dense body-sim reward term are hard requirements of any harness rewrite — see [01-design.md](01-design.md).
5. **Slow is the feature.** Run the full ladder, read the sweet spot off the curve, don't shortcut.

## Where things live
- [01-design.md](01-design.md) — task formats, holdout, ladder, reward, eval (the "how").
- Root [`RL_FINDINGS.md`](../../RL_FINDINGS.md) — **authoritative corrected results** (the old `02-results.md` table was invalidated by the extractor bug and removed).
- [RL_WEEKEND_RESULTS.md](RL_WEEKEND_RESULTS.md) — archive of what failed and why (prior art).
