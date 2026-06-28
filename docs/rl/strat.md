# Jac RL Strategy

*Slow, ground-up ladder to find where fine-tuning makes a Qwen3-Coder model fill in Jac code — and eventually generate whole Jac codebases. Workflow correctness first, scores later.*

| | |
|---|---|
| Goal | Models that (1) fill in parts of Jac code, (2) eventually generate new Jac codebases. |
| Dataset | `this_is_jac/` only (77 `.jac`, diverse: graph walkers, libs, littlex, raylib, guestbook). |
| Models | `jac-qwen3coder` (already SFT+DPO on jac) vs **fresh** `qwen3coder`, head to head. Qwen3.6 removed (OOM training on 48GB). |
| Verification | Jac compiler + runtime — splice completion, `jac run`, **exact stdout** = pass. No learned reward model. |
| Lever | SFT is the proven mover; GRPO re-tested as a controlled challenger, not the primary bet. |

See [00-overview.md](00-overview.md) for the north star and [01-design.md](01-design.md) for the concrete design. This doc is the *why*.

---

## The reset

The weekend GRPO run ([RL_WEEKEND_RESULTS.md](RL_WEEKEND_RESULTS.md)) proved the harness/reward/eval correct on real 30B, and proved two hard limits: LoRA-GRPO does not move a 30B's greedy decoding at this scale, and SFT warm-start was the only lever that moved holdout (0%→14.3%). The old work chased scores before the workflow was trustworthy. So: restart slowly, change one variable at a time, and measure the curve before chasing a number.

---

## Three anchors

What replaces "just train and hope" with something measurable.

1. **Compiler + runtime = the free oracle.** Every completion is spliced into a real `this_is_jac`-derived file and run. Exact stdout is an honest, unfakeable hard gate. Rejection is free, so cheap signal is abundant — the limit is the model, not the grader.
2. **The ladder = controlled scaling.** Train-set size climbs 1→3→5→10→20→all against a *fixed* holdout pool. Every rung is comparable to the last because only one thing changed (task count). The shape of the holdout curve is the result, not any single cell.
3. **Two models = the prior-knowledge probe.** Running the identical ladder on an already-jac-trained base and a fresh base isolates exactly what prior jac SFT/DPO buys on the memorize→generalize curve.

---

## Research questions

The ladder exists to answer these, in order:

1. **Does the plumbing work?** Rung 1: can SFT overfit a *single* task to 100% on itself? If not, nothing downstream is trustworthy. (Memorize gate.)
2. **Where does generalization start?** At what train-N does fixed-holdout pass first move off the floor? Below that, the model is memorizing; above it, it is learning the task shape.
3. **Where is the sweet spot?** The train-N where added tasks stop improving holdout (plateau). That number — and how it differs between the two models — is the headline finding.
4. **Does GRPO ever beat SFT here?** Its only plausible win is a mid-rung where SFT has lifted base-pass above 0 (so reward variance exists) but holdout still lags. The raw-base-GRPO control confirms the σ=0 cold-start trap reproduces.
5. **Does prior jac knowledge help?** Does `jac-qwen3coder` climb faster / plateau higher than fresh `qwen3coder`, or has DPO already saturated the easy gains?

---

## Hypotheses — VERDICTS (run complete, 2026-06-28)

- **H1 — CONFIRMED.** Rung-1 SFT mem-recall = 100% for both models. Plumbing sound; every downstream number is trustworthy.
- **H2 — REFUTED (saturated holdout) / weak (hard holdout).** On the gb+lib holdout, greedy pass@1 is **flat at 26.67% across all train-N** — no rising curve. Only on the harder sg-inclusive holdout (step 7) does fresh `qwen3coder` SFT rise (14.3%→28.6% at rung-all, sg slice 0→1/5). There is no clean "rises-then-plateaus" curve; SFT moves greedy only where there's headroom (hard, unsaturated tasks) and even then barely.
- **H3 — CONFIRMED, and strengthened.** SFT+GRPO ≈ SFT ≈ base on greedy pass@1 (26.67%); the **tuned** arm (500it/1e-5) is identical; raw-base GRPO ≈ base. GRPO adds nothing at any rung, model, or tuning. **The σ=0 mechanism was REFUTED** — the dense body-sim reward gave real variance (σ=0.09–0.21) yet GRPO still moved nothing, so the null is *not* a cold-start artifact: LoRA-GRPO is genuinely inert here.
- **H4 — PARTIAL / no convergence.** `jac-qwen3coder` starts higher (mem-recall; base gen on the sg holdout 21% vs 14%) but the curves do **not** converge — both stay flat on greedy generalization; the fresh model is the only one that moves (on the hard holdout). Prior jac SFT/DPO buys a higher floor, not a different ceiling.

## CONCLUSION

**At 30B-A3B / LoRA / 48GB on file-disjoint this_is_jac: neither SFT nor GRPO — even tuned — moves greedy holdout generalization, except a faint SFT bump on the hardest unsaturated idiom (sg walkers, 0→1/5).** What training *does* move: **pass@k mean** (SFT +7pp, sampling efficiency — but the boundary/max pass@k is unchanged, base already reaches it) and **train-recall**. GRPO ≈ SFT on every metric. This is the weekend null, now leak-free, properly powered-honest (n=15, wide CIs), and with the "GRPO undertrained" and "σ=0 artifact" escape hatches both closed. **RL is not the lever at this scale — SFT+DPO + dataset quality is.** RL thread closed; see [02-results.md](02-results.md) for all 32 cells across both holdouts.

---

## Carried scars (non-negotiable build requirements)

The weekend cost real time to find these; the fresh harness must not re-bleed:

1. **`unwrap_unit` splice.** Models emit the whole enclosing unit (`can ... { body }`), not the bare body. Unwrap one enclosing unit before splicing into `__HOLE__`, or the file nests `can {...can {...}...}` and never runs — this silently faked the entire first run.
2. **Dense body-sim reward term.** `difflib ratio(body, gold refbody)`, computed for *every* completion including non-compiling ones, is the only term that gives within-group variance at 0% pass. Without it GRPO advantage is `(r−mean)/σ = 0` → zero gradient → the σ=0 trap.

---

## Scope discipline

- **Hole-fill first.** The proven primitive. Map its full ladder before anything else.
- **Whole-file is a separate later track.** Regenerating an entire `.jac` from spec is the bridge to "generate codebases" but is harder and lower-yield; it gets its own ladder so it can't muddy the hole-fill curve.
- **Exact stdout is the headline.** near-pass (osim≥0.9), avg-osim, and the graded reward score are diagnostics — they explain misses, they never replace the pass bar.
- **Run the full ladder.** No early stop. The sweet spot is read off the finished curve, not guessed mid-run.
