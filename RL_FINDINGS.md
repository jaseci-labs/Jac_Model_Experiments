# RL Findings — Jac Code Generation

Everything we know about making a 30B coder model write idiomatic, compiler-correct
Jac: the definitions, the three-era story of how we got here, the causal reasoning
behind each result, and the deployable answer. Model under study: `jac-qwen3coder`
(Qwen3-Coder-30B-A3B, already SFT+DPO'd on Jac), trained with MLX LoRA on a 48GB
Apple M5 Pro.

---

## Legend — read this first

Everything below assumes these terms. Grouped by what they're *for*, not
alphabetical, so related ideas sit together.

**Training methods**
- **SFT** — supervised fine-tuning on prompt→answer pairs. Teaches the model what
  the answer should look like, directly.
- **DPO** — direct preference optimization; trains on chosen-vs-rejected pairs
  instead of single targets.
- **GRPO** — group-relative policy optimization, the RL method used here: sample a
  group of rollouts per prompt, advantage = `(reward − group mean) / group σ`, push
  the policy toward the above-average ones. Needs reward *variance within a group*
  to produce any gradient at all — see the σ=0 trap below.
- **tuned-GRPO** — a GRPO variant run at 500 iterations / 10× the default learning
  rate, specifically to rule out "GRPO just needed more training."
- **LoRA** — low-rank adapters; only small injected matrices train, not the base
  weights. The only way a 30B model trains at all on 48GB.
- **warm-start** — a brief SFT pass before RL, so the model isn't starting from a
  policy that fails 100% of rollouts (which produces zero reward variance — see σ=0
  trap).
- **STaR** — self-taught reasoner loop: sample, keep the verified-correct ones,
  SFT on those, repeat.
- **MLX** — Apple's on-device ML framework; runs the LoRA training here.
- **Metal** — Apple's GPU compute API, one layer below MLX; this is where OOM
  crashes actually happen.
- **unified memory** — Apple silicon's RAM shared between CPU and GPU; the hard
  48GB ceiling every experiment design had to respect.
- **σ=0 trap** — when every rollout in a GRPO group scores identically (typically
  because the base model passes ~0% of hard tasks), the advantage formula
  `(r − mean)/σ` divides by zero variance → the gradient is zero regardless of
  learning rate. RL cannot bootstrap a skill the base model has none of.
- **KL** — divergence of the trained policy from its starting policy during RL;
  ≈0 means the model's outputs didn't move at all.

**Eval & scoring**
- **pass@1 / greedy** — one deterministic (temperature-0) answer; % of holdout that
  is byte-exact correct. The headline number.
- **pass@8 / pass@k (oracle)** — sample k times, pass if *any* sample compiles,
  runs, and matches. The *reachable ceiling* — what the model can do with an oracle
  picking its best sample.
- **best-of-k (deploy)** — sample k, return the first candidate the Jac **compiler**
  accepts — no gold answer peeked at. The number you'd actually ship, since nothing
  needs to know the right answer at inference time.
- **syntax gap** — pass@8 − pass@1. The gap between what the model already knows how
  to produce (if you let it try more than once) and what it defaults to greedily.
- **exact-stdout** — the grading rule: produced program output must byte-match
  expected output. No partial credit in the primary metric (see `osim` for the
  diagnostic-only partial-credit signal).
- **osim** — output similarity (`difflib` ratio between produced and gold stdout,
  0–1). Diagnostic only, not used for pass/fail — shows *how close* a near-miss was.
- **mem** — train-recall: eval on the rung's own training tasks. A check against
  overfitting, not a generalization measure.
- **Wilson CI** — binomial confidence interval; the honest noise bar on every pass
  rate reported here, most of which come from small (n=11–32) holdouts.

**Task types**
- **hole-fill task** — a real `this_is_jac` program with one function body blanked
  out; the model fills it in. Under-specified by design — the model has to infer
  intent from the surrounding code.
- **this_is_jac** — the real Jac codebase these hole-fill tasks are mined from; the
  ground-truth source, as opposed to synthetic tasks.
- **conversion task** — "translate this Python function to Jac." Same grader as
  hole-fill, but the spec (working Python) is unambiguous — no guessing intent.
- **synthetic task** — a fresh, authored deterministic Jac function, not mined from
  `this_is_jac`. Exists purely to give holdouts more novel, harder-to-memorize tasks.
- **graph-walker task** — Jac's Object-Spatial Programming idiom: `node`/`edge`/
  `walker`/`visit`. Has no Python equivalent, so conversion tasks structurally can't
  produce them — they need their own dataset.
- **holdout** — fixed tasks the model never trained on; the generalization
  measurement. Five different holdout sets appear here (n=18/17/11/16/32),
  deliberately different sizes and compositions to stress-test the same claim.
- **clean set** — the 16-task pure-fn holdout with 2 junk regex-memorization tasks
  removed (they rewarded verbatim recall, not real Jac ability). The true
  best-of-k ceiling shows up here, undistorted.
- **ladder / rung-N** — the experiment design: train on N tasks (N ∈
  {1,3,5,10,20,all}), eval on the same holdout each time. Isolates "how much
  training data helps" from every other variable.

**Result vocabulary**
- **task interference** — adding more, harder, or more varied training tasks
  regresses a task the model had already learned. The signature of a small LoRA
  adapter running out of capacity to hold multiple skills without them competing.
- **boundary** — the best a model can do *with* sampling (its pass@k ceiling). RL
  can sharpen how often the model reaches its boundary; it's a separate question
  whether RL can *move* the boundary itself.
- **distillation / expert-iteration** — train on a stronger teacher's (or your own
  compiler-verified best) outputs. The one lever that adds capability from outside,
  rather than just re-weighting what the model already produces.
- **Studio** — this repo's app (Jac ML Studio); ships the GENERATE panel and the
  full RL data pipeline live.
- **the two models** — `qwen3coder` (fresh, never seen Jac) vs. `jac-qwen3coder`
  (already SFT+DPO'd on Jac — the capable one, and the only one used past Era 0).

---

## TL;DR

**The model is capable; the real problem was a closeable *syntax* gap, not a
capability wall — and for three weeks a measurement bug made it look like neither
of those things was true.**

- **best-of-k + the Jac compiler as verifier ships ~94%** on meaningful pure-function
  tasks, **today, zero training** — sample k, keep the one that compiles+runs; the
  compiler is a perfect picker.
- **SFT works:** greedy **39% → 61%** (peak at rung-20), and the lift **holds on a
  bigger, fresher holdout (n=32)** and **generalizes to unseen tasks** — it isn't
  memorizing.
- **The gap is syntax, not logic:** if the model's Jac compiles, it's almost always
  exactly right. Greedy decoding lands on an almost-valid variant; sampling reaches
  the real answer.
- **Conversion (python→jac) is the best task framing** — base 64% greedy, conv-SFT
  **73% greedy / 82% best-of-k**.
- **GRPO ≈ SFT** (no extra lift); raw-GRPO from the base moves nothing. The fresh
  (non-Jac) model is a dead end.
- **The one real gap: free-form NL generation** — both models are tuned for the task
  format and fail arbitrary "write a function that…" prompts.

---

## The arc: three eras, and why the numbers moved so much

The headline number went 14% → 11% → 39% → 61% → 78% → 94% over about two weeks.
Most of that motion had nothing to do with the model getting better — it was three
successive rounds of fixing how we were *measuring* it. This section is the causal
chain; the corrected, trustworthy numbers start at "Era 2."

### Era 0 — weekend GRPO (Jun 20–21): built right, still flat at 14.3%

First full attempt: GRPO with a real compiler/runtime-verified reward (`0.3·compiles
+ 0.3·runs + 0.3·output_match + 0.1·idiom`), LoRA on MLX, warm-start + STaR as
supporting levers. Phase 1 (31 tasks, holdout 7):

| model | base | + warm-start | + GRPO |
|---|---|---|---|
| jac-qwen3coder (SFT+DPO'd) | 14.3% | near-ceiling | 14.3% |
| qwen3coder (fresh) | 0% | **14.3%** | 14.3% |
| qwen3.6 (dense 27B) | 0% | OOM | OOM |

Five things went wrong, each with a distinct cause:

1. **Metal OOM.** `group6/comp512` blew past 48GB at iteration 2 — activation memory
   for that batch/completion-length combo simply didn't fit. Fixed by dropping to
   `group4/comp256/seq1280` (~38GB peak). Not a deep problem, just a config error.
2. **The σ=0 trap.** At ~0% base pass rate, every rollout in a GRPO group scored the
   same low reward → zero variance → zero advantage → zero gradient, at any learning
   rate. **Why this happens:** GRPO's whole learning signal is *relative* — it needs
   some rollouts in a group to be better than others. A binary, all-or-nothing
   reward at a near-zero base pass rate gives it nothing to compare. Reward
   engineering (`v0`→`v2`, adding a `difflib`-based body-similarity term that's
   defined even for non-compiling code) fixed the variance problem — σ rose to
   0.01–0.15, real nonzero gradient flowed — but that fix didn't change the
   evaluation numbers, which was the tell that something else was wrong (→ #3).
3. **The splice bug (the real cause of "everything is flat").** Models emit the
   *whole* `can name with Type entry { <body> }` unit, but the task template's hole
   sat *inside* that unit. Splicing the model's full unit into the inner hole
   produced doubly-nested, uncompilable Jac — regardless of whether the model's
   actual code was right. Base, warm-started (SFT loss 0.006 — genuinely converged),
   and every GRPO adapter scored identically **because the splice, not the model,
   was broken.** Every "RL changed nothing" reading from Attempts 1–2 was measuring
   nested garbage. Fixed by `unwrap_unit`: strip one enclosing unit before splicing.
4. **The real null, after the splice was fixed.** With correct splicing and real
   reward variance (σ up to 0.11, training loss 0.02–0.05 — a live, working gradient
   descent), GRPO's output was still byte-identical to its warm-start, KL≈0. At a
   feasible learning rate (≤1e-5) and 300 iterations, LoRA-GRPO barely perturbs a
   30B model's greedy decoding. `pass@8` re-measurement confirmed `pass@1 == pass@8`
   — this null was real, not a greedy-decoding blind spot.
5. **Dense/large-MoE models don't fit at all.** Dense Qwen3.6-27B OOM'd at
   iteration 1 in every config tried — a dense model activates all 27B params per
   token, so activation memory blows 48GB regardless of LoRA rank. Swapping to a
   256-expert Qwen3.6-35B-A3B (MoE) OOM'd too, for a different reason: that many
   experts stay resident in memory (~18GB at q4) and backprop through them still
   exceeds 48GB. Only Qwen3-Coder-30B-A3B's *fewer*-expert, ~3B-active design fits.
   The whole Qwen3.6 line is inference-only on this hardware.

**STaR** (iterate warm-start, keep verified wins) added a genuine but transient
flicker — one round reached pass@4 = 25% — that didn't persist; greedy never left
the SFT floor.

**Verdict at the time**, reasoned honestly from the evidence available: "supervised
levers (SFT, DPO, warm-start) move the model here; RL does not." That conclusion
about *this specific attempt* was correct — the σ=0 trap and splice bug were real,
found, and fixed, and the resulting null (#4) held up under a pass@8 re-check. What
it got wrong was assuming this generalized to "RL can't work on Jac at all," which
Era 2 would show rested on a different, still-undiscovered bug.

### Era 1 — the SFT ladder, and a false verdict (Jun 25–28)

A properly leak-free ladder was built to settle the question at scale: train-N ∈
{1,3,5,10,20,all}, conditions {base, SFT, SFT+GRPO, raw-GRPO control, tuned-GRPO},
2 models × 30 cells, corpus grown from 51 to 84 deterministic tasks, sound
train/holdout splits, monotone reward, Wilson CIs throughout.

**Result, on the (still-broken) eval:** greedy pass@1 came back **flat** in every
single cell — ~26.7% on the 66-task corpus, 11.1% on the 84-task corpus — no matter
which condition. The Jac specialist model was frozen at the exact same number
(26.67 / 11.11 / 21.43%) across every row of the ladder. Nothing — not more SFT
data, not GRPO, not tuning GRPO's hyperparameters — moved the needle at all.

**Why this looked so convincing:** a real signal (even a weak one) usually varies
at least a little between conditions. A number that's *exactly* flat across 30 cells
built with real training runs is a strong prior that you're not measuring the
model — you're measuring some fixed artifact of the harness. That's what it was
(Era 2), but at the time it read as definitive evidence.

**v1 verdict (Jun 28):** "RL is a dead end." Declared done. Every number behind
that sentence rested on one shared piece of code.

### Era 2 — the correction (Jul 1–2): why the flat numbers were a broken ruler

The eval script and the GRPO reward function shared one helper —
`extract_jac`/`unwrap_unit` — to pull the model's answer out of its raw output.
When the model echoed back the *entire* driver file around its answer (a common,
otherwise-harmless habit), that extractor's boundary logic grabbed the driver's
docstring instead of the model's actual completion. The docstring, spliced into the
grading harness, auto-failed every time — a clean, structural, ~3.5–4× undercount
that had **nothing to do with model capability.**

**Why this explains both eras' nulls:** in Era 1, this bug silently caps every
cell's ceiling regardless of training condition — SFT, GRPO, more data, none of it
matters if the grader auto-fails a fixed fraction of genuinely correct answers, so
of course the ladder looked flat. And because the *GRPO reward* used the same
extractor, Era 0's and Era 1's RL runs weren't just being scored wrong — they were
being **trained** against a partially garbage reward signal the entire time,
independent of Era 0's already-confirmed LoRA-GRPO-is-weak result. Two real
problems (a weak RL method, and a broken grader) were stacked on top of each other
and looked like one bigger problem than either was alone.

Fixed in commit `8164ee2` — name-targeted, brace-matched body extraction that
doesn't get confused by an echoed driver. Re-measured on the same holdout (n=18):

| greedy pass@1 | broken extractor | fixed extractor |
|---|---|---|
| fresh qwen3coder | 11.1% | **33.3%** |
| jac-qwen3coder | 11.1% | **38.9%** |

≈**3.5× undercount**, uniformly. Everything from this point on is measured with the
fixed extractor.

**Takeaway:** verify the grader before trusting a null result. A flat, convincing-
looking null can be a broken ruler, not a finding.

---

## Corrected results — the full ladder (jac-qwen3coder, pure-fn holdout n=18)

| cell | greedy pass@1 | oracle pass@8 | best-of-k deploy | why |
|---|---|---|---|---|
| base | 38.9% | 72.2% | 72.2% | true capability once measured correctly — the fresh model's real floor for Jac, not zero |
| SFT rung-5 | 55.6% | **83.3%** | — | a small, low-conflict sample already teaches most of the syntax fast |
| **SFT rung-20** | **61.1%** (peak) | 72.2% | — | sweet spot: enough coverage to generalize, not yet enough to introduce cross-task conflict |
| SFT rung-all | 55.6% | 72.2% | 77.8% | **task interference** — the larger pool pulls in harder graph-walker examples that compete with pure-fn skills for the same limited LoRA capacity, regressing one already-learned task (`lib_log`) |
| SFT + GRPO (rung-all) | 55.6% | 77.8% | 77.8% | flat vs. SFT alone — consistent with Era 0's finding that LoRA-GRPO barely perturbs a 30B's argmax once a strong policy already exists to perturb |
| raw-GRPO control | 38.9% | 72.2% | — | equals base exactly — confirms GRPO alone cannot manufacture syntax knowledge the base doesn't have; it needs an SFT warm-start providing correct samples to reinforce in the first place |

**Why SFT works but "more SFT" doesn't:** the model isn't missing programming
logic — it's missing exposure to Jac's specific surface syntax. A small, curated
set of examples is enough to shift that; a bigger, more heterogeneous set starts
trading one skill for another inside the same fixed-rank adapter (task
interference). More data ≠ better once you're past the point where new data starts
conflicting with what's already been learned.

---

## Why the syntax gap exists, and why the compiler closes it for free

- jac-qwen3coder: pass@1 38.9% vs. pass@8 66.7% — a **+27.8pp gap**. The correct Jac
  is *reachable*; the model produces it inside 8 samples. Greedy decoding just
  doesn't default there.
- Failures are consistently **compile-fails, not wrong answers** — a missing `;`,
  `here.jid` vs. `jid(here)`. Output similarity, "runs," and "exact" all move
  together: when the model's Jac runs at all, it's almost always exactly right.
- The fresh model has **no gap at all** (pass@1 == pass@8 == 33.3%) — sampling finds
  nothing it doesn't already produce greedily, because it has no latent Jac syntax
  knowledge to surface. This is a real dead end, not an undertrained one.

**Why this pattern shows up:** the model was pretrained broadly on many languages
and DSLs, so its underlying programming logic generalizes fine to Jac's semantics —
but Jac's concrete syntax (statement terminators, its specific idiom for `here`)
is a narrow, low-frequency surface form the model hasn't overfit to. At greedy
decoding it drifts to a neighboring, more common syntax pattern that happens to be
almost — but not exactly — valid Jac. Sampling works because the *correct* token
sequence still has real, nonzero probability mass; it's just not the single
highest-probability (argmax) choice.

**Why this makes the Jac compiler a free, perfect verifier:** because correctness
and compilability are so tightly coupled here (compiles ⟹ almost always exactly
right), you don't need a learned reward model, a human grader, or ground truth at
inference time — sample k candidates and keep the first one the compiler accepts.
Deploy accuracy equals oracle accuracy throughout every measurement in this
document, because the compiler picks as well as an oracle would.

---

## Why conversion beats hole-fill by +9–11pp

On identical underlying functions: conversion 63.6%→72.7% greedy vs. hole-fill
38.9%→61.1% greedy (base→SFT). **Why:** hole-fill under-specifies the target — the
model has to infer intent from the surrounding driver/docstring before it can even
attempt the syntax, adding an extra inference step where mistakes creep in.
Conversion tasks hand the model an unambiguous ground-truth spec (working Python)
and reduce the problem to one transferable skill: syntax transliteration. There's
nothing left to guess.

The tradeoff: conversion can only source **pure functions** — Python has no
`walker`/`node`/`edge` construct to translate from, so the graph-walker idiom
structurally cannot be taught this way. It needs its own dataset (see the
graph-walker fix plan below).

---

## Why GRPO lands at "≈ SFT, no extra lift" — and why that's a real result, not a bug

Across every measurement here — the corrected ladder, the extended holdouts, Era
0's careful reward-engineering pass — GRPO on top of a working SFT policy adds
nothing measurable, and raw GRPO from the base moves nothing at all. This is
consistent with published findings on RL post-training (Yue et al., 2504.13837):
**RL sharpens a model's *sampling distribution* around capability it already has —
it doesn't expand the underlying capability boundary.** Once SFT has already moved
the model's greedy output to something close to its own pass@k ceiling, there's
little room left for GRPO to sharpen further, and LoRA's limited capacity (see task
interference above) leaves even less headroom to search for anything past that
ceiling.

The practical implication: **expanding** the boundary — not just sharpening
sampling within it — needs either full fine-tuning (removing LoRA's capacity
ceiling entirely, untested here, needs >48GB) or **distillation / expert
iteration** (training on a stronger teacher's or the model's own compiler-verified
best outputs — injecting capability from outside rather than re-weighting what's
already there). That's why expert iteration is flagged as the one method with real
remaining headroom in Open Leads below.

---

## Follow-ups: tightening the result

**E1 — the true ceiling is ~94%, not 72%.** Two holdout tasks
(`lib_fstring_rules`/`lib_innerstring_rules`) turned out to reward verbatim regex
memorization, not real Jac ability. Dropping them (n=16, the "clean set") revealed
base best-of-k **93.8%** — the earlier 72% number was being dragged down by two
tasks that weren't testing what they claimed to.

**Exploit vs. explore, in the same data:** SFT sharpens greedy accuracy
(43.8%→68.8%) but very slightly *narrows* the sampled distribution — base best-of-k
(93.8%) edges out SFT's best-of-k (87.5%) by one task. **Why:** SFT concentrates
probability mass toward its single best answer, which improves the odds that
argmax *is* that answer (higher greedy) at a small cost to how diverse the model's
sampled alternatives are (slightly lower best-of-k ceiling). If you're taking one
shot, SFT wins; if you have a sampling budget, base + best-of-k edges it out.

**E2 — the one real gap: free-form natural language.** Both the hole-fill-SFT and
conversion-SFT models score **0/3** on arbitrary "write a function that…" prompts
with no starter code. Neither model has been trained on this input distribution —
they're tuned for the task *format* used throughout this document, not for
open-ended NL. Not something model choice fixes; needs its own SFT data.

**E3 — the SFT lift generalizes, it isn't memorized.** On a bigger, fresher holdout
(n=32, +16 synthetic tasks that didn't exist at training time): greedy 34.4%→43.8%,
best-of-k 62.5%→71.9% — the lift holds. Fresh/synthetic tasks are harder than
familiar ones (25% vs. 44% base greedy), which is the honest signal that these
numbers reflect real generalization, not an inflated, memorization-friendly
holdout.

---

## Across all five holdouts

| holdout (n) | greedy: base→SFT | best-of-k: base→SFT |
|---|---|---|
| pure-fn (18) | 38.9%→61.1% | 72.2%→77.8% |
| graph-walker (17) | 35.3%→**29.4%** (regresses) | 52.9%→64.7% |
| conversion (11) | 63.6%→72.7% | 72.7%→**81.8%** (peak) |
| clean (16) | 43.8%→68.8% | **93.8%**→87.5% (SFT narrows sampling, see E1) |
| big+fresh (32) | 34.4%→43.8% | 62.5%→71.9% |

Conversion is the highest-scoring family throughout; graph-walker is the one family
where SFT actively hurts the greedy number — see the dedicated section below.

---

## The graph-walker weak spot, and the fix plan

Graph-walker is the **only** family where SFT makes the greedy number worse
(35.3%→29.4%) — every other family improves. Best-of-k still climbs (52.9%→64.7%),
so this isn't a capability gap, it's the clearest live example of **task
interference**: the SFT training mix is dominated by hole-fill/conversion examples,
so graph-walker examples get statistically outvoted during training and the model's
greedy default for *this* family drifts away from correct, even while sampling
still finds the right answer.

Fix plan, in priority order:

1. **A dedicated spec→jac dataset for the graph-walker idiom.** Conversion
   structurally can't source these examples — Python has no `walker`/`node`/`edge`
   to translate from — so this family needs synthetic tasks authored directly
   against the OSP spec.
2. **Curriculum-weighted SFT, or a separate per-family adapter.** This fixes the
   interference mechanism itself rather than just adding more data (which risks
   the same problem in reverse, this time hurting hole-fill/conversion). Weighted
   sampling gives graph-walker proportional gradient signal instead of being
   outvoted by hole-fill/conversion volume; a per-family LoRA adapter, routed by
   task type at inference, sidesteps the shared-capacity problem entirely.
3. **A zero-training stopgap, shippable today:** detect graph-walker prompts in
   `rl/generate.py` and bump `k` specifically for that family (k=32 instead of the
   default) — best-of-k already recovers most of the gap, this just leans on that
   harder for the one family that needs it.

\#1 and \#2 are the actual fix. \#3 is a band-aid that can be flipped on right now
while \#1 and \#2 are being built.

---

## The deployable answer, today

**Conversion prompting + SFT (rung-20) + best-of-k with the Jac compiler as
verifier.** No further training needed to ship these numbers:

| where | best-of-k accuracy |
|---|---|
| pure functions | ~78% |
| conversion tasks | 82% (peak) |
| clean pure-fn subset | 94% (ceiling, not the general number) |
| pure-fn at k=32 | 89% |
| graph-walker | 65% (the acknowledged weak spot, see above) |
| free-form NL prompts | 0% (untested gap, don't ship this path yet) |

Shipped as `rl/generate.py`.

---

## Shipped

- **`rl/generate.py`** — the best-of-k Jac generator (sample k, return the first the
  Jac compiler accepts). Live-verified; deploy == oracle throughout.
- **Studio RL section** — the full story (11%→94% journey), the ladder, best-of-k
  across all 5 holdouts, k-scaling, and a live **GENERATE JAC** panel. Backed by
  `get_rl_corrected()` → `resultspub/rl/corrected_summary.json` (assembled by
  `rl/make_summary.py`).
- **Studio Data pipeline** — RL datasets browsable by use: SFT gold · GRPO reward
  tasks · conversion · synthetic · splits.
- **Graphs** — `resultspub/rl/` (journey, ladder, all-holdouts, k-scale, follow-up).
- **32 synthetic tasks** — `rl/drivers/syn_*.jac`, ingested by `rl/build_tasks.jac`.
- **Raw record** — `docs/rl/raw/`, `results/corrected_*.jsonl`.

---

## Open leads (not yet done)

- **NL→jac SFT** — close the E2 free-form gap, the one measured weakness left.
- **Graph-walker fix** — see the dedicated section above; dataset + curriculum fix
  is the real lever, the k-bump stopgap ships today.
- **Bigger generated dataset** — every n here is still small (±8–12pp Wilson-CI
  noise); more synthetic families would tighten every number in this document.
- **Distillation / expert iteration** — per the GRPO section above, this is the one
  untested method that could expand the capability boundary rather than just
  sharpen sampling within it.
- **Full fine-tune** (cloud / >48GB) — the only untested way to check whether GRPO
  can beat SFT once LoRA's capacity ceiling is off the table.

---

## References

Yue et al. 2504.13837 — RL sharpens sampling, doesn't expand the capability
boundary (cited above as the likely explanation for "GRPO ≈ SFT").
ProRL 2505.24864 — prolonged RL *can* expand the boundary, but needs full
fine-tuning and thousands of tasks, neither available here.
Spurious Rewards 2506.10947.
Full notes: `docs/rl/references.md`. Original Era-0 write-up (verbatim):
`docs/rl/RL_WEEKEND_RESULTS.md`. Pre-correction ladder findings (superseded,
archived for the reasoning trail): `docs/rl/RL_FINDINGS_v1_invalidated.md`.
