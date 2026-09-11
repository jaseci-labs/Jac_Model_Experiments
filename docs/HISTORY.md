# History: experiments 01 to 08

The goal throughout: a model that writes **idiomatic, compiler-correct Jac**
(generate, debug, explain, convert), served through the Jac MCP in coding
assistants. Jac builds on Python but adds object-spatial constructs (`node`,
`edge`, `walker`, `can ... with <Node> entry`, `spawn`, `++>`, `visit`,
`has`). A stock model writes Python-shaped code that looks right and doesn't
compile. There is no real Jac corpus to scrape, so all training data is
synthetic and checked by the compiler. The gate is always `jac run`, never
`jac check`, which over-rejects idiomatic untyped Jac.

Hardware for everything: one Apple Silicon Mac with 48 GB unified memory, MLX
LoRA. Base model since 01: **Qwen3-Coder-30B-A3B-Instruct** (MoE, about 3B
active), at 4 bits.

The directories for 01 to 07 were deleted on 2026-09-11. Every experiment that
left an adapter (04, 06, 07, 08) now lives under [../experiments/](../experiments/),
each as `adapter/`, `results/` and `report.md` (08 also keeps its `dataset/` and
`docs/`). The remaining reports of 01 to 05 are kept, unedited except for dead
links, in [history/](history/).

## Summary

| # | Method | Headline | Verdict |
|---|---|---|---|
| 01 | SFT + DPO, 6-model base bake-off | function tier 0% to 94% after one SFT pass; graph tier 46% (SFT) to 61% (DPO), n=13 | SFT works; kept Qwen3-Coder as the base |
| 02 | GRPO on top of 01 | greedy 38.9% to 61.1% from SFT; GRPO added nothing; best-of-k with the compiler as verifier ~94% on clean pure functions | SFT moves the model, GRPO doesn't |
| 03 | Continual pretraining on Jac docs | CPT-v1 null (MCQ 18/20 before and after); CPT-v2 cleared 0 of 3 acceptance gates | Rejected |
| 04 | SFT/DPO on CPT-v2 vs fresh base; Spectrum vs stock layers | CPT edge +37 pp at base, +2.8 pp (p≈0.20) after SFT; Spectrum SFT 74.7% vs stock 69.8% (p=0.023) | CPT washes out; Spectrum wins; best adapter to date |
| 05 | GRPO on both 04 arms | never built | Docs only |
| 06 | Spectrum replication on Nitin's dataset v1 | Spectrum SFT 74.4%; 1 significant win, 1 significant loss | Spectrum not dataset-invariant; data about as good as 04's with 32% fewer rows |
| 07 | Same, Nitin's dataset v2 (24% more rows) | Spectrum SFT 72.4%, DPO-best 73.0%; 0 of 6 cells significant vs 06, all six lower | Bigger pool trended worse |
| 08 | Spectrum SFT on a 7-source merged corpus (2.18x 07) | 70.5% on holdout (a); 37.7% on 07's holdout (b) | Lowest of the lineage |

Holdout (a) scores: 855 code-graded rows of the shared holdout
(`experiments/08-nitin-new2-ds/dataset/holdout_a_shared855.jsonl`), same base, same
harness, from 04 onward. 01 to 03 used different instruments and cannot be put
in the same column.

---

## 01: SFT + DPO

**Goal.** Show that LoRA SFT on synthetic, compiler-validated data takes a
model from no Jac to mostly-correct Jac, then use DPO to push idiomatic style.
Pick the base model.

**What ran.** Data came from three anchors: Jac grammar (coverage), the Jac
compiler plus cross-compiled tests (the oracle), and Python as a proxy
distribution (translate validated Python to Jac, MultiPL-T style). 1,500 rows
from `jac py2jac` on mined runnable functions, 147 hand-written idiomatic rows
including graph-tier tasks, and 147 DPO pairs (idiomatic chosen,
transpile-shaped rejected). LoRA r16, 600 iters. The same treatment then ran on
five other candidate bases.

**Result.**

| stage | function tier (n=150) | graph tier (n=13) |
|---|---|---|
| base | 0% | 0% |
| SFT | 94% | 46% |
| DPO | 93% | 61%, all correct outputs idiomatic |

On pure functions an idiomatic answer is basically the transpile, so DPO had
nothing to push. The graph tier is where idiom matters, and DPO helped there.
Bake-off: no candidate beat Qwen3-Coder beyond run-to-run noise.
Qwen3-30B-A3B-Instruct tied on behavior and wrote slightly more idiomatic graph
code, but a tie keeps the incumbent.

**Verdict.** SFT works. Qwen3-Coder-30B-A3B stays the base.
Reports: [01-sft-dpo-phase.md](history/01-sft-dpo/01-sft-dpo-phase.md),
[bake-off](history/01-sft-dpo/2026-06-26-sft-dpo-bakeoff-results.md).

## 02: RL / GRPO

**Goal.** Starting from 01's SFT+DPO model, see whether GRPO with the Jac
compiler as a free, verifiable reward pushes correctness further.

**What ran.** Three eras over about two weeks. Era 1: GRPO on MLX LoRA, flat
at 14.3%. Two real bugs fixed along the way: the σ=0 trap (a group where every
rollout fails has zero variance and zero gradient) and a splice bug that
nested the model's output inside an existing unit. Era 2: a 30-cell ladder
(train-N x {base, SFT, SFT+GRPO, raw GRPO, tuned GRPO} x 2 models), perfectly
flat everywhere, declared a dead end. Era 3: the eval and the reward shared an
extraction helper that grabbed the driver's docstring whenever the model
echoed its surrounding file. That was a ~3.5x undercount in every measurement,
and it also corrupted the GRPO reward. Fixed, re-measured: 11.1% became 38.9%.

**Result (corrected, pure-function holdout n=18).** Base 38.9% greedy. SFT
peaked at 61.1% with 20 examples; the full mix dropped back to 55.6% (task
interference). SFT+GRPO 55.6%, no better than SFT. Raw GRPO from the base
equalled the base exactly. Best-of-k with the compiler picking the first
sample that runs: 82% on conversion tasks, ~78% on pure functions (94% on the
cleanest subset), 65% on graph-walker tasks, 0% on free-form prompts.

**Verdict.** SFT moves the model. GRPO doesn't add anything once SFT has run,
and can't create a skill the base lacks. Failures are mostly compile errors,
so the compiler works as a free verifier at inference time.
Report: [RL_FINDINGS.md](history/02-rl-grpo/RL_FINDINGS.md).

## 03: continual pretraining (CPT)

**Goal.** Test whether the ceiling after 01/02 came from missing Jac/OSP
*semantics*, and whether LoRA CPT on raw docs, the OSP paper and blogs (with
general-code rehearsal) would add them before SFT.

**What ran.** CPT-v1 (2026-07-14): clean training, catastrophic-forgetting
check 16/16, but semantic MCQ scored 18/20 before and after with the same two
wrong. CPT-v2 (2026-07-18/20): docs-only corpus, curation pass, 12-epoch
checkpoint loop.

**Result.** CPT-v2 cleared 0 of 3 acceptance gates. Cosine-to-oracle margin
+0.008 vs base (needed +0.03) and +0.001 vs CPT-v1. A blind pairwise judge
preferred the RAG oracle in 91 of 100 cases (needed 50% win-or-tie). The model
learned to write fluent, plausible, wrong-domain syntax (Next.js, Cypher)
instead of admitting it didn't know.

**Verdict.** Rejected. This is a structural limit of next-token CPT on doc
prose, not a bad run. Skip further CPT.
Reports: [cpt-1-analysis.md](history/03-cpt-only/cpt-1-analysis.md),
[cpt-2-results.md](history/03-cpt-only/cpt-2-results.md).

## 04: SFT/DPO on CPT-v2 vs fresh base, and Spectrum

**Goal.** Check whether CPT-v2 still helps once real SFT runs on top, with a
different instrument than 03. Then test Spectrum layer selection against the
stock trailing-16 LoRA placement.

**What ran.** New data pipeline (`jacgen2`, 8,100 SFT rows, 654 DPO pairs).
A new holdout: 1,428 rows across 7 categories, 855 code-graded, which became
**holdout (a)** for every later experiment. Two arms (fresh base, CPT-v2 base)
with the identical recipe: 8,200 SFT iters, then 250 DPO iters. Then Spectrum
on both arms.

**Result, CPT vs fresh (stock layers).** Base 10.5% vs 47.3% (+37 pp). After
SFT 69.8% vs 72.6% (+2.8 pp, p≈0.20). DPO-best tied SFT in both arms; DPO run
to 250 iters lost ~8 pp in both. The first DPO results showed both arms at
2 to 12%. Cause: `mlx_lm.fuse` re-quantizes the 4-bit base and rounds away the
SFT delta, so DPO had been training on an untrained base. Fix: no fusing, seed
DPO from the SFT adapter with `--resume-adapter-file`. A q_proj SVD probe found
CPT-v2's fingerprint still visible in the SFT adapter's weights even though it
had no behavioral effect.

**Result, Spectrum vs stock** (fresh arm, holdout a): SFT 74.7% vs 69.8%
(p=0.023), DPO-best 74.2% vs 69.8% (p=0.046), DPO-final 72.7% vs 62.1%
(p<0.0001). On the CPT-v2 arm there was no significant difference. A fifth arm
(CPT itself on the Spectrum layers) was paused mid-run and never finished.

**Verdict.** CPT's head start washes out under SFT. DPO caps at SFT parity.
Spectrum is a real win on a fresh base. The fresh-arm Spectrum SFT adapter,
**74.7% (639/855), is still the best model in the lineage** (in
[../experiments/04-cpt-sft/adapter/](../experiments/04-cpt-sft/adapter/)).
Reports: [cpt-vs-fresh](history/04-cpt-sft/2026-07-cpt-vs-fresh-comparison.md),
[spectrum-vs-stock](../experiments/04-cpt-sft/report.md),
[five-arms overview](history/04-cpt-sft/five-arms-overview.md).

## 05: GRPO on top of 04 (not run)

**Goal.** Run the GRPO stage of 03's original four-stage plan on both 04 arms,
with a bigger mined corpus and AST-equivalence grading, to see whether CPT's
structural fingerprint shows up behaviorally under RL.

**Result.** Docs only. No harness, data or training was ever built. Given 02
(GRPO ≈ SFT) and 04 (CPT washes out), it was dropped.
Index: [05 README](history/05-cpt-sft-grpo/README.md).

## 06: Spectrum replication on Nitin's dataset v1

**Goal.** RQ1: does Spectrum's win replicate on an independently sourced
dataset? RQ2: is that dataset (`chess10kp/jac-data-gen` @ `7c25aff3`, 5,474
SFT rows, flat Python-shaped functions) better than 04's jacgen2 data?

**Result (holdout a).** Spectrum SFT-final 74.4%, stock 72.4% (p=0.35).
DPO-final 74.4% vs 68.2% (p=0.0046, the one significant win). On its own
in-distribution holdout (b) both arms hit 97 to 99% and stock won SFT-final by
1.4 pp (p=0.014). Against 04's data: 1 of 4 comparisons significant (stock
DPO-final +6.1 pp), none significantly worse.

**Verdict.** Spectrum is not dataset-invariant (1 win, 1 loss, 4 nulls). The
new data is about as good as jacgen2 with 32% fewer rows, and if forced to
choose, better. Incidents: two external-drive disconnects, a best-checkpoint
tracker that reset across a crash-resume, and a holdout (b) with no `messages`
field whose first eval "finished" in 30 seconds without running.
Report: [06 final comparison](../experiments/06-nitin-ds-sft/report.md).

## 07: Nitin's dataset v2

**Goal.** Same two questions on the next corpus version (`@ 11fa3f45`, 9,367
idiomatic py2jac records, 6,781 SFT rows, 24% more than 06, with richer typed
Jac).

**Result (holdout a).** Spectrum SFT-final 72.4%, DPO-best 73.0%, DPO-final
71.0%. Spectrum beat stock significantly at DPO-best (+4.8 pp, p=0.030) and
DPO-final (+4.7 pp, p=0.037), with no significant losses anywhere. Against 06:
0 of 6 cells significant, but **all six point estimates were lower** (closest
p=0.057). Holdout (b), 07's own: 97 to 98% for both arms, ceilinged.

**Verdict.** Spectrum is recurring but its size depends on the dataset. More
and "better-written" rows did not give a better model. Significance was the
unpaired z-test only; the planned paired McNemar was never computed because no
per-row results were saved.
Report: [07 final comparison](../experiments/07-nitin-ds-new-sft/report.md).

## 08: seven-source merged corpus

**Goal.** Fold every new `jac-data-gen` source (at `c95b7563`) into one pool:
js2jac translations, filtered composer py2jac, farm CRUD walkers, osp
graph-native modules, step4 work rows, and step4 DPO pairs. First eval-denylist
filter in the lineage, a test-count quality filter aimed at 07's "bigger but
weaker" pool, a license filter. 14,792 SFT rows, 2.18x 07's, 15.7%
graph-native. Holdout (b) reused byte-for-byte from 07.

**What ran.** The design called for 2 arms x 2 holdouts x 3 stages. It was cut
twice on 2026-09-11 to **Spectrum SFT only**, evaluated on both holdouts. Stock
SFT trained cleanly but was never evaluated and has since been deleted. No DPO.

**Result.** Holdout (a) 70.5% (603/855). That is numerically the lowest in the
lineage, though not significantly below 06 (p=0.074), 07's SFT-final (p=0.39)
or 04 (p=0.051), unpaired. Holdout (b) 37.7% (322/855), against 98.2% for 07 on
the same file. The likely cause is dilution: only about a quarter of 08's SFT
rows had the py2jac shape holdout (b) tests. That is inferred, not measured.
Incidents: silent NaN from 5 js2jac rows whose prompt alone exceeded
`max_seq_length` (fixed with `nan_guard.jac`), a lid-close sleep that killed
the stock run, a watchdog that blocked its own relaunch.

**Verdict.** Folding in a larger, more varied corpus made the model worse on
both holdouts. 04's 74.7% still stands. 08 is kept as the working template
because its scripts are the most complete and hardened; see
[PLAYBOOK.md](../PLAYBOOK.md).
Report: [08 final comparison](../experiments/08-nitin-new2-ds/report.md).

---

## What carries forward

- Only SFT has reliably moved accuracy (01, 02, 04 to 08).
- Second passes on top of SFT have not beaten SFT: GRPO (02), CPT (03, 04),
  DPO (04, 06, 07; best checkpoint ties, full run regresses).
- Spectrum layer selection is the one lever that has beaten its baseline,
  significant in 04 and 07, never a significant loss on holdout (a).
- Data mix and fit to the eval surface matter more than row count (06 vs 07
  vs 08).
- Grade the grader first (02's docstring bug, 04's fuse bug, 06's empty
  holdout). A flat or collapsed number is a measurement bug until proven
  otherwise.

## Other material

- [reference/finetuning-strategy.md](reference/finetuning-strategy.md): the
  original whole-stack plan from before 01 (it names Gemma as the base; the
  01 bake-off replaced that with Qwen3-Coder). Still useful for its twelve
  synthetic-data recipes.
- [reference/spectrum-layer-sft-probe-design.md](reference/spectrum-layer-sft-probe-design.md):
  the Spectrum design from 04.
- [blog/](blog/): three of the four posts written in early August 2026, after
  04 (CPT control arm, layer selection, overview). The fourth, on the JMS app,
  lives with the app (`jms/docs/blog/jms-model-studio/` in the
  jac_model_studio repo), not here.
- [presentation/](presentation/): slide deck (`slides.pdf`, source
  `slides.tex` / `slides.md`) on 04's five arms, JMS, and NightShift.
