# CPT-v2 results and acceptance verdict

2026-07-18. Companion to [design.md](design.md) (spec) and [implementation-plan.md](implementation-plan.md) (Tasks 1-19). Verdict computed by `03-new/cpt_train/eval_v2/acceptance_readout.py`, recorded at `03-new/results/cpt-v2/acceptance.json`.

## Verdict: REJECTED

**CPT-v2 does not clear design.md section 6.4's acceptance bar.** Both tracks fail, one of them (Track B) by a wide margin:

| bar | required | actual | pass? |
|---|---|---|---|
| Track A margin vs base | ≥ +0.03 cosine | +0.008 | ❌ |
| Track A margin vs cpt-v1 | ≥ +0.03 cosine | +0.001 | ❌ |
| Track B win-or-tie rate | ≥ 0.50 | 0.09 | ❌ |

Same discipline as CPT-v1's null (`03-new/docs/cpt-1/analysis.md`): stated plainly, not rounded up. The corpus rebuild, curation pass, and epoch-loop training all executed cleanly and as designed — the mechanism worked, the hypothesis it was built to test (does docs-dominant, curated, fully-annealed CPT move the semantic ceiling) did not survive contact with a harder, source-grounded eval.

## 1. Corpus actually built

| source | design.md projected | actual built | windows |
|---|---|---|---|
| docs (incl. jac-llmdocs) | 1.96M | 2,061,194 | 553 |
| osp_paper | 35K | 44,624 | 12 |
| blogs | 64K | 66,011 | 17 |
| rehearsal (codeparrot, CF-insurance) | ~230K | 234,847 | 64 |
| code (17-repo) | 0 (dropped) | 0 | 0 |
| **total** | **~2.29M** | **2,406,676** | **646** |

Actual build landed ~5% over the design's projection (real chunking produced slightly more tokens than the back-of-envelope estimate) — close enough that the projected corpus shape (docs-dominant, code fully dropped, rehearsal ~10%) held as designed. 85/15 split → 544 train windows / 102 val windows, matching `manifest.json`.

**Fable curation pass** (`03-new/dataset/cpt-v2/curation.json`) reviewed 10,157 raw chunks pre-pack: 7,136 kept as-is, 471 upweighted, 2,550 dropped (noise, boilerplate, near-duplicate, or off-topic per Fable's verdicts).

## 2. Epoch-loop training (Task 13)

12 legs, floor-6/target-8/ceiling-12 schedule, one coherent cosine LR decay across all legs (not per-epoch restarts). Every leg passed the CF-check (16/16 general-Python coding tasks, exact-output graded) — **zero regression across the entire run**, all 12 `cf_check_*.json` snapshots inspected, all clean/idiomatic/correct.

| leg | train loss | val loss | final LR | CF-check | decision |
|---|---|---|---|---|---|
| 1 | 1.756 | 1.371 | 8.33e-06 | 16/16 | continue |
| 2 | 1.251 | 1.158 | 9.90e-06 | 16/16 | continue |
| 3 | 1.365 | 1.218 | 9.51e-06 | 16/16 | continue |
| 4 | 0.898 | 1.107 | 8.85e-06 | 16/16 | continue |
| 5 | 1.000 | 1.076 | 7.95e-06 | 16/16 | continue |
| 6 | 1.065 | 0.974 | 6.89e-06 | 16/16 | continue |
| 7 | 0.886 | 0.976 | 5.74e-06 | 16/16 | continue |
| 8 | 0.613 | 0.823 | 4.57e-06 | 16/16 | continue |
| 9 | 0.497 | 0.893 | 3.46e-06 | 16/16 | continue (val uptick) |
| 10 | 0.451 | 0.784 | 2.49e-06 | 16/16 | continue |
| 11 | 0.392 | 0.942 | 1.73e-06 | 16/16 | continue (val uptick) |
| **12** | **0.319** | **0.783** | **1.22e-06** | **16/16** | **halt_keep_this (ceiling)** |

Net: train loss −82% (1.756 → 0.319), val loss −43% (1.371 → 0.783). Legs 9 and 11 show val-loss upticks against a monotonically-falling train loss — a plausible early-stop signal — but leg 12 recovered to tie leg 10's best val loss (0.783 vs 0.784) with the run's best train loss and a clean CF-check, which reads as oscillation around a floor rather than genuine overfitting past leg 10. Stopped at the ceiling as designed: the LR schedule was fully spent (final LR at its designed floor), extending past leg 12 was outside the approved plan, and there was no signal more training would move val loss further. Fused to `models/qwen-cpt-v2-fused-q4`.

## 3. Track A — cosine similarity to jac-gpt oracle (n=100)

Base, cpt-v1, and cpt-v2 each generated 300-token answers to the same 100 questions; jac-gpt (RAG-grounded oracle, FAISS + cross-encoder reranking over ~89 real docs) answered the same questions; all three scored via `all-mpnet-base-v2` cosine similarity to the oracle's answer.

| model | mean cosine-to-oracle |
|---|---|
| base | 0.8051 |
| cpt_v1 | 0.8126 |
| **cpt_v2** | **0.8133** |

Paired per-question analysis:
- cpt_v2 vs base: mean delta **+0.0081**, t=2.07, 55 wins / 45 losses — a real but tiny edge, well under the 0.03 acceptance margin.
- cpt_v2 vs cpt_v1: mean delta **+0.0007**, t=0.22 — **statistically indistinguishable from noise**, 47 wins / 53 losses (a coin flip).

**Honest-gap spot check** (design.md §6.2's discipline: cosine similarity can reward verbose-but-wrong answers over correct-but-differently-worded ones — don't trust the aggregate blind):

- `b003-q-any-inference` (untyped Python call → `any` type inference): cpt_v2 scored **0.9355** cosine — the *2nd-highest* score in the whole set — yet **lost** the blind Track B judgment outright. cpt_v2 hedged ("may infer or leave as unknown") instead of confirming the passage's actual rule (untyped results are typed `any`); the oracle answer stated the rule correctly. High similarity, wrong answer — exactly the failure mode the design doc warned about.
- `b003-q-here-vs-self` scored highest overall (0.9461) and *did* correspond to a genuine Track B tie — both answers were in fact comparably correct here, so cosine wasn't misleading in this case.
- Lowest cpt_v2 scores (`b001-q-jac-dir-artifacts` 0.53, `b003-q-persist-reachability` 0.59, `b001-q-unrooted-nodes` 0.59, `b000-q-client-rpc` 0.59, `b000-q-scale-builtin` 0.66) cross-checked against Track B: all five also lost their blind pairwise judgment to the oracle — low cosine and blind-judge loss agreed in these cases, unlike the `any-inference` case above.

Net read: cosine similarity is not reliable enough on its own to call this track a pass or fail — it agrees with the blind judge most of the time but materially disagreed on at least one high-stakes case. Track B is the tie-breaker, and it's unambiguous (next section).

## 4. Track B — blind pairwise judge vs jac-gpt oracle (n=100, judged directly by Sonnet, source-blind)

| outcome | count |
|---|---|
| oracle wins | 91 |
| cpt_v2 wins | 7 |
| ties | 2 |
| **cpt_v2_win_or_tie_rate** | **0.09** (need ≥ 0.50) |

The oracle won overwhelmingly and by a wide margin — not a marginal miss. Reading all 100 justifications during judging, the recurring failure mode in cpt_v2's losing answers was **inventing fictional or wrong-domain syntax and APIs** where the oracle answer stayed grounded in the real Jac syntax shown in the source passage: Neo4j/Cypher query syntax, Gremlin/JanusGraph, generic OOP/tree-traversal pseudocode, fabricated Jac constructs (`struct` keyword, `graph {}` blocks, `node.add_child()`/`node.bfs_traverse()`, inline `sem:` annotations, an invented `@jac/desktop` plugin restriction), and outright framework misidentification (answering a Jac routing question as if it were Next.js, or a fullstack RPC question as "Jamstack, not Jac").

Representative oracle-win justifications (source-blind judgments, one per distinct topic):

- *def vs can*: "Answer B's can example correctly uses 'with entry' matching the passage's event-triggered ability framing, while Answer A invents non-idiomatic can syntax ('with name -> string') and mischaracterizes def as merely 'standalone functions' rather than object methods."
- *walker ability dispatch*: "Answer B correctly explains type-matched entry/exit abilities... while Answer A invents a fictional abilities-list/priority system with syntax not used in real Jac."
- *route groups*: "A misidentifies the passage as Next.js routing rather than Jac's own file-based routing, while B correctly frames it as Jac..."
- *unrooted nodes*: "A correctly states unrooted nodes behave like regular in-memory objects that are garbage collected after execution... while B fabricates an unrelated TensorFlow/PyTorch computational-graph framing."
- *node gatekeeper (visitor/disengage)*: "B closely mirrors the passage's Gateway example with correct Jac syntax (`visitor.clearance`, `disengage`), while A gives generic Python pseudocode with no Jac grounding and cuts off incomplete."

The 7 cpt_v2 wins and 2 ties were genuine, not blowouts either way — e.g. cpt_v2 correctly called out that `spawn :>`/`spawn |>` piped forms no longer work (oracle wrongly claimed they still do), correctly distinguished `|` vs `|=` for dict merge, and correctly affirmed that named arguments can reorder freely around a defaulted parameter (oracle invented an unsupported restriction in that one case). The ties (`here` vs `self`, deferred-exit LIFO ordering) were both cases where both answers were genuinely, comparably correct. These aren't asymmetric flukes — the oracle can be wrong too — but at 7 wins + 2 ties against 91 losses, cpt_v2 is not competitive with the RAG-grounded oracle on this eval.

## 5. Why Track A and Track B disagree in magnitude (but agree in direction)

Both tracks point the same way — cpt_v2 does not beat the oracle-grounded eval — but Track A's aggregate (a coin-flip vs cpt_v1, a small real edge vs base) looks far less damning than Track B's 91-7-2. This is consistent with design.md §5's original reason for building Track B at all: cosine similarity to a reference answer rewards stylistic/topical proximity (same vocabulary, same shape of answer, same length) independent of whether the specific claims in the answer are true. A model that has absorbed the *style* of Jac documentation from CPT (which cpt_v2 plausibly did — mean cosine did rise vs base) can still get the *facts* wrong in ways a same-style oracle answer would not, and a blind pairwise judge reading for correctness against the actual source passage catches that in a way a single similarity number cannot. That's exactly what the `any-inference` spot-check above shows directly.

## 6. Bottom line

CPT-v2's engineering executed cleanly: the docs-dominant curated corpus built as designed (±5% of projection), the epoch-loop training ran all 12 legs with zero CF regression and a real ~43% val-loss improvement, and both eval tracks ran to completion exactly as specified. None of that changes the outcome — semantic correctness against a RAG-grounded reference did not move enough to accept. Per §6.4, CPT-v2 checkpoint is **not accepted**; Phase 4 (SFT/DPO) should not build on it as-is, same posture as CPT-v1's original null.
