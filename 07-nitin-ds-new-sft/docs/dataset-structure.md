# Dataset structure — 07-nitin-ds-new-sft

> **SCAFFOLD. Every count below is TBD.** No corpus is pinned yet
> (`../CONTEXT_BRIEF.md` §1), so `dataset/candidate_pool.jsonl`,
> `dataset/nitin_holdout.jsonl`, `dataset/nitin_holdout_eval.jsonl` and
> `docs/reports/corpus-triage-report.md` do not exist. Fill these tables from
> those artifacts — do not estimate, do not carry a number forward from 06 or
> `04-cpt-sft`, and do not round a drop count up to a tidy total.

Source: `[TODO: confirm repo]` @ `[TODO: confirm commit]`, `[TODO: n]` files,
one transpiled function per file `[TODO: confirm shape — 06's was
data/jac_outputs/*.jac, docstring + one `def`]`. Both arms train on identical
copies of the releases below. Both arms eval on two holdouts: `04-cpt-sft`'s
unchanged 1,428-row set (855 code-graded), and this phase's own
`nitin_holdout_eval.jsonl`.

## Triage yield

Every row is a real count with an attributed reason. A pool that shrinks a lot
is a finding to report, not a number to smooth over. Stage labels match
`scripts/pipeline.jac`'s own funnel output.

| stage | n | Δ | note |
|---|---:|---:|---|
| S0  files at commit pin | TBD | — | verified by counting the corpus dir |
| S1  compiles (`jac run` exit 0) | TBD | TBD | drops recorded in `/tmp/nitin_new_triage/stats.json` |
| S2  quality drop: no_def | TBD | TBD | |
| S2  quality drop: multi_def | TBD | TBD | more than one top-level `def` |
| S2  quality drop: no_docstring | TBD | TBD | nothing to reverse-author from |
| S2  quality drop: trivial_body | TBD | TBD | < 12 body tokens or < 3 code lines |
| S3  after exact-hash dedup | TBD | TBD | normalized token hash |
| S4  after 14-token-shingle near-dup dedup (≥0.5) | TBD | TBD | |
| S5  after decontam vs the 4 reference holdouts | TBD | TBD | strict 14-token + relaxed identifier-only 10-token |
| S5b after func-name collision drop (ident-Jaccard ≥0.5) | TBD | TBD | |
| **clean pool** (`candidate_pool.jsonl`) | **TBD** | | |
| — carved to `nitin_holdout.jsonl` | TBD | | frozen before any generation, seed 42 |
| — remaining training pool | TBD | | |

Holdout leak-check (holdout vs candidate pool, same shingle machinery):
**TBD** failures — must be 0.

Cross-holdout independence check (`spec.md` §6): near-dups between
`nitin_holdout.jsonl` and `04-cpt-sft`'s holdouts — **TBD** (must be 0, or the
two holdout columns are not independent tests).

`[TODO: if 06's own holdout was added as a fifth decontam reference set (the
open decision in `spec.md` §6 / `scripts/pipeline.jac`), add its row here.]`

## SFT — category / task-type composition (n = TBD)

If the corpus is the same flat-function shape 06's was, it supports exactly one
task shape and a single-row table is the honest picture, not an omission.
`[TODO: confirm against `spec.md` §2.1's sampling — a second row belongs here
only if the corpus genuinely contains a second shape.]`

| category | task_type | n | example prompt |
|---|---|---:|---|
| conversion | python_to_jac_function | TBD | TBD — paste a real authored instruction from `dataset/sft_train.jsonl` once it exists |

Schema mirrors `04-cpt-sft/dataset/fresh/releases/sft_train.jsonl`'s field set.
Phase-specific values (set in `scripts/pipeline.jac` and `scripts/collect.jac`):

| field | value |
|---|---|
| `id` | `nitinnew_<sha1[:12]>__<file stem>` |
| `category` | `"conversion"` |
| `task_type` | `"python_to_jac_function"` |
| `gate_class` | `"compile_only"` — this corpus ships no pinned expected outputs, so no behavioural assertion is available. Deliberately **not** `"behavioral"`. |
| `test_pass` | `null` — follows from the gate class |
| `origin` | `"<repo>:<commit>:data/jac_outputs/<file>"` |
| `generator` | `"opus-api"` (reverse-instruction authoring) |
| `source_prompt_version` | `"nitin-new-reverse-instruct-v1.0"` |
| `dataset_version` | `"nitin-new-ds-v1.0.0"` |
| `run_tag` | `"nitin-new"` |
| `seed_id` | sha1 of `<repo>:<commit>:<file>`, first 12 hex chars |

Split sizes:

| file | n |
|---|---:|
| `dataset/sft_train.jsonl` | TBD |
| `dataset/sft/train.jsonl` | TBD (85%, seed 42) |
| `dataset/sft/valid.jsonl` | TBD (15% — training-time validation split, NOT a scored eval set) |
| `dataset/nitin_holdout.jsonl` | TBD (frozen holdout, raw rows) |
| `dataset/nitin_holdout_eval.jsonl` | TBD (same rows WITH authored instructions — this is what the harness reads) |

For reference, `04-cpt-sft`'s fresh arm trained on 8,100 SFT rows and 654 DPO
pairs; 06 trained on 5,474 SFT rows. A materially different size here is a
**named confound** on research question 2 (`spec.md` §7.2), not a footnote — 06
hit exactly this and left it unresolved.

## DPO — 1 axis (n = TBD pairs, prompt/chosen/rejected)

| axis | n | example prompt |
|---|---:|---|
| correctness | TBD | TBD — paste a real prompt from `dataset/dpo_train.jsonl` once it exists |

Four axes from `04-cpt-sft/docs/dpo-plan.md` are **deliberately absent**, not
missing: `graph_native` and `idiomatic` had no expressible form in 06's corpus
of flat utility functions; `auth_security` and `typing` had no surface at all.
Recorded as an inherited scope decision (`spec.md` §3.1) —
`[TODO: revisit if `spec.md` §2.1's sampling finds graph-native material in the
new corpus.]`

Sampling parameters actually used (`scripts/prepare_batches.jac`):

| parameter | value |
|---|---|
| `DPO_SAMPLE_N` | 1000 `[TODO: confirm against the new pool size]` |
| `DPO_SAMPLE_SEED` | 42 |
| `DPO_MIN_TOKENS` | 30 `[TODO: recheck against this corpus's median body size]` |
| `DPO_MIN_CODE_LINES` | 5 `[TODO: same]` |
| rows clearing the floor | TBD of TBD |

Per-bug-type breakdown of the `rejected` side:

| injected bug type | n |
|---|---:|
| off-by-one | TBD |
| flipped comparison operator | TBD |
| swapped branch bodies | TBD |
| mutated default argument | TBD |
| inverted accumulator | TBD |
| dropped edge-case guard | TBD |
| **total** | **TBD** |

Dual-gate accounting (`workflow.md` Stage 2):

| check | pass | fail |
|---|---:|---:|
| `chosen` compiles | TBD | TBD |
| `rejected` compiles | TBD | TBD |
| `chosen != rejected` after normalization | TBD | TBD |
| signature survived in `rejected` | TBD | TBD |
| pairs accepted | TBD | — |

`rejected` sides that fail to compile are dropped — a non-compiling rejected
side trains "prefer code that parses" rather than "prefer code that is correct."

Split sizes:

| file | n |
|---|---:|
| `dataset/dpo_train.jsonl` | TBD |
| `dataset/dpo/train.jsonl` | TBD (85%, seed 42) |
| `dataset/dpo/valid.jsonl` | TBD (15% — `mlx_lm_lora`'s training-time validation split) |

**Stated limitation:** with no pinned expected outputs in this corpus, "the
injected bug actually changes behaviour" is not mechanically proven the way
`04-cpt-sft`'s `gen_debug` proved it (buggy variant must fail, fixed variant
must pass). Where the collect script constructed an input and diffed stdout,
record the count here — **TBD of TBD pairs behaviourally confirmed** — and do
not imply coverage beyond it.

**Second limitation, carried from 06:** several source functions in 06's corpus
had code contradicting their own docstring (pre-existing upstream). Fill agents
resolved these inconsistently — some matching documented *intent*, some matching
actual *code* — real label noise in both the training set and the holdout.
`[TODO: check whether the successor corpus fixed this; if not, give fill agents
an explicit tie-break rule up front and record it here.]`

## Eval holdouts

| holdout | rows | code-graded | reused or new |
|---|---:|---:|---|
| `04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl` | 1428 | 855 | reused **unchanged** |
| `dataset/nitin_holdout_eval.jsonl` | TBD | TBD | new, frozen pre-generation, instructions authored in Stage 1b |

Composition of the shared 855, for the per-slice breakdown `spec.md` §7.2
requires:

| slice | n |
|---|---:|
| conversion / behavioral | 322 |
| code_gen / compile_only | 227 |
| trajectory / compile_only | 113 |
| code_gen / behavioral | 86 |
| trajectory / behavioral | 70 |
| debug / behavioral | 33 |
| debug / compile_only | 3 |
| migration / behavioral | 1 |
| **total** | **855** |

The remaining 573 rows of the 1,428 are `prose_lexical` (`explanation` 365,
`documentation` 208) and are not scored by the functional harness.

Note that 322/855 — 38% of the graded set — is `conversion`, which is (if
`spec.md` §2.1 holds) exactly this corpus's own task shape. Any
dataset-comparison win concentrated in that slice is a specialization result,
and the report must say so.

**Expect a ceiling on the new holdout.** 06's equivalent holdout ran 96–99% for
both arms after training — single category, `compile_only` gate, essentially the
same distribution as its own training pool. If that recurs, note it explicitly:
a ceilinged holdout compresses every arm-vs-arm delta and a null there is not
evidence of no effect.

Sources, once they exist: `dataset/sft_train.jsonl`, `dataset/dpo_train.jsonl`,
`dataset/nitin_holdout.jsonl`, `dataset/nitin_holdout_eval.jsonl`,
`docs/reports/corpus-triage-report.md`.
