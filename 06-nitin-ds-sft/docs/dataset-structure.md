# Dataset structure — 06-nitin-ds-sft

> **SCAFFOLD. Every count below is TBD.** The concurrent triage agent has not
> landed `dataset/candidate_pool.jsonl`, `dataset/nitin_holdout.jsonl`, or
> `docs/reports/corpus-triage-report.md` yet. Fill these tables from those
> artifacts — do not estimate, do not carry a number forward from
> `04-cpt-sft`, and do not round a drop count up to a tidy total.

Source: `chess10kp/jac-data-gen` @ `7c25aff3110f526eec59e0123ffe6c0c152cce91`,
7,627 files under `data/jac_outputs/*.jac`, one transpiled Python function per
file. Both arms train on MD5-verified identical copies of the releases below.
Both arms eval on two holdouts: `04-cpt-sft`'s unchanged 1,428-row set (855
code-graded), and this phase's own `nitin_holdout.jsonl`.

## Triage yield

Every row is a real count with an attributed reason. A pool that shrinks a
lot is a finding to report, not a number to smooth over.

| stage | n | Δ | note |
|---|---:|---:|---|
| files at commit pin | 7627 | — | verified by `ls data/jac_outputs/*.jac \| wc -l` |
| compiles (`jac run` exit 0) | TBD | TBD | drops → `dataset/rejected/compile/` |
| after exact-hash dedup | TBD | TBD | |
| after 14-token-shingle near-dup dedup (≥0.5) | TBD | TBD | |
| after decontam vs `04-cpt-sft` sft/valid.jsonl | TBD | TBD | |
| after decontam vs `04-cpt-sft` dpo/valid.jsonl | TBD | TBD | |
| after decontam vs `01-sft-dpo` eval_holdout/conversion.jsonl | TBD | TBD | |
| after decontam vs `01-sft-dpo` eval_holdout/graph_conversion.jsonl | TBD | TBD | |
| **clean pool** (`candidate_pool.jsonl`) | **TBD** | | |
| — carved to `nitin_holdout.jsonl` | TBD | | frozen before any generation |
| — remaining training pool | TBD | | |

Cross-holdout independence check (`spec.md` §6): near-dups between
`nitin_holdout.jsonl` and `04-cpt-sft`'s holdouts — **TBD** (must be 0, or
the two holdout columns are not independent tests).

## SFT — 1 category, 1 task type (n = TBD)

The corpus supports exactly one task shape (`spec.md` §2.1). A single-row
composition table is the honest picture, not an omission.

| category | task_type | n | example prompt |
|---|---|---:|---|
| conversion | python_to_jac_function | TBD | TBD — paste a real authored instruction from `dataset/sft_train.jsonl` once it exists |

Schema mirrors `04-cpt-sft/dataset/fresh/releases/sft_train.jsonl`'s field
set. Phase-specific values:

| field | value |
|---|---|
| `category` | `"conversion"` |
| `task_type` | `"python_to_jac_function"` |
| `gate_class` | `"compile_only"` — this corpus ships no pinned expected outputs, so no behavioral assertion is available. Deliberately **not** `"behavioral"`. |
| `test_pass` | `null` — follows from the gate class |
| `origin` | `"jac-data-gen:7c25aff3110f526eec59e0123ffe6c0c152cce91:<path>"` |
| `generator` | `"opus-api"` (reverse-instruction authoring) |
| `run_tag` | `"nitin"` |
| `seed_id` | the corpus file's stable id |

Split sizes:

| file | n |
|---|---:|
| `dataset/sft_train.jsonl` | TBD |
| `<arm>_probe/dataset/sft/train.jsonl` | TBD |
| `<arm>_probe/dataset/sft/valid.jsonl` | TBD (training-time validation split) |
| `dataset/nitin_holdout.jsonl` | TBD (eval holdout — never trained on) |

For reference, `04-cpt-sft`'s fresh arm trained on 8,100 SFT rows and 654 DPO
pairs. A materially different size here is a named confound on research
question 2 (`spec.md` §7.2), not a footnote.

## DPO — 1 axis (n = TBD pairs, chosen/rejected format)

| axis | n | example prompt |
|---|---:|---|
| correctness | TBD | TBD — paste a real prompt from `dataset/dpo_train.jsonl` once it exists |

Four axes from `04-cpt-sft/docs/dpo-plan.md` are **deliberately absent**, not
missing: `graph_native` and `idiomatic` have no expressible form in a corpus
of flat utility functions with no graph-native rewrite; `auth_security` and
`typing` have no surface here at all. Recorded as a scope decision
(`spec.md` §3.1).

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
| pairs accepted | TBD | — |

`rejected` sides that fail to compile are dropped — a non-compiling rejected
side trains "prefer code that parses" rather than "prefer code that is
correct."

**Stated limitation:** with no pinned expected outputs in this corpus, "the
injected bug actually changes behavior" is not mechanically proven the way
`04-cpt-sft`'s `gen_debug` proved it (buggy variant must fail, fixed variant
must pass). Where the collect script constructed an input and diffed stdout,
record the count here — **TBD of TBD pairs behaviorally confirmed** — and do
not imply coverage beyond it.

## Eval holdouts

| holdout | rows | code-graded | reused or new |
|---|---:|---:|---|
| `04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl` | 1428 | 855 | reused **unchanged** |
| `dataset/nitin_holdout.jsonl` | TBD | TBD | new, frozen pre-generation |

Composition of the shared 855, for the per-slice breakdown
`spec.md` §7.2 requires:

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

The remaining 573 rows of the 1,428 are `prose_lexical`
(`explanation` 365, `documentation` 208) and are not scored by the functional
harness.

Note that 322/855 — 38% of the graded set — is `conversion`, which is exactly
this corpus's own task shape. Any dataset-comparison win concentrated in that
slice is a specialization result, and the report must say so.

Sources, once they exist: `dataset/sft_train.jsonl`,
`dataset/dpo_train.jsonl`, `dataset/nitin_holdout.jsonl`,
`docs/reports/corpus-triage-report.md`.
