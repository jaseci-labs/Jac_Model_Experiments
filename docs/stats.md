# Dataset Stats

Generated from repository artifacts at `2026-05-11T21:39:45Z`.

- Dataset version: `jac-synth-v0.1.0`
- Release status: `blocked`
- Clean examples: `187`
- Rejected examples: `110`
- Raw output files: `38`
- Review files: `33`
- Log files: `139`

For command details, see [release readiness](operations.md#release-readiness) and the [scale runbook](scale_runbook.md).

## Category Progress

| Category | Clean | Rejected | Raw files | Review files | Target range |
| --- | ---: | ---: | ---: | ---: | --- |
| `code_gen` | 82 | 34 | 12 | 11 | 3,000-5,000 |
| `debug` | 39 | 31 | 9 | 8 | 2,000-3,000 |
| `explanation` | 31 | 10 | 7 | 6 | 1,000-2,000 |
| `conversion` | 32 | 35 | 9 | 7 | 1,000-2,000 |
| `trajectory` | 3 | 0 | 1 | 1 | 2,000-3,000 |

## Readiness

- Current total: `185`
- Target total range: `10,000-15,000`
- Manual review status: `blocked_manual_review_pending`
- Near duplicates unresolved: `6`
- Near duplicates resolved: `29`
- Candidate audit status: `warning`
- Candidate audit failures: `0`
- Candidate audit warnings: `150`

## Blockers

- Manual review sample is not fully passed.
- Clean example count is below the 10,000 example release minimum.
- Near-duplicate clusters require manual review.

## Prompt And Context Versions

- Prompt versions: `prompt-code_gen-v2`, `prompt-code_gen-v3`, `prompt-code_gen-v4`, `prompt-conversion-v1`, `prompt-conversion-v2`, `prompt-conversion-v3`, `prompt-conversion-v4`, `prompt-debug-v1`, `prompt-debug-v2`, `prompt-debug-v3`, `prompt-debug-v4`, `prompt-explanation-v1`, `prompt-explanation-v2`, `prompt-explanation-v3`, `trajectory-prompt-v1`
- Context bundle versions are recorded per example; see `dataset/context/` and `docs/context.md`.

## Interpretation Notes

- Clean counts include records currently present in `dataset/clean_dataset/` before any final release freeze.
- Rejected counts include records kept for inspection or possible recycling; they are not training-ready examples.
- Readiness status comes from the release audit logic and can be blocked by volume, manual review, validation, or unresolved near duplicates.
- This file is generated; update it with `python -m data_generation.docs_stats --version jac-synth-v0.1.0 --output docs/stats.md`.
