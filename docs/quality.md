# Quality Gates

Quality is enforced before examples become release candidates. The main gates are schema validation, compiler validation, manual review, deduplication, and release audit checks.

## Validation

`src/data_generation/validation.py` defines the validation stages and compiler field policies:

- `code_gen`: `code` must compile.
- `debug`: `broken_code` must fail as intended and `fixed_code` must compile.
- `explanation`: `code` must compile.
- `conversion`: `jac_code` must compile.
- `trajectory`: `final_output.code` must compile.

Schema validation runs before compiler checks for scripted categories. Behavior tests are recorded where useful, but failed behavior tests route examples to manual review instead of silently accepting or deleting them.

### Cross-Compiled Test Validation

For `code_gen` and `conversion` categories with deterministic behavior, cross-compiled test validation is a hard gate. Tests are generated in Python (where LLMs produce reliable tests), verified for correctness and 90% line coverage against the Python source, then compiled to Jac using a deterministic rule-based test compiler. The compiled tests validate the Jac output without any LLM involvement in the test layer.

- `code_gen`: when the task has deterministic expected behavior, cross-compiled tests must pass. Failure rejects the example.
- `conversion`: cross-compiled tests from the Python source must pass against the Jac translation. Failure rejects the translation.
- `debug`: `fixed_code` must pass cross-compiled tests when the original working code had them.
- `explanation` and `trajectory`: no cross-compiled tests. Manual review remains the quality gate.

Test validation is a hard gate for deterministic categories because compilation alone is insufficient — code that compiles can still produce incorrect results. At scale (300k+ examples), routing test failures to manual review is not feasible.

## Thresholds

The core thresholds are:

- Pilot compiler pass target: 80%.
- Scaled-batch compiler warning threshold: 70%.
- Manual review pass minimum: 80%.
- JSON parse pass target before scaling: 100%.
- Target hard-example ratio: 20% per category during release readiness.
- Cross-compiled test pass target for code_gen: 70% of compilable examples with deterministic behavior.
- Cross-compiled test pass target for conversion: 80% of compilable translations.
- Python source test coverage minimum: 90% line coverage before translation.

## Manual Review

Manual review criteria live in `src/data_generation/manual_review.py`. Each category has explicit checks, such as idiomatic Jac for code generation, exactly one realistic error for debugging, accurate Jac semantics for explanation, behavior preservation for conversion, and logical MCP tool use for trajectories.

Review records use `pending`, `passed`, `failed`, or `waived`. Failed and waived reviews require notes.

## Deduplication

Exact duplicate removal compares category-specific content keys such as prompts, code, broken/fixed code pairs, conversion pairs, and trajectory turns. Near-duplicate detection flags high normalized content similarity and requires a recorded decision: `keep_distinct`, `waived`, or `remove_duplicate`.

For conversion examples with multiple candidate translations per source function (50--100 candidates), deduplicate within the candidate set first using ROUGE-L (threshold 0.6), then across the full dataset.

## Release Audit

Readiness can be blocked by missing task artifacts, invalid metadata, missing validation logs, incomplete manual review, insufficient volume, category imbalance, hard-example imbalance, or unresolved near duplicates. Use [`operations.md`](operations.md) for commands and [`stats.md`](stats.md) for the current generated snapshot.
