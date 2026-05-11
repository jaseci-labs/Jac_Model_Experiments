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

## Thresholds

The core thresholds are:

- Pilot compiler pass target: 80%.
- Scaled-batch compiler warning threshold: 70%.
- Manual review pass minimum: 80%.
- JSON parse pass target before scaling: 100%.
- Target hard-example ratio: 20% per category during release readiness.

## Manual Review

Manual review criteria live in `src/data_generation/manual_review.py`. Each category has explicit checks, such as idiomatic Jac for code generation, exactly one realistic error for debugging, accurate Jac semantics for explanation, behavior preservation for conversion, and logical MCP tool use for trajectories.

Review records use `pending`, `passed`, `failed`, or `waived`. Failed and waived reviews require notes.

## Deduplication

Exact duplicate removal compares category-specific content keys such as prompts, code, broken/fixed code pairs, conversion pairs, and trajectory turns. Near-duplicate detection flags high normalized content similarity and requires a recorded decision: `keep_distinct`, `waived`, or `remove_duplicate`.

## Release Audit

Readiness can be blocked by missing task artifacts, invalid metadata, missing validation logs, incomplete manual review, insufficient volume, category imbalance, hard-example imbalance, or unresolved near duplicates. Use [`operations.md`](operations.md) for commands and [`stats.md`](stats.md) for the current generated snapshot.
