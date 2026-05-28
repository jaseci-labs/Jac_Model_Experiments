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

## Credibility Scoring For Untrusted Tests

Synthetic tests are not trustworthy on their own. Following CodeDPO (2410.05605), when tests are generated rather than cross-compiled, weight them by mutual code↔test credibility. For each task, generate multiple candidate Jac solutions (15 at temperature 1.5) and multiple candidate tests, build a bipartite pass/fail graph, and iterate two coupled PageRank-style scores (damping 0.85, ~10 iterations): a solution gains credibility from passing tests that credible solutions pass; a test gains credibility from being passed by credible solutions. Use credibility scores to select trusted tests and rank solutions. This correlated with true accuracy at Spearman 0.86 versus 0.61 for naive "passed all tests." Cross-compiled tests (deterministic, from validated Python) remain the strongest gate; credibility scoring covers the categories where only generated tests are available.

Gate outcomes (compiler and test 0–1 labels) should also be retained as training data for a reward model (DeepSeek-Coder-V2, 2406.11931), which gives a denser, less noisy signal than the binary gate for sparse-coverage tasks.

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
- Credibility-score correlation target: prefer the credibility ranking over naive test-count ranking when generated tests are untrusted.
- DPO preference pairs: drop pairs whose winner/loser credibility scores are near-identical (ambiguous).
- Cosine-to-holdout diagnostic: prefer generation methods with lower mean similarity to the eval holdout while still passing gates (Magicoder).

## Manual Review

Manual review criteria live in `src/data_generation/manual_review.py`. Each category has explicit checks, such as idiomatic Jac for code generation, exactly one realistic error for debugging, accurate Jac semantics for explanation, behavior preservation for conversion, and logical MCP tool use for trajectories.

Review records use `pending`, `passed`, `failed`, or `waived`. Failed and waived reviews require notes.

Validate the idiom judge with a cross-family model (Magpie, 2406.08464): periodically re-score a sample of idiom-judge outputs with an out-of-model-family judge (for example Qwen versus Claude) to confirm the judge is not self-favoring.

## Deduplication

Exact duplicate removal compares category-specific content keys such as prompts, code, broken/fixed code pairs, conversion pairs, and trajectory turns. Near-duplicate detection flags high normalized content similarity and requires a recorded decision: `keep_distinct`, `waived`, or `remove_duplicate`.

For conversion examples with multiple candidate translations per source function (50--100 candidates), deduplicate within the candidate set first using ROUGE-L (threshold 0.6), then across the full dataset.

Use min-neighbor-distance as a quality filter, not only exact/near-dedup (Magpie). Embed examples and use FAISS nearest-neighbor distance combined with quality, difficulty, and length thresholds — keep the most isolated examples rather than merely dropping duplicates. No single filter configuration wins across benchmarks; produce several differently-filtered subsets and blend them. Retain a small controlled fraction of compiler-valid but partially-implemented samples (stubs, `pass`) — Magicoder (2312.02120) found this beat the fully-cleaned set. For multi-file synthetic Jac projects, run repo-level MinHash near-dedup over the concatenated project rather than per file (DeepSeek-Coder, 2401.14196).

## Release Audit

Readiness can be blocked by missing task artifacts, invalid metadata, missing validation logs, incomplete manual review, insufficient volume, category imbalance, hard-example imbalance, or unresolved near duplicates. Use [`operations.md`](operations.md) for commands and [`stats.md`](stats.md) for the current generated snapshot.
