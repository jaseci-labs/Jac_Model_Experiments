# Task 6: Deduplication, Review, Versioning, and Release

## Purpose

Prepare validated examples for a reproducible dataset release. This task defines deduplication, manual review sampling, versioning, final audit, and release criteria.

## Inputs Needed

- [`context.md`](context.md) for target distribution and quality rules.
- [`task1.md`](task1.md) for metadata, naming, and storage policy.
- [`task3.md`](task3.md) for validation logs and pass-rate thresholds.
- Clean candidates from single-turn generation and trajectory collection.
- Review notes and rejection logs.

## Artifacts To Produce

- Deduplicated clean dataset.
- Manual review sample and review results.
- Dataset version manifest.
- Final audit report.
- Frozen release artifact.

## Step-By-Step Checklist

- [ ] Gather all clean candidates by category.
- [ ] Confirm every candidate has required metadata.
- [ ] Confirm every candidate has a validation log.
- [ ] Run exact deduplication:
  - Remove identical prompts.
  - Remove identical Jac code outputs.
  - Remove identical Python inputs for conversion examples.
  - Remove identical trajectory transcripts.
- [ ] Run near-duplicate review:
  - Flag prompts that differ only by variable names.
  - Flag code that differs only by renamed identifiers.
  - Flag repeated task patterns within the same category.
  - Flag trajectories with the same task solved in nearly the same way.
- [ ] Keep the highest-quality item from each duplicate cluster.
- [ ] Record removed duplicates with their source batch IDs.
- [ ] Build manual review samples:
  - 5-10% from code generation.
  - 5-10% from debugging.
  - 5-10% from explanation.
  - 5-10% from conversion.
  - 5-10% from trajectories.
- [ ] Review sampled examples for correctness, idiomatic Jac, metadata accuracy, and training usefulness.
- [ ] Stop release if any category's manual review pass rate is below 80%.
- [ ] Compare final category counts against target distribution.
- [ ] Confirm hard examples remain approximately 20% of each category.
- [ ] Freeze a dataset version in the format `jac-synth-vMAJOR.MINOR.PATCH`.
- [ ] Create a version manifest with:
  - Dataset version.
  - Generation date range.
  - Category counts.
  - Prompt versions.
  - Context bundle versions.
  - Validator version.
  - Deduplication summary.
  - Manual review summary.
  - Known limitations.
- [ ] Move the frozen dataset to release storage.
- [ ] Record which training runs consume the release version.

## Testing And Validation Checklist

- [ ] Search for missing required metadata fields.
- [ ] Confirm no example has `compiler_pass: false` in the clean dataset.
- [ ] Confirm testable examples have `test_pass: true` or a documented manual-review exception.
- [ ] Confirm all sampled review failures are resolved, removed, or documented.
- [ ] Confirm duplicate clusters were handled intentionally.
- [ ] Confirm release counts match the manifest.
- [ ] Confirm the release artifact is immutable after version freeze.

## Failure Conditions And Retry Guidance

- If deduplication removes too much data from a category, return to Task 4 or Task 5 with stronger diversity requirements.
- If manual review pass rate falls below 80%, stop release and revise prompts, validation, or context before generating replacement examples.
- If metadata is inconsistent, fix the metadata generation process before patching records manually in volume.
- If category distribution is badly imbalanced, generate only the missing category after rechecking its quality gates.
- If hard examples are underrepresented, add targeted hard-example pilots before scaling.
- If release artifacts and manifest counts disagree, stop and reconcile before training uses the dataset.

## Completion Criteria

This task is complete when the dataset is deduplicated, reviewed, versioned, audited, frozen, and ready to be referenced by training runs.
