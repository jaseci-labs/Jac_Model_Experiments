# Documentation Index

This documentation covers the Jac synthetic data pipeline for both maintainers and dataset consumers. Start with the README for a quick entry point, then use these pages for the operational details.

## Reading Paths

- Maintainers: [`pipeline.md`](pipeline.md), [`dataset.md`](dataset.md), [`quality.md`](quality.md), [`operations.md`](operations.md), then [`tasks/`](tasks/).
- Dataset consumers: [`stats.md`](stats.md), [`dataset.md`](dataset.md), [`quality.md`](quality.md), and release artifacts under `dataset/releases/` when a version is frozen.
- Operators scaling data: [`operations.md`](operations.md), [`scale_runbook.md`](scale_runbook.md), and [`tasks/task6.md`](tasks/task6.md).

## Pages

- [`context.md`](context.md): strategy, category goals, and high-level generation rationale.
- [`pipeline.md`](pipeline.md): end-to-end generation, validation, review, dedupe, and release flow.
- [`dataset.md`](dataset.md): artifact layout, metadata contract, IDs, versions, and category schemas.
- [`quality.md`](quality.md): compiler gates, pass-rate thresholds, manual review criteria, and duplicate handling.
- [`operations.md`](operations.md): commands for readiness, stats, scale planning, manual review, trajectory ingestion, audit, and freeze.
- [`stats.md`](stats.md): generated dataset snapshot from current repository artifacts.
- [`scale_runbook.md`](scale_runbook.md): detailed scale and release runbook.
- [`tasks/`](tasks/): roadmap and task-by-task build plan for the pipeline.

## Source Of Truth

The docs intentionally point back to code-backed definitions:

- `src/data_generation/foundation.py` defines categories, storage paths, metadata fields, and ID helpers.
- `src/data_generation/prompt_design.py` defines prompt templates, prompt versions, schemas, and prompt guardrails.
- `src/data_generation/validation.py` defines validation stages, compiler field policies, retry limits, and thresholds.
- `src/data_generation/manual_review.py` defines review criteria and review status workflows.
- `src/data_generation/release.py` defines readiness, audit, deduplication, manifest, and freeze behavior.
- `src/data_generation/docs_stats.py` generates the stats page from current artifacts.
