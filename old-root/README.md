# Jac Data Generation

Synthetic data generation pipeline for training Jac coding models. The project builds, validates, reviews, deduplicates, and releases Jac examples across code generation, debugging, explanation, conversion, and agentic trajectory categories.

## Quickstart

```bash
python -m pytest
python -m data_generation.release readiness --version jac-synth-v0.1.0
python -m data_generation.docs_stats --version jac-synth-v0.1.0 --output docs/stats.md
```

Set `OPENAI_API_KEY` or `OPEN_AI_API_KEY` before live OpenAI generation.

## Current Snapshot

The generated dataset snapshot lives in [`docs/stats.md`](docs/stats.md). It summarizes clean counts, rejected counts, review files, readiness blockers, prompt versions, and duplicate-review status from repository artifacts.

## Documentation

- [`docs/index.md`](docs/index.md): documentation map and reading paths.
- [`docs/pipeline.md`](docs/pipeline.md): how generation flows from context bundle to release.
- [`docs/dataset.md`](docs/dataset.md): storage layout, metadata, naming, and categories.
- [`docs/quality.md`](docs/quality.md): validation, review, dedupe, and release gates.
- [`docs/operations.md`](docs/operations.md): command reference for generation, review, stats, and release.
- [`docs/tasks/`](docs/tasks/): original task roadmap and phase checklists.
