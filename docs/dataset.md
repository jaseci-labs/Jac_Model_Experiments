# Dataset Layout

The dataset is organized by artifact lifecycle first, then by category. Definitions are backed by `DATASET_STORAGE_PATHS` in `src/data_generation/foundation.py`.

## Categories

- `code_gen`: natural language prompt to compilable Jac code.
- `debug`: broken Jac code plus error information to fixed compilable Jac code.
- `explanation`: compilable Jac code to natural language explanation.
- `conversion`: Python source to idiomatic compilable Jac code.
- `trajectory`: agentic Jac task-solving transcripts with Jac MCP/tooling evidence.

## Storage Areas

- `dataset/raw_output/<category>/`: unvalidated OpenAI responses and raw trajectory inputs.
- `dataset/clean_dataset/<category>/`: validation-passing candidate examples, one JSON object per line.
- `dataset/rejected/<category>/`: rejected or discarded examples retained for inspection or recycling.
- `dataset/review/<category>/`: manual review files and adjudication records.
- `dataset/logs/generation/`: generation metadata, model information, pass rates, and prompt hashes.
- `dataset/logs/validation/`: per-example validation records.
- `dataset/logs/prompt_revisions/`: prompt change records and observed quality effects.
- `dataset/logs/scale_decisions/`: per-batch scale/revise/pause decisions.
- `dataset/logs/deduplication/`: exact and near-duplicate reports and resolutions.
- `dataset/logs/audit/`: readiness and release audit reports.
- `dataset/context/`: Jac context bundles used by prompts and validators.
- `dataset/releases/`: frozen release artifacts.

## Naming

- Batch IDs use `YYYYMMDD-category-seq`, for example `20260511-code_gen-006`.
- Example IDs use `category-YYYYMMDD-seq-example`, for example `code_gen-20260511-006-0001`.
- Dataset versions use `jac-synth-vMAJOR.MINOR.PATCH`.
- Prompt versions use `prompt-category-vN` for scripted categories and `trajectory-prompt-vN` for trajectory collection.

## Required Metadata

Every clean candidate must carry:

- `id`
- `batch_id`
- `category`
- `complexity`
- `compiler_pass`
- `test_pass`
- `manually_reviewed`
- `generator`
- `generation_date`
- `source_prompt_version`
- `context_bundle_version`
- `validator_version`
- `dataset_version`

Optional category-specific fields include `error_type`, `granularity`, `trajectory_length_tokens`, `dedup_hash`, `reviewer`, `review_status`, and `rejection_reason`.

## Release Artifacts

A frozen release under `dataset/releases/<version>/` contains deduplicated category JSONL files, `manifest.json`, `audit.json`, `manual_review_sample.json`, `deduplication_summary.json`, `training_runs.json`, and `IMMUTABLE_RELEASE.json` with checksums.
