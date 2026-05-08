# Task 1: Dataset Foundation and Context Setup

## Purpose

Establish the source-of-truth structure for the Jac synthetic dataset before any generation work begins. This task defines where artifacts live, what metadata every example must carry, how raw and clean data are separated, and what Jac context must be available to generation and validation steps.

## Inputs Needed

- [`context.md`](context.md) as the strategy source of truth.
- The current Jac syntax documentation.
- The current `skills.md` or equivalent Jac guidance used by the Jac MCP.
- Access to the Jac compiler or Jac MCP for later validation tasks.
- Agreement on the target dataset range: 10,000-15,000 clean examples.

## Artifacts To Produce

- A dataset directory policy covering raw output, clean examples, rejected examples, review notes, logs, and releases.
- A stable metadata schema for every example.
- A naming convention for example IDs, batch IDs, category names, and dataset versions.
- A Jac context bundle definition for OpenAI prompts and vibe-coding agent sessions with Jac MCP/tooling.

## Step-By-Step Checklist

- [ ] Confirm the dataset categories: `code_gen`, `debug`, `explanation`, `conversion`, and `trajectory`.
- [ ] Define storage areas:
  - `raw_output/` for unvalidated OpenAI API responses and raw trajectory transcripts.
  - `clean_dataset/` for examples that pass required validation gates.
  - `rejected/` for failed generations that may be inspected or recycled.
  - `review/` for manual review samples, notes, and adjudication records.
  - `logs/` for generation, parsing, compiler, test, retry, and deduplication logs.
  - `releases/` for frozen dataset versions used by training runs.
- [ ] Define category-specific subdirectories under each storage area.
- [ ] Define batch IDs in the format `YYYYMMDD-category-seq`, such as `20260507-code_gen-001`.
- [ ] Define example IDs in the format `category-batch-seq`, such as `code_gen-20260507-001-0007`.
- [ ] Define dataset versions in the format `jac-synth-vMAJOR.MINOR.PATCH`.
- [ ] Define required metadata fields:
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
- [ ] Define optional metadata fields:
  - `error_type`
  - `granularity`
  - `trajectory_length_tokens`
  - `dedup_hash`
  - `reviewer`
  - `review_status`
  - `rejection_reason`
- [ ] Define allowed values for `generator`: `openai-api`, `cursor-jac-mcp`, `codex-jac-mcp`, and `claude-code-jac-mcp`.
- [ ] Define allowed values for `complexity`: `simple`, `medium`, and `hard`.
- [ ] Create the Jac context bundle requirements:
  - Jac syntax reference.
  - Jac idioms and pitfalls.
  - `skills.md` or equivalent MCP guidance.
  - Valid examples of walkers, nodes, edges, abilities, imports, type annotations, and standard library usage.
  - Output schema instructions for the target category.
- [ ] Decide how context bundle versions are named and recorded in metadata.
- [ ] Document that OpenAI API prompts must use the same context bundle version as the validation logs record.

## Testing And Validation Checklist

- [ ] Check that every dataset category has a storage location.
- [ ] Check that metadata can represent all five categories without special-case fields becoming required for unrelated categories.
- [ ] Check that raw, clean, rejected, review, log, and release artifacts cannot be confused by naming.
- [ ] Check that example IDs are unique across categories and batches.
- [ ] Check that the context bundle can be referenced by version in every generated example.

## Failure Conditions And Retry Guidance

- If two categories need incompatible required metadata fields, move those fields to optional category-specific metadata.
- If ID formats are too long or hard to scan, revise them before generation starts.
- If the Jac context bundle is too large for planned OpenAI API calls, reduce batch size before removing Jac guidance.
- If source documents disagree about Jac syntax or idioms, stop and update the context bundle before generating examples.

## Completion Criteria

This task is complete when storage policy, metadata schema, naming conventions, and Jac context bundle requirements are documented clearly enough that prompt design and validation planning can proceed without inventing new structure.
