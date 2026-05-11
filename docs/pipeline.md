# Pipeline

The pipeline turns Jac context and task specifications into validated dataset artifacts. It uses scripted OpenAI API generation for single-turn examples and Cursor/Jac MCP sessions for agentic trajectories.

```mermaid
flowchart TD
    ContextBundle[Jac Context Bundle] --> PromptBuilder[Prompt Builder]
    PromptBuilder --> OpenAI[OpenAI API Batches]
    PromptBuilder --> AgentSession[Jac MCP Agent Sessions]
    OpenAI --> RawArtifacts[Raw Output]
    AgentSession --> RawArtifacts
    RawArtifacts --> Validation[Schema And Compiler Validation]
    Validation --> CleanDataset[Clean Dataset]
    Validation --> Rejected[Rejected Or Review]
    CleanDataset --> ManualReview[Manual Review]
    ManualReview --> Deduplication[Exact And Near Deduplication]
    Deduplication --> Audit[Release Audit]
    Audit --> FrozenRelease[Frozen Release]
```

## Generation Modes

Scripted single-turn generation covers `code_gen`, `debug`, `explanation`, and `conversion`. `src/data_generation/single_turn_generation.py` builds prompt requests, calls the generation client, validates returned examples, and writes raw, clean, rejected, review, validation, generation, and scale-decision artifacts.

Agentic trajectory generation covers `trajectory`. `src/data_generation/trajectory_generation.py` plans task banks and ingests normalized transcripts that include user turns, assistant turns, tool calls, tool results, final Jac code, and validation evidence.

## Context And Prompts

The Jac context bundle is stored under `dataset/context/` and summarized in [`context.md`](context.md). Prompt templates and category schemas live in `src/data_generation/prompt_design.py`. Prompt versions are explicit because scale approval depends on knowing which guardrails produced each batch.

## Validation And Routing

Every batch goes through JSON/schema checks and compiler validation. The validation layer routes records to:

- `dataset/clean_dataset/` when required gates pass.
- `dataset/rejected/` when examples fail hard gates or are useful for later inspection.
- `dataset/review/` when manual review is required.
- `dataset/logs/` for validation, generation, retry, prompt revision, scale decision, audit, and dedupe records.

## Release Flow

Release readiness loads clean candidates, audits metadata and validation evidence, removes exact duplicates, flags near duplicates, builds a manual-review sample, checks target counts, and reports blockers. A frozen release writes clean category files, manifest, audit, review sample, dedupe summary, training-run references, and immutable checksums under `dataset/releases/<version>/`.
