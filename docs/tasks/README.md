# Data Generation Task Roadmap

This document is the master execution roadmap for building a synthetic Jac data generation workflow. The goal is to produce 10,000-15,000 clean training examples without rushing volume ahead of validation.

The work should proceed slowly: design a small sample, generate a tiny batch, validate it, manually inspect it, revise prompts or schemas, and only then scale. A category must not scale to thousands of examples until its pilot batches pass compiler validation, manual review, and deduplication checks.

---

## Operating Principles

- Quality is more important than volume. A smaller clean dataset is better than a larger noisy one.
- Every generated Jac code field must pass compiler validation before entering the clean dataset.
- For deterministic code_gen and conversion examples, cross-compiled test validation is a hard gate alongside compilation. Tests are generated in Python and compiled to Jac deterministically, following the MultiPL-T methodology.
- OpenAI API generation is used for single-turn examples: code generation, debugging, explanation, and conversion.
- Vibe-coding agent sessions with Jac MCP/tooling are used for agentic trajectories because the transcript is the artifact. Suitable environments include Cursor, Codex, and Claude Code.
- Raw output, rejected examples, and clean examples must be stored separately.
- Prompts, schemas, validation logs, and dataset versions must be reproducible.
- Scaling happens only after pilot batches show stable quality.
- Generation seeds from multiple sources (grammar matrix, Python translation, OSS-Instruct snippets, doc-grounded, personas, Evol-Instruct, and zero-seed extraction once a v0 model exists); track `seed_source` per example.
- Untrusted synthetic tests are weighted by mutual code↔test credibility (CodeDPO); cross-compiled tests remain the strongest gate.
- Track tokens per example and in aggregate per batch/run for context-window fit and cost.

---

## Phase Order

- [ ] [`task1.md`](task1.md): Dataset foundation and source-of-truth setup.
- [ ] [`task2.md`](task2.md): OpenAI API prompt and schema design.
- [ ] [`task3.md`](task3.md): Compiler validation, test harness, rejection handling, and retry loops.
- [ ] [`task4.md`](task4.md): Single-turn category generation loops.
- [ ] [`task5.md`](task5.md): Agentic trajectory generation with vibe-coding agents and Jac MCP/tooling.
- [ ] [`task6.md`](task6.md): Deduplication, manual review, versioning, and release readiness.
- [ ] [`task7.md`](task7.md): Advanced generation and preference recipes (snippet-seeded, zero-seed, credibility-ranked DPO, runtime-efficiency pairs, FIM data, repo-level projects). [Planned]

Each phase depends on the previous phase. Do not begin high-volume generation before the setup, prompt design, validation, and pilot review phases are complete.

---

## Recommended Cadence

Use the same cadence for every category:

1. Define the exact category schema.
2. Build a tiny prompt sample with 3-5 examples.
3. Generate a pilot batch of 5-10 examples through OpenAI's API or one vibe-coding agent trajectory session with Jac MCP/tooling.
4. Validate all Jac code fields with the compiler.
5. For conversion and Python-sourced code_gen, validate with cross-compiled tests (hard gate).
6. Run behavior tests where behavior can be specified.
7. Manually inspect the whole pilot batch.
8. Record failure modes and revise the prompt, schema, or validation rules.
9. Repeat until pilot quality is stable.
10. Increase to 20-50 examples per batch.
11. Run deduplication after each batch.
12. Scale only after the category passes the definition of done.

---

## Definitions of Done

### Task 1: Foundation

- Dataset storage policy is documented.
- Metadata schema is stable.
- Raw, clean, rejected, review, and release artifact locations are defined.
- Jac context bundle requirements are clear.
- Naming conventions are written.

### Task 2: Prompt and Schema Design

- Each category has an OpenAI prompt template.
- Each category has a strict JSON output schema.
- Prompt revision rules are documented.
- Pilot batch settings are defined.
- JSON parsing and schema validation expectations are written.
- Type inference, Python source filtering, snippet-seeded, and interleaved generation modes are documented.
- `seed_source` and `token_count` are recorded on every generated example.

### Task 3: Validation

- Compiler validation rules are documented for each code-bearing field.
- Test harness expectations are documented for behavior-checkable examples.
- Rejected-example recycling rules are documented.
- Retry limits and failure thresholds are defined.
- Logs needed for debugging generation quality are specified.
- Cross-compiled test infrastructure is built and validated for deterministic categories.
- Python source filtering criteria are documented and the source pool is populated.

### Task 4: Single-Turn Generation

- Code generation, debugging, explanation, and conversion loops are documented separately.
- Pilot batch requirements exist for every category.
- Scaling thresholds are explicit.
- Per-category failure modes and retry steps are documented.

### Task 5: Agentic Trajectories

- Vibe-coding agent session setup is documented for Cursor, Codex, Claude Code, or another approved environment with Jac MCP/tooling.
- Transcript format is defined.
- Keep/discard criteria are explicit.
- Review rules and length limits are documented.
- Recovery trajectory preference is documented.

### Task 6: Release Readiness

- Deduplication rules are documented (including repo-level dedup for multi-file projects).
- Manual review sampling process is documented.
- Dataset versioning rules are documented.
- Final audit checklist is complete.
- Release criteria are clear.
- DPO pairs are selected by credibility, filtered for ambiguity, and include a runtime-efficiency axis.
- Every clean example records `token_count`; aggregate token usage per batch is logged.

### Task 7: Advanced Generation and Preference Recipes

- Snippet-seeded and zero-seed generation modes are documented with pilot results.
- Credibility-ranked DPO and runtime-efficiency pairs are documented.
- FIM data format and repo-level synthetic projects are documented.
- Evolution control (merge-all-rounds, Evol-Stop) is documented.
- A decision is recorded on which advanced recipes graduate into volume generation.

---

## Progress Gates

- [ ] Foundation documents reviewed before prompt work begins.
- [ ] Prompt templates reviewed before any scripted generation begins.
- [ ] Validation rules reviewed before any generated examples are accepted.
- [ ] Cross-compiled test compiler validated on a pilot batch before scaling conversion or Python-sourced code_gen.
- [ ] Filtered Python source pool populated with at least 1,000 functions before scaling Recipe 2.
- [ ] At least one pilot batch per single-turn category passes before scaling.
- [ ] At least one pilot trajectory passes before collecting trajectories in volume.
- [ ] Manual review pass rate remains at or above 80% before continuing volume generation.
- [ ] Deduplication is run after every batch and again before release.
- [ ] Final dataset version is frozen before training consumes it.
- [ ] Credibility scoring validated before building DPO pairs from generated (non-cross-compiled) tests.
- [ ] Token accounting active from the first batch.

---

## Scaling Warning

Do not optimize for overnight volume until the small loops are boring and reliable. If compile pass rate drops, JSON parsing fails, duplicates increase, explanations become vague, or manual review pass rate falls below 80%, stop scaling and return to prompt and validation iteration.
