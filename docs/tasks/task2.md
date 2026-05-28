# Task 2: OpenAI API Prompt and Schema Design

## Purpose

Design the OpenAI API prompts and strict JSON schemas used for the scripted single-turn generation categories. This task prevents ambiguous outputs, invalid JSON, and low-quality Jac code before any high-volume generation begins.

## Inputs Needed

- [`context.md`](../context.md) for the overall strategy.
- [`task1.md`](task1.md) for metadata, naming, and context bundle requirements.
- The Jac context bundle version selected in Task 1.
- Category targets for code generation, debugging, explanation, and conversion.
- Access to a small set of known-good Jac examples for prompt demonstrations.

## Artifacts To Produce

- One system prompt template for OpenAI API single-turn generation.
- One user prompt template per category.
- One JSON output schema per category.
- A prompt version naming convention.
- Pilot batch settings for each category.
- A prompt revision log policy.
- Python source filtering criteria and type inference methodology for conversion prompts.

## Step-By-Step Checklist

- [ ] Define a shared OpenAI system prompt that states:
  - The model is generating training data for Jac.
  - Jac is its own language and must not be treated as Python.
  - All Jac code will be compiler-validated.
  - Output must be strict JSON with no markdown wrapper.
  - The model must prefer idiomatic Jac constructs.
- [ ] Include the Jac context bundle in the system prompt or prompt assembly process.
- [ ] Add concrete valid Jac examples covering walkers, nodes, edges, abilities, imports, type annotations, and common control flow.
- [ ] Define the code generation user prompt template:
  - Requested count.
  - Complexity distribution.
  - Required constructs.
  - Domains to cover.
  - JSON schema requiring `prompt`, `code`, and `complexity`.
- [ ] Define the debugging user prompt template:
  - Requested count.
  - One error type per example.
  - Requirement that `broken_code` fails and `fixed_code` compiles.
  - JSON schema requiring `broken_code`, `error_type`, `error_message`, `fixed_code`, and `fix_explanation`.
- [ ] Define the explanation user prompt template:
  - Requested count.
  - Granularity distribution: line, block, and module.
  - Requirement that source code is valid Jac.
  - JSON schema requiring `code`, `granularity`, and `explanation`.
- [ ] Define the conversion user prompt template:
  - Requested count.
  - Python source style and complexity constraints.
  - Requirement that Jac output is idiomatic, not a direct syntax translation.
  - JSON schema requiring `python_code`, `jac_code`, and `conversion_notes`.
- [ ] Add type inference instructions to the conversion user prompt:
  - Infer Python argument and return types from test execution (runtime observation) or Pyright static analysis.
  - Inject inferred types into the Jac translation prompt so the LLM produces correctly typed Jac code.
  - Include type annotations in the prompt format: provide the Python function signature with inferred types alongside the code and docstring.
- [ ] Define Python source filtering criteria for conversion and Python-sourced code_gen:
  - Python function must have a docstring.
  - Python function must pass Pyright type-checking and return a value.
  - Python function must not contain TODO, FIXME, or incomplete markers.
  - Python function must not overlap with known Code LLM benchmarks (HumanEval, MBPP).
  - Python function must have LLM-generated unit tests with at least 90% line coverage.
- [ ] Add a docstring-quality filter to Python source curation: drop functions with misleading docstrings via a binary docstring-quality LLM classifier (SelfCodeAlign, 2410.24198). The return-value requirement above already guarantees tests have meaningful assertions.
- [ ] Define an interleaved generation mode for directly-generated code categories: the model emits the Jac solution and its Jac tests in one completion, filtered on execution (SelfCodeAlign).
- [ ] Define a snippet-seeded prompt variant: seed generation with 1--15 random Python lines, abstract to concepts, then request an unrelated self-contained Jac task (OSS-Instruct / Magicoder, 2312.02120).
- [ ] Record `seed_source` (`grammar_matrix`, `python_translation`, `oss_instruct`, `zero_seed`, `persona`, `doc`, `evol`) and `token_count` on every generated example.
- [ ] Define multi-candidate translation settings for conversion:
  - Generate 50--100 candidate Jac translations per Python source function.
  - Use high temperature (0.8) to encourage diverse translations.
  - Keep all candidates that pass cross-compiled tests.
  - Deduplicate within candidates using ROUGE-L (threshold 0.6) before adding to clean dataset.
- [ ] Define pilot batch sizes:
  - 5 examples for code generation.
  - 5 examples for debugging.
  - 5 examples for explanation.
  - 5 examples for conversion.
- [ ] Define scale-up batch sizes after pilots pass:
  - 20-50 examples per OpenAI API call.
  - Smaller batches if context size or output length causes truncation.
- [ ] Define JSON parser expectations:
  - Top-level value is an array.
  - Every item matches the category schema.
  - No extra prose outside JSON.
  - Required fields are non-empty strings except controlled enum fields.
- [ ] Define prompt version IDs in the format `prompt-category-vN`.
- [ ] Define a revision log entry format:
  - Prompt version.
  - Changed fields.
  - Reason for change.
  - Batch IDs affected.
  - Observed pass-rate change.

## Testing And Validation Checklist

- [ ] Run each prompt mentally against the schema and confirm it asks for exactly the required fields.
- [ ] Confirm every schema can be parsed without regex or markdown stripping.
- [ ] Confirm every prompt tells the OpenAI model that compiler validation is mandatory.
- [ ] Confirm category prompts ask for diversity in constructs, domains, and complexity.
- [ ] Confirm pilot batch sizes are small enough to inspect manually.

## Failure Conditions And Retry Guidance

- If OpenAI responses include markdown fences or explanations outside JSON, strengthen the system prompt and reject that batch.
- If required fields are missing, revise the category schema instructions and retry a tiny batch.
- If generated Jac resembles Python syntax, add more Jac-specific examples and reduce batch size.
- If output truncates, reduce requested count before removing Jac context.
- If examples cluster around the same pattern, add explicit diversity constraints for constructs and problem domains.
- If type inference produces incorrect types (Jac compilation fails on type annotations), fall back to untyped Jac translation and note `type_inference_method: none` in metadata.
- If cross-compiled tests reject more than 70% of candidate translations for a source function, flag the source function as unsuitable for translation and skip it.

## Completion Criteria

This task is complete when every scripted category has a prompt template, JSON schema, pilot settings, and prompt revision policy ready for validation planning.
