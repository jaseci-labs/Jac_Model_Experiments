# Task 4: Single-Turn Category Generation Loops

## Purpose

Define the slow, repeatable generation loops for the four scripted OpenAI API categories: code generation, debugging, explanation, and code conversion. Each category must pass pilot validation before scaling.

## Inputs Needed

- [`context.md`](../context.md) for category targets and quality rules.
- [`task1.md`](task1.md) for storage, metadata, and context bundle requirements.
- [`task2.md`](task2.md) for prompt templates and schemas.
- [`task3.md`](task3.md) for validation, retries, and pass-rate thresholds.
- Access to OpenAI's API and the Jac compiler validation process.

## Artifacts To Produce

- Pilot batches for each single-turn category.
- Category-specific prompt revisions based on pilot failures.
- Clean, rejected, review, and log records for every pilot batch.
- A scale-up decision for each category.

## Step-By-Step Checklist

### Code Generation

- [ ] Generate a pilot batch of 5 examples.
- [ ] Request a mix of simple, medium, and hard tasks.
- [ ] Require coverage of at least two Jac constructs in the pilot batch.
- [ ] Validate that each item has `prompt`, `code`, and `complexity`.
- [ ] Compile every `code` field.
- [ ] Run behavior tests when prompts include expected observable outputs.
- [ ] For code_gen examples sourced from Python (translated tasks), use cross-compiled tests as a hard gate: generate tests in Python, verify coverage, compile to Jac deterministically, reject examples that fail.
- [ ] Add a snippet-seeded generation variant: seed with 1--15 random Python lines, abstract to concepts, request an unrelated self-contained Jac task (OSS-Instruct / Magicoder, 2312.02120) to inject real-world domain diversity.
- [ ] For testable code generation, emit the Jac solution and its tests in one completion and filter on execution (SelfCodeAlign, 2410.24198).
- [ ] Manually inspect all pilot examples for prompt clarity and idiomatic Jac.
- [ ] Revise the prompt if examples are vague, trivial, duplicated, or Python-like.
- [ ] Scale only after the pilot has clean JSON, strong compiler results, and acceptable review quality.

### Debugging

- [ ] Seed debugging examples from compiler-verified valid Jac code where possible.
- [ ] Generate a pilot batch of 5 examples with one error type per example.
- [ ] Validate that each item has `broken_code`, `error_type`, `error_message`, `fixed_code`, and `fix_explanation`.
- [ ] Confirm `broken_code` fails with the intended compiler or runtime error.
- [ ] Confirm `fixed_code` compiles.
- [ ] Manually inspect whether the injected error is realistic.
- [ ] Reject pairs where `broken_code` compiles or `fixed_code` fails.
- [ ] Revise the prompt if explanations are generic or the injected errors are artificial.
- [ ] Scale only after error types are realistic and fixes are precise.

### Explanation

- [ ] Generate a pilot batch of 5 examples.
- [ ] Include line-level, block-level, and module-level explanation examples.
- [ ] Validate that each item has `code`, `granularity`, and `explanation`.
- [ ] Compile every `code` field.
- [ ] Manually inspect every pilot explanation for accuracy.
- [ ] Reject explanations that overstate behavior, omit important Jac semantics, or describe Python behavior instead of Jac behavior.
- [ ] Revise the prompt if explanations become generic or fail to mention key constructs.
- [ ] Scale only after manual review confirms the explanations are accurate and specific.

### Code Conversion

- [ ] Keep only Python sources that return a value (meaningful assertions) and drop misleading-docstring sources via a docstring-quality classifier (SelfCodeAlign).
- [ ] Build a filtered Python source pool before generating any conversions:
  - Filter Python functions to require: docstring present, Pyright type-check passing, returns a value, no TODO/FIXME/incomplete markers, no overlap with HumanEval/MBPP benchmarks.
  - Generate unit tests for each Python function using an LLM. Verify tests pass. Require at least 90% line coverage.
  - Infer Python argument and return types from test execution (runtime observation) or Pyright static analysis.
  - Target: 10,000+ filtered Python functions as the translation source pool.
- [ ] Build or adapt a deterministic Python-to-Jac test compiler:
  - Translate Python assertions (`assert f(x) == y`) to Jac assertions.
  - Handle first-order values: ints, strings, booleans, lists, tuples, dicts.
  - No LLM involvement in test compilation.
  - Drop test cases that use Python features without Jac equivalents.
- [ ] Generate a pilot batch of 5 Python-to-Jac conversions using cross-compiled test validation:
  - For each Python source, generate 50--100 candidate translations at high temperature (0.8).
  - Compile each candidate.
  - Run cross-compiled tests against each compilable candidate.
  - Keep all candidates that pass. Deduplicate within candidates using ROUGE-L (threshold 0.6).
- [ ] Include at least one function conversion and one class or data-structure conversion.
- [ ] Validate that each item has `python_code`, `jac_code`, and `conversion_notes`.
- [ ] Inject inferred Python types into the translation prompt for correctly typed Jac output.
- [ ] Manually inspect whether surviving Jac translations are idiomatic and graph-aware where appropriate.
- [ ] Reject conversions that preserve Python structure mechanically without using Jac concepts.
- [ ] Revise the prompt if conversion notes do not explain meaningful design changes.
- [ ] Scale only after the pilot shows cross-compiled test pass rate above 80% for compilable candidates, and surviving translations are idiomatic.

## Testing And Validation Checklist

- [ ] Every pilot batch has a validation log.
- [ ] Every clean candidate passes the relevant compiler gate.
- [ ] Every testable behavior has either a passing test or a manual-review flag.
- [ ] Every rejected example has a rejection reason.
- [ ] Every prompt revision is recorded with the affected batch IDs.
- [ ] No category scales before its pilot manual review pass rate is at least 80%.

## Failure Conditions And Retry Guidance

- If code generation prompts are ambiguous, add tighter task constraints and examples.
- If debugging pairs contain multiple errors, revise the prompt to require exactly one error type.
- If explanation quality is weak, reduce batch size and provide more examples of good explanations.
- If conversion output is too Python-like, add prompt examples that demonstrate Jac nodes, edges, abilities, and walkers.
- If a category's failures repeat across two prompt revisions, pause that category and inspect the Jac context bundle.
- If scaled batches introduce duplicates, add stronger diversity constraints and run deduplication before continuing.
- If cross-compiled tests reject more than 70% of candidate translations for a Python source function, remove that source from the pool rather than relaxing test criteria.
- If the deterministic test compiler cannot handle a Python test case's assertion format, drop that test case. If zero test cases survive compilation for a source function, flag it for manual review.
- If type inference produces incorrect types causing Jac compilation failures, fall back to untyped translation and record `type_inference_method: none`.
- If grammar-driven and persona-driven prompts produce a narrow domain distribution, add snippet-seeded (OSS-Instruct) batches to broaden semantic-domain coverage.

## Completion Criteria

This task is complete when all four scripted categories have passed pilot generation, validation, manual review, prompt revision, and scale-up decision checks.
