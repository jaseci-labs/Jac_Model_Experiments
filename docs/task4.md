# Task 4: Single-Turn Category Generation Loops

## Purpose

Define the slow, repeatable generation loops for the four scripted OpenAI API categories: code generation, debugging, explanation, and code conversion. Each category must pass pilot validation before scaling.

## Inputs Needed

- [`context.md`](context.md) for category targets and quality rules.
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

- [ ] Generate a pilot batch of 5 Python-to-Jac conversions.
- [ ] Include at least one function conversion and one class or data-structure conversion.
- [ ] Validate that each item has `python_code`, `jac_code`, and `conversion_notes`.
- [ ] Compile every `jac_code` field.
- [ ] Run equivalence tests where inputs and expected outputs are clear.
- [ ] Manually inspect whether the Jac output is idiomatic and graph-aware where appropriate.
- [ ] Reject conversions that preserve Python structure mechanically without using Jac concepts.
- [ ] Revise the prompt if conversion notes do not explain meaningful design changes.
- [ ] Scale only after converted Jac compiles and preserves behavior in tested cases.

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

## Completion Criteria

This task is complete when all four scripted categories have passed pilot generation, validation, manual review, prompt revision, and scale-up decision checks.
