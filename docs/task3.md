# Task 3: Validation, Rejection Handling, and Retry Loops

## Purpose

Define the validation gates that protect the clean dataset. This task covers compiler validation, behavior tests, JSON/schema checks, rejected-example handling, retries, pass-rate thresholds, and logs.

## Inputs Needed

- `[context.md](context.md)` for validation principles.
- `[task1.md](task1.md)` for metadata and storage policy.
- `[task2.md](task2.md)` for OpenAI prompt schemas.
- Access to the Jac compiler or Jac MCP compiler tool. cursor and claude code both have these so there's nothing to double check here. 
- A small set of expected-good examples for sanity checks.

## Artifacts To Produce

- Validation policy for every generated field that contains Jac code.
- Test harness policy for behavior-checkable examples.
- Rejected-example recycling rules.
- Retry limits for OpenAI generation, JSON parsing, compiler failures, and test failures.
- Batch-level pass-rate thresholds.
- Required validation log fields.

## Step-By-Step Checklist

- Define JSON validation as the first gate:
  - Parse the OpenAI API response as JSON.
  - Confirm the top-level value is an array.
  - Confirm every item matches the category schema.
  - Reject the batch if required fields are missing or empty.
- Define compiler validation fields:
  - Code generation: compile `code`.
  - Debugging: confirm `broken_code` fails as expected and `fixed_code` compiles.
  - Explanation: compile `code`.
  - Conversion: compile `jac_code`.
  - Trajectory: compile the final Jac output from the transcript.
- Define compiler pass handling:
  - Passing examples can move to the next gate.
  - Failing code generation examples move to `rejected/` and may be recycled into debugging seeds.
  - Failing `fixed_code` examples cause the debugging pair to be discarded.
  - Debugging `broken_code` must fail; if it compiles, reject or regenerate the pair.
- Define behavior test expectations:
  - Code generation examples get tests when the prompt defines observable behavior.
  - Conversion examples get equivalence tests where known inputs and outputs are available.
  - Explanation examples do not get behavior tests; they require manual review.
  - Trajectories rely on final compiler validation plus transcript review.
- Define soft-gate behavior:
  - Test failures do not automatically delete examples.
  - Test failures mark examples for manual review.
  - Manual review decides whether the issue is the code, test, or prompt.
- Define retry limits:
  - One immediate retry for malformed JSON after strengthening format instructions.
  - Two retries for low compile pass rate after prompt revision.
  - No blind retries for the same failing prompt version after repeated schema failure.
  - No more than three consecutive failed compiler attempts in a trajectory.
- Define pass-rate thresholds:
  - Pilot compiler pass rate target: at least 80% for code-bearing clean candidates.
  - Scaled-batch compiler pass rate warning: below 70%.
  - Manual review pass rate minimum: 80%.
  - JSON parse pass rate target: 100% before scaling.
- Define validation logs:
  - Batch ID.
  - Prompt version.
  - Context bundle version.
  - Category.
  - Example ID.
  - JSON schema result.
  - Compiler result.
  - Test result.
  - Rejection reason.
  - Retry count.
  - Final disposition.
- Define rejected-example recycling:
  - Only recycle code generation failures into debugging when the compiler error is clear.
  - Preserve the original prompt and compiler error.
  - Do not recycle examples with malformed JSON or missing code fields.
- Define stop conditions:
  - Stop a category if pilot manual review falls below 80%.
  - Stop scaling if duplicates or repeated patterns dominate a batch.
  - Stop generation if compiler failures point to stale Jac context.

## Testing And Validation Checklist

- Confirm every category has a validation path from raw output to final disposition.
- Confirm no example can enter `clean_dataset/` without required compiler validation.
- Confirm debugging examples treat `broken_code` failure as expected, not as a rejected state.
- Confirm test failures are marked for review instead of silently accepted.
- Confirm validation logs are sufficient to reproduce why an example was kept or rejected.

## Failure Conditions And Retry Guidance

- If JSON validation fails, revise prompt formatting instructions before retrying.
- If compiler pass rate is low, inspect actual compiler errors before changing prompts.
- If test failures are common, review whether tests are incorrectly specified before blaming generated code.
- If rejected examples are being recycled into poor debugging pairs, tighten recycling criteria.
- If logs do not explain failures clearly, expand log fields before generating more data.

## Completion Criteria

This task is complete when every generated example has a clear validation path, retry policy, rejection policy, and logging requirement before category-specific generation begins.