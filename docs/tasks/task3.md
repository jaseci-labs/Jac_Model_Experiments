# Task 3: Validation, Rejection Handling, and Retry Loops

## Purpose

Define the validation gates that protect the clean dataset. This task covers compiler validation, behavior tests, JSON/schema checks, rejected-example handling, retries, pass-rate thresholds, and logs.

## Inputs Needed

- [`context.md`](../context.md) for validation principles.
- [`task1.md`](task1.md) for metadata and storage policy.
- [`task2.md`](task2.md) for OpenAI prompt schemas.
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
- Define cross-compiled test validation as a hard gate for deterministic categories:
  - For `code_gen` examples with deterministic behavior: generate tests in Python, verify 90% line coverage, compile tests to Jac using a deterministic rule-based test compiler, run compiled tests against the Jac code. Pass = proceed. Fail = reject.
  - For `conversion` examples: compile the Python source's existing test suite to Jac. Run compiled tests against each candidate translation. Pass = proceed. Fail = reject the candidate.
  - For `debug` examples where the original working code had cross-compiled tests: `fixed_code` must pass the same cross-compiled tests. Fail = reject the debugging pair.
  - The deterministic test compiler translates Python assertions (`assert f(x) == y`) to Jac assertions. It handles first-order values (ints, strings, booleans, lists, tuples, dicts). It does NOT use an LLM. If a test case uses Python features that cannot be compiled to Jac (e.g., complex object comparisons), that test case is dropped. If zero test cases survive compilation, the example is flagged for manual review rather than auto-rejected.
  - Cross-compiled test validation does not apply to `explanation` or `trajectory` categories.
- Define credibility scoring for untrusted generated tests (CodeDPO, 2410.05605):
  - Generate multiple candidate solutions (15 at temperature 1.5) and multiple candidate tests per task.
  - Build a bipartite pass/fail graph; iterate coupled PageRank scores (damping 0.85, ~10 iterations).
  - Use credibility to select trusted tests and rank solutions instead of trusting any single synthetic test.
  - Cross-compiled tests remain the strongest gate; credibility scoring applies where only generated tests exist.
- Retain compiler/test gate outcomes (0--1 labels) as training data for a reward model that gives a denser signal than the binary gate (DeepSeek-Coder-V2, 2406.11931).
- Define compiler pass handling:
  - Passing examples can move to the next gate.
  - Failing code generation examples move to `rejected/` and may be recycled into debugging seeds.
  - Failing `fixed_code` examples cause the debugging pair to be discarded.
  - Debugging `broken_code` must fail; if it compiles, reject or regenerate the pair.
- Define behavior test expectations:
  - Code generation examples with deterministic behavior get cross-compiled tests as a hard gate. Tests are generated in Python, verified for coverage, and compiled to Jac deterministically. Test failure rejects the example.
  - Conversion examples get equivalence tests where known inputs and outputs are available.
  - Explanation examples do not get behavior tests; they require manual review.
  - Trajectories rely on final compiler validation plus transcript review.
- Define soft-gate behavior:
  - Soft-gate behavior applies only to non-deterministic categories (explanation, trajectory) and examples where cross-compiled tests could not be generated. For deterministic code_gen and conversion examples, test validation is a hard gate — see cross-compiled test validation above.
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
  - Cross-compiled test pass rate target for code_gen: at least 70% of compilable deterministic examples.
  - Cross-compiled test pass rate target for conversion: at least 80% of compilable candidate translations.
  - Python source test coverage minimum before translation: 90% line coverage.
  - Prefer credibility ranking over naive passed-test-count ranking when generated tests are untrusted.
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
  - Stop conversion generation for a Python source function if more than 70% of candidate translations fail cross-compiled tests. Flag the source function as unsuitable.

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
