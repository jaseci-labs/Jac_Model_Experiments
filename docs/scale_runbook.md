# Scale Runbook

This runbook describes the safe operating flow for scaling Jac synthetic data toward a 10,000-15,000 example release.

## Preconditions

- Set `OPENAI_API_KEY` or `OPEN_AI_API_KEY` before live single-turn generation.
- Run all commands from the repository root.
- Do not hand-edit generated dataset artifacts unless the command workflow below cannot represent the review decision.
- Build or verify the deterministic Python-to-Jac test compiler before scaling conversion or Python-sourced code_gen examples.
- Build or verify the filtered Python source pool (docstrings, type-checked, 90% test coverage, no benchmark contamination) before scaling Recipe 2 / conversion generation.
- Configure token accounting so per-example token counts and aggregate per-batch usage are logged from the first batch.
- For generated-test categories, enable credibility scoring (CodeDPO) so DPO pairs are selected from trusted tests rather than any single synthetic test.

## 1. Check Readiness

```bash
python -m data_generation.release readiness --version jac-synth-v0.1.0 --write-report
```

Use the report to confirm the remaining blockers. Before expensive scale, expected blockers should be volume, manual review of new samples, or intentionally unresolved duplicate-review work.

## 2. Dry-Run The Scale Plan

```bash
python -m data_generation.single_turn_generation scale --version jac-synth-v0.1.0 --date 20260511 --target-total 10000 --dry-run
```

For a targeted category catch-up:

```bash
python -m data_generation.single_turn_generation scale --version jac-synth-v0.1.0 --date 20260511 --target-total 10000 --category code_gen --max-batches 2 --dry-run
```

The dry run shows category counts, missing counts, next sequence numbers, hard-example targeting, and paused categories.

## 3. Run A Small Live Smoke Batch

```bash
python -m data_generation.single_turn_generation scale --version jac-synth-v0.1.0 --date 20260511 --target-total 10000 --category code_gen --max-batches 1 --max-retries 1
```

Inspect the generated validation, generation, review, and scale-decision logs before continuing.

## 3.5 Validate Cross-Compiled Test Infrastructure

Before scaling conversion or Python-sourced code_gen categories, verify the cross-compiled test pipeline end to end:

```bash
# Verify Python source pool filtering
python -m data_generation.python_source filter --min-coverage 90 --output dataset/context/python_source/

# Verify test compilation works on a sample
python -m data_generation.test_compiler compile --input dataset/context/python_source/sample.json --output dataset/context/python_source/compiled_tests/

# Run a smoke test: translate 5 Python functions, validate with cross-compiled tests
python -m data_generation.single_turn_generation scale --version jac-synth-v0.1.0 --date 20260511 --target-total 10000 --category conversion --max-batches 1 --max-retries 1 --use-cross-compiled-tests
```

Inspect that cross-compiled tests correctly catch bad translations while accepting good ones. If the test compiler drops too many test cases for a source function (zero surviving), flag that function for removal from the source pool.

## 4. Continue Scaling

```bash
python -m data_generation.single_turn_generation scale --version jac-synth-v0.1.0 --date 20260511 --target-total 10000 --max-batches 20 --max-retries 1 --resume
```

The scale runner chooses the next sequence number and refuses to overwrite existing batch artifacts.

## 5. Collect Trajectories

Plan trajectory collection:

```bash
python -m data_generation.trajectory_generation plan --target-count 2000 --existing-count 0
```

Ingest normalized transcript exports:

```bash
python -m data_generation.trajectory_generation ingest --input transcript.json --date 20260511 --sequence 2 --generation-date 2026-05-11T00:00:00Z
```

Normalized transcripts must include `task`, `turns`, `final_code`, and `validation_result`.

## 6. Complete Manual Review

List pending reviews:

```bash
python -m data_generation.manual_review list --status pending
```

Mark a reviewed example:

```bash
python -m data_generation.manual_review mark --id code_gen-20260511-006-0001 --status passed --reviewer ayush --criteria prompt_clarity=true --criteria idiomatic_jac=true --criteria construct_diversity=true --criteria not_python_like=true --notes "Reviewed."
```

Validate review files:

```bash
python -m data_generation.manual_review validate
```

## 7. Resolve Near Duplicates

Run the release audit to refresh near-duplicate clusters:

```bash
python -m data_generation.release audit --version jac-synth-v0.1.0 --write-report
```

Resolve each cluster by recording a `keep_distinct`, `waived`, or `remove_duplicate` decision through the release tooling before a full release.

## 8. Audit And Freeze

Write an audit snapshot:

```bash
python -m data_generation.release audit --version jac-synth-v0.1.0 --write-report
```

Freeze only after the audit status is ready:

```bash
python -m data_generation.release freeze --version jac-synth-v0.1.0 --training-run TRAINING_RUN_ID
```

Frozen releases include `IMMUTABLE_RELEASE.json` with checksums and an audit fingerprint.

## Failure Playbooks

- Compiler pass drops: pause that category, inspect `dataset/logs/validation`, revise the prompt, and rerun a one-batch smoke test.
- High rejection rate: inspect `dataset/rejected`, group rejection reasons, and lower batch size or target simpler examples.
- OpenAI timeout or transient server error: rerun with `--max-retries 1` or resume the scale run.
- Category imbalance: use `--category` to generate only the missing category.
- Hard-ratio imbalance: rely on dry-run `complexity_target`; if needed, run targeted category batches until the hard ratio returns to the expected band.
- Manual review failure: revise prompt/context or remove the failed examples, then regenerate replacements.
- Duplicate explosion: resolve clusters or strengthen prompt diversity before continuing scale.
- Cross-compiled test pass rate drops: inspect whether the test compiler is too strict (dropping valid assertions) or the translations are genuinely wrong. If test compiler is too strict, expand assertion support. If translations are wrong, revise the translation prompt.
- Python source pool exhaustion: if all filtered Python functions have been translated and more conversion data is needed, expand the source pool by relaxing the docstring requirement to include functions with comments, or by generating Python functions synthetically and filtering them through the same pipeline.
- Token budget overrun: check the aggregate token report by generator and recipe; shift bulk volume from expensive generators to cheap APIs behind the compiler/test gate.
- DPO pairs noisy or ambiguous: filter pairs whose winner/loser credibility scores are near-identical; regenerate candidates at higher temperature (1.5) to widen the credibility spread.
