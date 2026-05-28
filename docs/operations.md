# Operations

Run commands from the repository root. Set `OPENAI_API_KEY` or `OPEN_AI_API_KEY` before live OpenAI generation.

## Tests

```bash
python -m pytest
```

## Documentation Stats

Regenerate the stats snapshot:

```bash
python -m data_generation.docs_stats --version jac-synth-v0.1.0 --output docs/stats.md
```

Write a raw JSON snapshot instead:

```bash
python -m data_generation.docs_stats --version jac-synth-v0.1.0 --output docs/stats.json --json
```

## Release Readiness

Check readiness without writing artifacts:

```bash
python -m data_generation.release readiness --version jac-synth-v0.1.0
```

Write a readiness report:

```bash
python -m data_generation.release readiness --version jac-synth-v0.1.0 --write-report
```

## Single-Turn Scaling

Dry-run the scale plan:

```bash
python -m data_generation.single_turn_generation scale --version jac-synth-v0.1.0 --date 20260511 --target-total 10000 --dry-run
```

Run one smoke batch for a category:

```bash
python -m data_generation.single_turn_generation scale --version jac-synth-v0.1.0 --date 20260511 --target-total 10000 --category code_gen --max-batches 1 --max-retries 1
```

## Trajectories

Plan trajectory collection:

```bash
python -m data_generation.trajectory_generation plan --target-count 2000 --existing-count 0
```

Ingest a normalized transcript:

```bash
python -m data_generation.trajectory_generation ingest --input transcript.json --date 20260511 --sequence 2 --generation-date 2026-05-11T00:00:00Z
```

## Python Source Pool And Cross-Compiled Tests

Filter Python source functions for the translation pool:

```bash
python -m data_generation.python_source filter --min-coverage 90 --require-docstring --require-typecheck --exclude-benchmarks --output dataset/context/python_source/
```

Generate and validate Python test suites for source functions:

```bash
python -m data_generation.python_source generate-tests --input dataset/context/python_source/ --min-coverage 90 --max-suites 5
```

Infer Python types from test execution:

```bash
python -m data_generation.python_source infer-types --input dataset/context/python_source/ --method runtime_observation
```

Compile Python tests to Jac:

```bash
python -m data_generation.test_compiler compile --input dataset/context/python_source/tests/ --output dataset/context/python_source/compiled_tests/
```

Validate a conversion batch with cross-compiled tests:

```bash
python -m data_generation.single_turn_generation validate-cross-compiled --batch-id 20260511-conversion-001 --tests dataset/context/python_source/compiled_tests/
```

Check Python source pool stats:

```bash
python -m data_generation.python_source stats
```

## Token Accounting And Credibility Scoring

Report per-example and aggregate token usage:

```bash
python -m data_generation.tokens report --version jac-synth-v0.1.0 --by generator,recipe
```

Compute mutual code↔test credibility scores for a task's candidate solutions and tests (CodeDPO):

```bash
python -m data_generation.credibility score --task-id code_gen-20260511-006-0001 --solutions 15 --temperature 1.5 --damping 0.85 --iterations 10
```

Build DPO preference pairs from credibility scores (drops near-identical-score pairs, adds runtime-efficiency pairs):

```bash
python -m data_generation.credibility build-dpo --version jac-synth-v0.1.0 --include-runtime-efficiency
```

Cross-validate the idiom judge with an out-of-family model:

```bash
python -m data_generation.manual_review cross-judge --sample 50 --judge-family qwen
```

## Manual Review

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

## Audit And Freeze

Write an audit report:

```bash
python -m data_generation.release audit --version jac-synth-v0.1.0 --write-report
```

Resolve a near-duplicate cluster:

```bash
python -m data_generation.release resolve-near-duplicate --version jac-synth-v0.1.0 --cluster-id CLUSTER_ID --action keep_distinct --reviewer ayush --notes "Distinct task intent."
```

Freeze a ready release:

```bash
python -m data_generation.release freeze --version jac-synth-v0.1.0 --training-run TRAINING_RUN_ID
```

For the longer operational sequence and failure playbooks, see [`scale_runbook.md`](scale_runbook.md).
