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
