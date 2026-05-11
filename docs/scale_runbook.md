# Scale Runbook

This runbook describes the safe operating flow for scaling Jac synthetic data toward a 10,000-15,000 example release.

## Preconditions

- Set `OPENAI_API_KEY` or `OPEN_AI_API_KEY` before live single-turn generation.
- Run all commands from the repository root.
- Do not hand-edit generated dataset artifacts unless the command workflow below cannot represent the review decision.

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
