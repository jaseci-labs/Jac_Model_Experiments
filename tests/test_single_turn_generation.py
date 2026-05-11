import json

import pytest

from data_generation.foundation import SCRIPTED_CATEGORIES
from data_generation.prompt_design import build_prompt_request
from data_generation.single_turn_generation import (
    FakeGenerationClient,
    SingleTurnPilotRunner,
    build_scale_plan,
)
from data_generation.validation import CompilerResult, Disposition, validate_example


def fake_compiler(code: str) -> CompilerResult:
    if "BROKEN" in code:
        return CompilerResult(passed=False, error_message="synthetic compiler failure")
    return CompilerResult(passed=True)


class SequencedGenerationClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate_batch(self, prompt_request):
        self.calls += 1
        response = self.responses.pop(0)
        from data_generation.openai_generation import GenerationResult

        return GenerationResult(examples=response, raw_response={"response": response})


def test_task4_prerequisites_cover_every_scripted_category():
    examples = {
        "code_gen": {"prompt": "Say hi", "code": "valid jac", "complexity": "simple"},
        "debug": {
            "broken_code": "BROKEN jac",
            "error_type": "syntax",
            "error_message": "synthetic compiler failure",
            "fixed_code": "valid jac",
            "fix_explanation": "Removed the invalid token.",
        },
        "explanation": {"code": "valid jac", "granularity": "line", "explanation": "Explains the line."},
        "conversion": {"python_code": "print('hi')", "jac_code": "valid jac", "conversion_notes": "Uses Jac."},
    }

    for category in SCRIPTED_CATEGORIES:
        request = build_prompt_request(category=category, context_bundle="Jac context")
        result = validate_example(category=category, example=examples[category], compiler=fake_compiler)

        assert request["category"] == category
        assert result.disposition == Disposition.CLEAN


def test_run_category_pilot_writes_artifacts_and_splits_clean_rejected(tmp_path):
    examples = [
        {"prompt": "Say hi", "code": "valid jac", "complexity": "simple"},
        {"prompt": "Broken", "code": "BROKEN jac", "complexity": "medium"},
    ]
    runner = SingleTurnPilotRunner(
        workspace_root=tmp_path,
        generation_client=FakeGenerationClient({"code_gen": examples}),
        compiler=fake_compiler,
        context_bundle="Jac context",
    )

    summary = runner.run_category_pilot(category="code_gen", date="20260508", sequence=1)

    assert summary.batch_id == "20260508-code_gen-001"
    assert summary.clean_count == 1
    assert summary.rejected_count == 1
    assert summary.review_count == 0
    assert (tmp_path / "dataset/raw_output/code_gen/20260508-code_gen-001.json").exists()
    assert (tmp_path / "dataset/clean_dataset/code_gen/20260508-code_gen-001.jsonl").exists()
    assert (tmp_path / "dataset/rejected/code_gen/20260508-code_gen-001.jsonl").exists()
    assert (tmp_path / "dataset/logs/validation/20260508-code_gen-001.jsonl").exists()
    assert (tmp_path / "dataset/logs/scale_decisions/20260508-code_gen-001.json").exists()
    assert (tmp_path / "dataset/logs/prompt_revisions/20260508-code_gen-001.json").exists()


def test_run_category_pilot_writes_release_ready_metadata(tmp_path):
    examples = [
        {"python_code": "print('hi')", "jac_code": "valid jac", "conversion_notes": "Uses Jac."},
    ]
    runner = SingleTurnPilotRunner(
        workspace_root=tmp_path,
        generation_client=FakeGenerationClient({"conversion": examples}),
        compiler=fake_compiler,
        context_bundle="Jac context",
    )

    runner.run_category_pilot(category="conversion", date="20260508", sequence=1, requested_count=20)

    clean_path = tmp_path / "dataset/clean_dataset/conversion/20260508-conversion-001.jsonl"
    record = json.loads(clean_path.read_text().splitlines()[0])
    assert record["id"] == "conversion-20260508-001-0001"
    assert record["batch_id"] == "20260508-conversion-001"
    assert record["category"] == "conversion"
    assert record["complexity"] == "medium"
    assert record["compiler_pass"] is True
    assert record["test_pass"] is None
    assert record["manually_reviewed"] is False
    assert record["generator"] == "openai-api"
    assert record["source_prompt_version"] == "prompt-conversion-v1"
    assert record["context_bundle_version"] == "jac-context-v1"
    assert record["validator_version"] == "validator-v1"
    assert record["dataset_version"] == "jac-synth-v0.1.0"
    assert record["review_status"] == "pending"
    assert record["rejection_reason"] is None


def test_generation_log_includes_diagnostics_and_pass_rates(tmp_path):
    examples = [{"prompt": "Say hi", "code": "valid jac", "complexity": "simple"}]
    runner = SingleTurnPilotRunner(
        workspace_root=tmp_path,
        generation_client=FakeGenerationClient({"code_gen": examples}),
        compiler=fake_compiler,
        context_bundle="Jac context",
    )

    runner.run_category_pilot(category="code_gen", date="20260508", sequence=1, requested_count=20)

    generation_log = json.loads((tmp_path / "dataset/logs/generation/20260508-code_gen-001.json").read_text())
    assert generation_log["requested_count"] == 20
    assert generation_log["retry_count"] == 0
    assert generation_log["prompt_hash"]
    assert generation_log["context_bundle_hash"]
    assert generation_log["pass_rates"]["compiler_pass_rate"] == 1.0
    assert generation_log["rejection_reasons"] == {}


def test_dry_run_accepts_requested_count(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "context.md").write_text("strategy")
    (docs / "dataset_foundation.md").write_text("foundation")

    from data_generation.single_turn_generation import main

    assert main(["run-pilots", "--dry-run", "--date", "20260508", "--requested-count", "20"]) == 0

    assert '"requested_count": 20' in capsys.readouterr().out


def test_debug_pilot_requires_broken_code_to_fail_and_fixed_code_to_pass(tmp_path):
    examples = [
        {
            "broken_code": "BROKEN jac",
            "error_type": "syntax",
            "error_message": "synthetic compiler failure",
            "fixed_code": "valid jac",
            "fix_explanation": "Removed the invalid token.",
        },
        {
            "broken_code": "BROKEN jac",
            "error_type": "syntax",
            "error_message": "synthetic compiler failure",
            "fixed_code": "BROKEN jac",
            "fix_explanation": "Bad fix.",
        },
    ]
    runner = SingleTurnPilotRunner(
        workspace_root=tmp_path,
        generation_client=FakeGenerationClient({"debug": examples}),
        compiler=fake_compiler,
        context_bundle="Jac context",
    )

    summary = runner.run_category_pilot(category="debug", date="20260508", sequence=1)

    assert summary.clean_count == 1
    assert summary.discarded_count == 1


def test_malformed_generation_gets_retry_record_and_no_clean_output(tmp_path):
    runner = SingleTurnPilotRunner(
        workspace_root=tmp_path,
        generation_client=FakeGenerationClient({"code_gen": "not json"}),
        compiler=fake_compiler,
        context_bundle="Jac context",
    )

    summary = runner.run_category_pilot(category="code_gen", date="20260508", sequence=1)

    assert summary.clean_count == 0
    assert summary.rejected_count == 0
    retry_path = tmp_path / "dataset/logs/retry/20260508-code_gen-001.json"
    assert json.loads(retry_path.read_text())["errors"][0].startswith("malformed JSON")
    assert not (tmp_path / "dataset/clean_dataset/code_gen/20260508-code_gen-001.jsonl").exists()


def test_malformed_generation_retries_and_uses_successful_response(tmp_path):
    client = SequencedGenerationClient(
        [
            "not json",
            [{"prompt": "Say hi", "code": "valid jac", "complexity": "simple"}],
        ]
    )
    runner = SingleTurnPilotRunner(
        workspace_root=tmp_path,
        generation_client=client,
        compiler=fake_compiler,
        context_bundle="Jac context",
    )

    summary = runner.run_category_pilot(category="code_gen", date="20260508", sequence=1, parse_retry_limit=1)

    assert client.calls == 2
    assert summary.clean_count == 1
    retry_path = tmp_path / "dataset/logs/retry/20260508-code_gen-001.json"
    assert json.loads(retry_path.read_text())["retry_count"] == 1


def test_dry_run_materializes_context_without_calling_openai(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "context.md").write_text("strategy")
    (docs / "dataset_foundation.md").write_text("foundation")

    from data_generation.single_turn_generation import main

    exit_code = main(["run-pilots", "--dry-run", "--date", "20260508", "--start-sequence", "1"])

    assert exit_code == 0
    assert (tmp_path / "dataset/context/jac-context-v1.md").exists()
    assert not (tmp_path / "dataset/raw_output/code_gen/20260508-code_gen-001.json").exists()


def test_dry_run_accepts_single_category_filter(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "context.md").write_text("strategy")
    (docs / "dataset_foundation.md").write_text("foundation")

    from data_generation.single_turn_generation import main

    assert main(["run-pilots", "--dry-run", "--date", "20260508", "--category", "conversion"]) == 0

    assert '"categories": ["conversion"]' in capsys.readouterr().out


def test_dry_run_accepts_prompt_version_number(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "context.md").write_text("strategy")
    (docs / "dataset_foundation.md").write_text("foundation")

    from data_generation.single_turn_generation import main

    assert main(["run-pilots", "--dry-run", "--date", "20260508", "--prompt-version-number", "2"]) == 0

    assert '"prompt_version_number": 2' in capsys.readouterr().out


def test_scale_plan_computes_missing_counts_next_sequence_and_batch_plan(tmp_path):
    runner = SingleTurnPilotRunner(
        workspace_root=tmp_path,
        generation_client=FakeGenerationClient({"code_gen": []}),
        compiler=fake_compiler,
        context_bundle="Jac context",
    )
    runner.run_category_pilot(category="code_gen", date="20260508", sequence=1)

    plan = build_scale_plan(
        tmp_path,
        version="jac-synth-v0.1.0",
        date="20260508",
        target_total=10_000,
        category="code_gen",
        batch_size=50,
        max_batches=2,
    )

    assert plan["dry_run"] is True
    assert plan["target_total"] == 10_000
    assert plan["categories"]["code_gen"]["current_count"] == 0
    assert plan["categories"]["code_gen"]["target_count"] == 3_000
    assert plan["categories"]["code_gen"]["missing_count"] == 3_000
    assert plan["batches"][0]["sequence"] == 2
    assert plan["batches"][0]["requested_count"] == 50
    assert len(plan["batches"]) == 2


def test_scale_plan_marks_hard_complexity_target_when_category_ratio_is_low(tmp_path):
    examples = [{"prompt": "Say hi", "code": "valid jac", "complexity": "simple"}]
    runner = SingleTurnPilotRunner(
        workspace_root=tmp_path,
        generation_client=FakeGenerationClient({"code_gen": examples}),
        compiler=fake_compiler,
        context_bundle="Jac context",
    )
    runner.run_category_pilot(category="code_gen", date="20260508", sequence=1)

    plan = build_scale_plan(
        tmp_path,
        version="jac-synth-v0.1.0",
        date="20260508",
        target_total=10_000,
        category="code_gen",
        batch_size=50,
        max_batches=1,
    )

    assert plan["categories"]["code_gen"]["hard_ratio"] == 0.0
    assert plan["categories"]["code_gen"]["complexity_target"] == "hard"
    assert plan["batches"][0]["complexity_target"] == "hard"


def test_scale_plan_skips_paused_category_after_repeated_prompt_revisions(tmp_path):
    revisions = tmp_path / "dataset/logs/prompt_revisions"
    revisions.mkdir(parents=True)
    for sequence in (1, 2):
        (revisions / f"20260508-code_gen-{sequence:03d}.json").write_text(
            json.dumps(
                {
                    "category": "code_gen",
                    "observed_pass_rate_before": 0.2,
                    "previous_prompt_version": "prompt-code_gen-v4",
                    "reason_for_change": "compiler pass rate below pilot threshold",
                }
            )
        )

    plan = build_scale_plan(
        tmp_path,
        version="jac-synth-v0.1.0",
        date="20260508",
        target_total=10_000,
        category="code_gen",
        max_batches=1,
    )

    assert plan["categories"]["code_gen"]["status"] == "paused"
    assert plan["batches"] == []


def test_scale_plan_does_not_pause_for_manual_review_only_revision_logs(tmp_path):
    revisions = tmp_path / "dataset/logs/prompt_revisions"
    revisions.mkdir(parents=True)
    for sequence in (1, 2):
        (revisions / f"20260508-code_gen-{sequence:03d}.json").write_text(
            json.dumps(
                {
                    "category": "code_gen",
                    "observed_pass_rate_before": 1.0,
                    "previous_prompt_version": "prompt-code_gen-v4",
                    "reason_for_change": "Pilot batch did not meet Task 4 validation or manual review scale gates.",
                }
            )
        )

    plan = build_scale_plan(
        tmp_path,
        version="jac-synth-v0.1.0",
        date="20260508",
        target_total=10_000,
        category="code_gen",
        max_batches=1,
    )

    assert plan["categories"]["code_gen"]["status"] == "planned"
    assert len(plan["batches"]) == 1


def test_run_category_pilot_refuses_to_overwrite_existing_batch(tmp_path):
    examples = [{"prompt": "Say hi", "code": "valid jac", "complexity": "simple"}]
    runner = SingleTurnPilotRunner(
        workspace_root=tmp_path,
        generation_client=FakeGenerationClient({"code_gen": examples}),
        compiler=fake_compiler,
        context_bundle="Jac context",
    )
    runner.run_category_pilot(category="code_gen", date="20260508", sequence=1)

    with pytest.raises(FileExistsError):
        runner.run_category_pilot(category="code_gen", date="20260508", sequence=1)


def test_scale_dry_run_cli_writes_no_generation_artifacts(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "context.md").write_text("strategy")
    (docs / "dataset_foundation.md").write_text("foundation")

    from data_generation.single_turn_generation import main

    assert (
        main(
            [
                "scale",
                "--version",
                "jac-synth-v0.1.0",
                "--date",
                "20260508",
                "--target-total",
                "10000",
                "--category",
                "code_gen",
                "--max-batches",
                "1",
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    assert '"batches"' in output
    assert '"category": "code_gen"' in output
    assert not (tmp_path / "dataset/logs/scale_runs").exists()


def test_execute_scale_plan_writes_scale_run_manifest(tmp_path):
    from data_generation.single_turn_generation import _execute_scale_plan, _write_scale_run_manifest

    runner = SingleTurnPilotRunner(
        workspace_root=tmp_path,
        generation_client=FakeGenerationClient(
            {"code_gen": [{"prompt": "Say hi", "code": "valid jac", "complexity": "simple"}]}
        ),
        compiler=fake_compiler,
        context_bundle="Jac context",
    )
    plan = {
        "version": "jac-synth-v0.1.0",
        "target_total": 10_000,
        "batches": [
            {
                "category": "code_gen",
                "date": "20260508",
                "sequence": 1,
                "batch_id": "20260508-code_gen-001",
                "requested_count": 1,
            }
        ],
    }

    summaries = _execute_scale_plan(runner, plan, prompt_version_number=None)
    manifest = _write_scale_run_manifest(
        tmp_path,
        args=type(
            "Args",
            (),
            {"date": "20260508", "version": "jac-synth-v0.1.0", "target_total": 10_000, "command": "scale"},
        )(),
        plan=plan,
        summaries=summaries,
    )

    assert summaries[0]["batch_id"] == "20260508-code_gen-001"
    assert manifest["completed_batch_count"] == 1
    assert (tmp_path / "dataset/logs/scale_runs/20260508-scale-001.json").exists()
