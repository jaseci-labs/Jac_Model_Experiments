import json

from data_generation.foundation import SCRIPTED_CATEGORIES
from data_generation.prompt_design import build_prompt_request
from data_generation.single_turn_generation import FakeGenerationClient, SingleTurnPilotRunner
from data_generation.validation import CompilerResult, Disposition, validate_example


def fake_compiler(code: str) -> CompilerResult:
    if "BROKEN" in code:
        return CompilerResult(passed=False, error_message="synthetic compiler failure")
    return CompilerResult(passed=True)


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
