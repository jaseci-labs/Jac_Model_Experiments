import json

from data_generation.manual_review import (
    CATEGORY_REVIEW_CRITERIA,
    ReviewRecord,
    ScaleDecisionStatus,
    build_prompt_revision_record,
    decide_scale_up,
    default_review_records,
    list_review_records,
    main,
    mark_review_record,
    validate_review_files,
)
from data_generation.validation import CompilerResult, validate_example


def passing_compiler(_code: str) -> CompilerResult:
    return CompilerResult(passed=True)


def test_scale_decision_approves_when_all_task4_gates_pass():
    results = [
        validate_example(
            category="code_gen",
            example={"prompt": "Say hi", "code": "valid jac", "complexity": "simple"},
            compiler=passing_compiler,
            test_pass=True,
        )
        for _ in range(5)
    ]
    reviews = [
        ReviewRecord(
            batch_id="20260508-code_gen-001",
            category="code_gen",
            example_id=f"code_gen-20260508-001-000{i}",
            review_status="passed",
            reviewer="manual-reviewer",
            criteria_results={"idiomatic_jac": True},
            notes="ok",
        )
        for i in range(5)
    ]

    decision = decide_scale_up(
        batch_id="20260508-code_gen-001",
        category="code_gen",
        validation_results=results,
        review_records=reviews,
    )

    assert decision.decision == ScaleDecisionStatus.SCALE_READY
    assert decision.manual_review_pass_rate == 1.0
    assert decision.prompt_revision_required is False


def test_scale_decision_requires_prompt_revision_when_manual_review_fails():
    results = [
        validate_example(
            category="conversion",
            example={"python_code": "print('hi')", "jac_code": "valid jac", "conversion_notes": "Uses Jac."},
            compiler=passing_compiler,
        )
        for _ in range(5)
    ]
    reviews = [
        ReviewRecord(
            batch_id="20260508-conversion-001",
            category="conversion",
            example_id=f"conversion-20260508-001-000{i}",
            review_status="failed" if i == 0 else "passed",
            reviewer="manual-reviewer",
            criteria_results={"not_python_like": i != 0},
            notes="too Python-like" if i == 0 else "ok",
        )
        for i in range(5)
    ]

    decision = decide_scale_up(
        batch_id="20260508-conversion-001",
        category="conversion",
        validation_results=results,
        review_records=reviews,
    )

    assert decision.decision == ScaleDecisionStatus.REVISE_PROMPT
    assert decision.manual_review_pass_rate == 0.8
    assert decision.prompt_revision_required is True
    assert "blocking review issue" in decision.reasons


def test_scale_decision_pauses_after_repeated_prompt_revisions():
    results = [
        validate_example(
            category="code_gen",
            example={"prompt": "Say hi", "code": "valid jac", "complexity": "simple"},
            compiler=passing_compiler,
        )
        for _ in range(5)
    ]

    decision = decide_scale_up(
        batch_id="20260508-code_gen-006",
        category="code_gen",
        validation_results=results,
        review_records=[],
        prompt_revision_count=2,
    )

    assert decision.decision == ScaleDecisionStatus.PAUSED
    assert decision.prompt_revision_required is False
    assert "prompt revision retry limit reached" in decision.reasons


def test_prompt_revision_record_matches_required_policy_fields():
    record = build_prompt_revision_record(
        prompt_version="prompt-code_gen-v2",
        previous_prompt_version="prompt-code_gen-v1",
        category="code_gen",
        changed_fields=["user_prompt"],
        reason_for_change="Pilot prompts were vague.",
        batch_ids_affected=["20260508-code_gen-001"],
        observed_pass_rate_before=0.6,
        observed_pass_rate_after=None,
        changed_at="2026-05-08T00:00:00Z",
        notes="Added tighter task constraints.",
    )

    assert record["prompt_version"] == "prompt-code_gen-v2"
    assert record["batch_ids_affected"] == ["20260508-code_gen-001"]
    assert record["observed_pass_rate_after"] is None


def test_category_review_criteria_cover_task4_categories():
    assert "prompt_clarity" in CATEGORY_REVIEW_CRITERIA["code_gen"]
    assert "exactly_one_error" in CATEGORY_REVIEW_CRITERIA["debug"]
    assert "accurate_explanation" in CATEGORY_REVIEW_CRITERIA["explanation"]
    assert "preserves_behavior" in CATEGORY_REVIEW_CRITERIA["conversion"]


def test_category_review_criteria_cover_trajectory():
    assert CATEGORY_REVIEW_CRITERIA["trajectory"] == (
        "task_solved",
        "logical_mcp_tool_use",
        "compiler_recovery_present",
        "final_code_validated",
        "idiomatic_jac",
        "no_private_context",
    )


def test_default_review_records_support_trajectory():
    records = default_review_records(
        "20260511-trajectory-001",
        "trajectory",
        ["trajectory-20260511-001-0001"],
    )

    assert records[0].review_status == "pending"
    assert records[0].criteria_results == {
        "task_solved": False,
        "logical_mcp_tool_use": False,
        "compiler_recovery_present": False,
        "final_code_validated": False,
        "idiomatic_jac": False,
        "no_private_context": False,
    }


def test_list_mark_and_validate_review_records(tmp_path):
    review_dir = tmp_path / "dataset/review/code_gen"
    review_dir.mkdir(parents=True)
    review_path = review_dir / "20260511-code_gen-001-review.json"
    review_path.write_text(
        json.dumps(
            [
                {
                    "batch_id": "20260511-code_gen-001",
                    "category": "code_gen",
                    "example_id": "code_gen-20260511-001-0001",
                    "review_status": "pending",
                    "reviewer": "manual-reviewer",
                    "criteria_results": {criterion: False for criterion in CATEGORY_REVIEW_CRITERIA["code_gen"]},
                    "notes": "Pending manual review.",
                }
            ]
        )
    )

    pending = list_review_records(tmp_path, status="pending")
    updated = mark_review_record(
        tmp_path,
        example_id="code_gen-20260511-001-0001",
        status="passed",
        reviewer="ayush",
        criteria_results={criterion: True for criterion in CATEGORY_REVIEW_CRITERIA["code_gen"]},
        notes="Looks good.",
    )
    validation = validate_review_files(tmp_path)

    assert pending[0]["example_id"] == "code_gen-20260511-001-0001"
    assert updated["review_status"] == "passed"
    assert validation["status"] == "complete"
    assert json.loads(review_path.read_text())[0]["reviewer"] == "ayush"


def test_mark_review_record_rejects_invalid_criteria_and_requires_failure_notes(tmp_path):
    review_dir = tmp_path / "dataset/review/debug"
    review_dir.mkdir(parents=True)
    (review_dir / "20260511-debug-001-review.json").write_text(
        json.dumps(
            [
                {
                    "batch_id": "20260511-debug-001",
                    "category": "debug",
                    "example_id": "debug-20260511-001-0001",
                    "review_status": "pending",
                    "reviewer": "manual-reviewer",
                    "criteria_results": {criterion: False for criterion in CATEGORY_REVIEW_CRITERIA["debug"]},
                    "notes": "Pending manual review.",
                }
            ]
        )
    )

    try:
        mark_review_record(
            tmp_path,
            example_id="debug-20260511-001-0001",
            status="failed",
            reviewer="ayush",
            criteria_results={"not_a_debug_criterion": False},
            notes="Bad.",
        )
    except ValueError as error:
        assert "invalid review criteria" in str(error)
    else:
        raise AssertionError("expected invalid criteria to fail")

    try:
        mark_review_record(
            tmp_path,
            example_id="debug-20260511-001-0001",
            status="failed",
            reviewer="ayush",
            criteria_results={criterion: False for criterion in CATEGORY_REVIEW_CRITERIA["debug"]},
            notes="",
        )
    except ValueError as error:
        assert "notes are required" in str(error)
    else:
        raise AssertionError("expected missing notes to fail")


def test_manual_review_cli_lists_marks_and_validates(tmp_path, monkeypatch, capsys):
    review_dir = tmp_path / "dataset/review/explanation"
    review_dir.mkdir(parents=True)
    (review_dir / "20260511-explanation-001-review.json").write_text(
        json.dumps(
            [
                {
                    "batch_id": "20260511-explanation-001",
                    "category": "explanation",
                    "example_id": "explanation-20260511-001-0001",
                    "review_status": "pending",
                    "reviewer": "manual-reviewer",
                    "criteria_results": {criterion: False for criterion in CATEGORY_REVIEW_CRITERIA["explanation"]},
                    "notes": "Pending manual review.",
                }
            ]
        )
    )
    monkeypatch.chdir(tmp_path)

    assert main(["list", "--status", "pending"]) == 0
    assert "explanation-20260511-001-0001" in capsys.readouterr().out
    assert (
        main(
            [
                "mark",
                "--id",
                "explanation-20260511-001-0001",
                "--status",
                "passed",
                "--reviewer",
                "ayush",
                "--criteria",
                "accurate_explanation=true",
                "--criteria",
                "specific_jac_semantics=true",
                "--criteria",
                "granularity_matches_request=true",
                "--criteria",
                "not_python_behavior=true",
                "--notes",
                "Reviewed.",
            ]
        )
        == 0
    )
    assert main(["validate"]) == 0
    assert '"status": "complete"' in capsys.readouterr().out
