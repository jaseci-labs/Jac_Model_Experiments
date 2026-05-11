from data_generation.manual_review import (
    CATEGORY_REVIEW_CRITERIA,
    ReviewRecord,
    ScaleDecisionStatus,
    build_prompt_revision_record,
    decide_scale_up,
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
