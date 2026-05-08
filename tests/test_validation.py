import json

from data_generation.validation import (
    PASS_RATE_THRESHOLDS,
    RETRY_LIMITS,
    VALIDATION_LOG_REQUIRED_FIELDS,
    CompilerResult,
    Disposition,
    build_validation_log_record,
    calculate_pass_rates,
    compile_fields_for_example,
    evaluate_batch_thresholds,
    parse_json_batch,
    should_recycle_rejected_example,
    validate_batch_schema,
    validate_example,
)


def fake_compiler(code: str) -> CompilerResult:
    if "BROKEN" in code:
        return CompilerResult(passed=False, error_message="syntax error near BROKEN")
    return CompilerResult(passed=True)


def test_parse_json_batch_accepts_valid_array():
    parsed, errors = parse_json_batch('[{"prompt": "Say hi", "code": "valid jac", "complexity": "simple"}]')

    assert parsed == [{"prompt": "Say hi", "code": "valid jac", "complexity": "simple"}]
    assert errors == []


def test_parse_json_batch_reports_malformed_json():
    parsed, errors = parse_json_batch('[{"prompt": "missing close"')

    assert parsed is None
    assert any("malformed JSON" in error for error in errors)


def test_parse_json_batch_requires_top_level_array():
    parsed, errors = parse_json_batch('{"prompt": "not an array"}')

    assert parsed is None
    assert "top-level JSON value must be an array" in errors


def test_validate_batch_schema_reports_missing_required_field():
    errors = validate_batch_schema("code_gen", [{"prompt": "Say hi", "complexity": "simple"}])

    assert "item 0 missing required field: code" in errors


def test_validate_batch_schema_reports_empty_required_string():
    errors = validate_batch_schema(
        "code_gen",
        [{"prompt": "", "code": "valid jac", "complexity": "simple"}],
    )

    assert "item 0 field prompt must be a non-empty string" in errors


def test_validate_batch_schema_reports_unexpected_field():
    errors = validate_batch_schema(
        "code_gen",
        [{"prompt": "Say hi", "code": "valid jac", "complexity": "simple", "extra": "nope"}],
    )

    assert "item 0 has unexpected field: extra" in errors


def test_code_generation_compiler_gate_compiles_code_field():
    passed, results, reason = compile_fields_for_example(
        "code_gen",
        {"prompt": "Say hi", "code": "valid jac", "complexity": "simple"},
        fake_compiler,
    )

    assert passed is True
    assert reason is None
    assert results[0].field_name == "code"
    assert results[0].expected_to_compile is True


def test_explanation_compiler_gate_compiles_code_field():
    passed, results, reason = compile_fields_for_example(
        "explanation",
        {"code": "valid jac", "granularity": "block", "explanation": "Explains the code."},
        fake_compiler,
    )

    assert passed is True
    assert reason is None
    assert results[0].field_name == "code"


def test_conversion_compiler_gate_compiles_jac_code_field():
    passed, results, reason = compile_fields_for_example(
        "conversion",
        {"python_code": "print('hi')", "jac_code": "valid jac", "conversion_notes": "Uses Jac."},
        fake_compiler,
    )

    assert passed is True
    assert reason is None
    assert results[0].field_name == "jac_code"


def test_debug_compiler_gate_expects_broken_code_to_fail_and_fixed_code_to_pass():
    passed, results, reason = compile_fields_for_example(
        "debug",
        {
            "broken_code": "BROKEN jac",
            "error_type": "syntax",
            "error_message": "syntax error near BROKEN",
            "fixed_code": "valid jac",
            "fix_explanation": "Removed the invalid token.",
        },
        fake_compiler,
    )

    assert passed is True
    assert reason is None
    assert [result.field_name for result in results] == ["broken_code", "fixed_code"]
    assert results[0].expected_to_compile is False
    assert results[1].expected_to_compile is True


def test_debug_compiler_gate_rejects_broken_code_that_compiles():
    passed, _results, reason = compile_fields_for_example(
        "debug",
        {
            "broken_code": "valid jac",
            "error_type": "syntax",
            "error_message": "syntax error near BROKEN",
            "fixed_code": "valid jac",
            "fix_explanation": "Removed the invalid token.",
        },
        fake_compiler,
    )

    assert passed is False
    assert reason == "broken_code compiled but must fail"


def test_debug_fixed_code_failure_is_discarded():
    result = validate_example(
        category="debug",
        example={
            "broken_code": "BROKEN jac",
            "error_type": "syntax",
            "error_message": "syntax error near BROKEN",
            "fixed_code": "BROKEN fixed jac",
            "fix_explanation": "Removed the invalid token.",
        },
        compiler=fake_compiler,
    )

    assert result.disposition == Disposition.DISCARDED
    assert result.rejection_reason == "fixed_code failed compiler validation"


def test_code_generation_compile_failure_is_rejected_and_recyclable():
    result = validate_example(
        category="code_gen",
        example={"prompt": "Say hi", "code": "BROKEN jac", "complexity": "simple"},
        compiler=fake_compiler,
    )

    assert result.disposition == Disposition.REJECTED
    assert result.rejection_reason == "code failed compiler validation"
    assert should_recycle_rejected_example(result) is True


def test_clean_code_generation_example_with_compiler_pass_is_clean():
    result = validate_example(
        category="code_gen",
        example={"prompt": "Say hi", "code": "valid jac", "complexity": "simple"},
        compiler=fake_compiler,
    )

    assert result.disposition == Disposition.CLEAN
    assert result.compiler_pass is True


def test_behavior_test_failure_marks_example_for_review():
    result = validate_example(
        category="code_gen",
        example={"prompt": "Say hi", "code": "valid jac", "complexity": "simple"},
        compiler=fake_compiler,
        test_pass=False,
        retry_count=1,
    )

    assert result.disposition == Disposition.REVIEW
    assert result.rejection_reason == "behavior test failed; manual review required"
    assert result.retry_count == 1


def test_explanation_example_does_not_require_behavior_test_result():
    result = validate_example(
        category="explanation",
        example={"code": "valid jac", "granularity": "module", "explanation": "Explains the module."},
        compiler=fake_compiler,
    )

    assert result.disposition == Disposition.CLEAN
    assert result.test_pass is None


def test_retry_limits_and_pass_rate_thresholds_match_task3_policy():
    assert RETRY_LIMITS == {
        "malformed_json_immediate_retry": 1,
        "low_compile_pass_rate_prompt_revisions": 2,
        "trajectory_consecutive_compiler_failures": 3,
    }
    assert PASS_RATE_THRESHOLDS == {
        "pilot_compiler_pass_target": 0.80,
        "scaled_batch_compiler_warning": 0.70,
        "manual_review_pass_minimum": 0.80,
        "json_parse_pass_target_before_scaling": 1.0,
    }


def test_recycling_requires_code_generation_rejection_with_clear_compiler_error():
    schema_failure = validate_example(
        category="code_gen",
        example={"prompt": "Say hi", "complexity": "simple"},
        compiler=fake_compiler,
    )
    explanation_failure = validate_example(
        category="explanation",
        example={"code": "BROKEN jac", "granularity": "module", "explanation": "Explains the module."},
        compiler=fake_compiler,
    )

    assert should_recycle_rejected_example(schema_failure) is False
    assert should_recycle_rejected_example(explanation_failure) is False


def test_calculate_pass_rates_and_threshold_warnings():
    results = [
        validate_example(
            category="code_gen",
            example={"prompt": "One", "code": "valid jac", "complexity": "simple"},
            compiler=fake_compiler,
            test_pass=True,
        ),
        validate_example(
            category="code_gen",
            example={"prompt": "Two", "code": "BROKEN jac", "complexity": "simple"},
            compiler=fake_compiler,
        ),
        validate_example(
            category="code_gen",
            example={"prompt": "Three", "code": "valid jac", "complexity": "simple"},
            compiler=fake_compiler,
            test_pass=False,
        ),
    ]

    pass_rates = calculate_pass_rates(results)
    warnings = evaluate_batch_thresholds(results)

    assert pass_rates["compiler_pass_rate"] == 2 / 3
    assert pass_rates["manual_review_pass_rate"] == 0.0
    assert "compiler pass rate 0.67 below scaled-batch warning threshold 0.70" in warnings


def test_validation_log_record_contains_required_fields_and_is_json_serializable():
    result = validate_example(
        category="code_gen",
        example={"prompt": "Say hi", "code": "valid jac", "complexity": "simple"},
        compiler=fake_compiler,
        test_pass=True,
    )

    record = build_validation_log_record(
        result,
        batch_id="20260507-code_gen-001",
        prompt_version="prompt-code_gen-v1",
        context_bundle_version="jac-context-v1",
        example_id="code_gen-20260507-001-0001",
        validator_version="validator-v1",
        dataset_version="jac-synth-v0.1.0",
    )

    for field in VALIDATION_LOG_REQUIRED_FIELDS:
        assert field in record
    assert record["final_disposition"] == "clean"
    assert record["compiler_result"][0]["field_name"] == "code"
    assert record["validator_version"] == "validator-v1"
    assert record["dataset_version"] == "jac-synth-v0.1.0"
    json.dumps(record)
