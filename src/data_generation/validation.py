from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable

from data_generation.prompt_design import CATEGORY_SCHEMAS


class ValidationStage(StrEnum):
    JSON = "json"
    SCHEMA = "schema"
    COMPILER = "compiler"
    TEST = "test"


class Disposition(StrEnum):
    CLEAN = "clean"
    REJECTED = "rejected"
    REVIEW = "review"
    DISCARDED = "discarded"


@dataclass(frozen=True)
class CompilerResult:
    passed: bool
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""


CompilerRunner = Callable[[str], CompilerResult]


@dataclass(frozen=True)
class FieldValidationResult:
    field_name: str
    expected_to_compile: bool
    compiler_result: CompilerResult


@dataclass(frozen=True)
class ExampleValidationResult:
    category: str
    example: Mapping[str, Any] | None
    json_schema_pass: bool
    compiler_pass: bool | None
    test_pass: bool | None
    disposition: Disposition
    rejection_reason: str | None
    retry_count: int
    compiler_results: tuple[FieldValidationResult, ...]


RETRY_LIMITS = {
    "malformed_json_immediate_retry": 1,
    "low_compile_pass_rate_prompt_revisions": 2,
    "trajectory_consecutive_compiler_failures": 3,
}

PASS_RATE_THRESHOLDS = {
    "pilot_compiler_pass_target": 0.80,
    "scaled_batch_compiler_warning": 0.70,
    "manual_review_pass_minimum": 0.80,
    "json_parse_pass_target_before_scaling": 1.0,
}

VALIDATION_LOG_REQUIRED_FIELDS = (
    "batch_id",
    "prompt_version",
    "context_bundle_version",
    "category",
    "example_id",
    "json_schema_result",
    "compiler_result",
    "test_result",
    "rejection_reason",
    "retry_count",
    "final_disposition",
)

_COMPILER_FIELD_POLICIES = {
    "code_gen": (("code", True),),
    "debug": (("broken_code", False), ("fixed_code", True)),
    "explanation": (("code", True),),
    "conversion": (("jac_code", True),),
    "trajectory": (),
}


def parse_json_batch(raw_response: str) -> tuple[list[dict[str, Any]] | None, list[str]]:
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return None, [f"malformed JSON: {exc.msg}"]

    if not isinstance(parsed, list):
        return None, ["top-level JSON value must be an array"]

    errors = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            errors.append(f"item {index} must be an object")

    if errors:
        return None, errors
    return parsed, []


def validate_batch_schema(category: str, examples: object) -> list[str]:
    if category not in CATEGORY_SCHEMAS:
        return [f"unsupported category: {category}"]

    schema = CATEGORY_SCHEMAS[category]
    errors = _validate_value(examples, schema, "batch")
    return [_normalize_schema_error(error) for error in errors]


def compile_fields_for_example(
    category: str,
    example: Mapping[str, Any],
    compiler: CompilerRunner,
) -> tuple[bool, tuple[FieldValidationResult, ...], str | None]:
    if category not in _COMPILER_FIELD_POLICIES:
        raise ValueError(f"Unsupported category: {category}")

    results = []
    for field_name, expected_to_compile in _COMPILER_FIELD_POLICIES[category]:
        code = example[field_name]
        compiler_result = compiler(code)
        field_result = FieldValidationResult(
            field_name=field_name,
            expected_to_compile=expected_to_compile,
            compiler_result=compiler_result,
        )
        results.append(field_result)

        if expected_to_compile and not compiler_result.passed:
            return False, tuple(results), f"{field_name} failed compiler validation"
        if not expected_to_compile and compiler_result.passed:
            return False, tuple(results), f"{field_name} compiled but must fail"

    return True, tuple(results), None


def validate_example(
    *,
    category: str,
    example: Mapping[str, Any],
    compiler: CompilerRunner,
    test_pass: bool | None = None,
    retry_count: int = 0,
) -> ExampleValidationResult:
    schema_errors = validate_batch_schema(category, [example])
    if schema_errors:
        return ExampleValidationResult(
            category=category,
            example=example,
            json_schema_pass=False,
            compiler_pass=None,
            test_pass=test_pass,
            disposition=Disposition.REJECTED,
            rejection_reason="; ".join(schema_errors),
            retry_count=retry_count,
            compiler_results=(),
        )

    compiler_pass, compiler_results, rejection_reason = compile_fields_for_example(category, example, compiler)
    if not compiler_pass:
        disposition = (
            Disposition.DISCARDED
            if category == "debug" and rejection_reason == "fixed_code failed compiler validation"
            else Disposition.REJECTED
        )
        return ExampleValidationResult(
            category=category,
            example=example,
            json_schema_pass=True,
            compiler_pass=False,
            test_pass=test_pass,
            disposition=disposition,
            rejection_reason=rejection_reason,
            retry_count=retry_count,
            compiler_results=compiler_results,
        )

    if test_pass is False:
        return ExampleValidationResult(
            category=category,
            example=example,
            json_schema_pass=True,
            compiler_pass=True,
            test_pass=test_pass,
            disposition=Disposition.REVIEW,
            rejection_reason="behavior test failed; manual review required",
            retry_count=retry_count,
            compiler_results=compiler_results,
        )

    return ExampleValidationResult(
        category=category,
        example=example,
        json_schema_pass=True,
        compiler_pass=True,
        test_pass=test_pass,
        disposition=Disposition.CLEAN,
        rejection_reason=None,
        retry_count=retry_count,
        compiler_results=compiler_results,
    )


def validate_batch(
    *,
    category: str,
    examples: Sequence[Mapping[str, Any]],
    compiler: CompilerRunner,
    test_results: Sequence[bool | None] | None = None,
    retry_count: int = 0,
) -> list[ExampleValidationResult]:
    test_results = test_results or [None] * len(examples)
    return [
        validate_example(
            category=category,
            example=example,
            compiler=compiler,
            test_pass=test_results[index],
            retry_count=retry_count,
        )
        for index, example in enumerate(examples)
    ]


def build_validation_log_record(
    result: ExampleValidationResult,
    *,
    batch_id: str,
    prompt_version: str,
    context_bundle_version: str,
    example_id: str,
    validator_version: str | None = None,
    dataset_version: str | None = None,
) -> dict[str, Any]:
    record = {
        "batch_id": batch_id,
        "prompt_version": prompt_version,
        "context_bundle_version": context_bundle_version,
        "category": result.category,
        "example_id": example_id,
        "json_schema_result": result.json_schema_pass,
        "compiler_result": [
            {
                "field_name": field_result.field_name,
                "expected_to_compile": field_result.expected_to_compile,
                "passed": field_result.compiler_result.passed,
                "error_message": field_result.compiler_result.error_message,
            }
            for field_result in result.compiler_results
        ],
        "test_result": result.test_pass,
        "rejection_reason": result.rejection_reason,
        "retry_count": result.retry_count,
        "final_disposition": result.disposition.value,
    }
    if validator_version is not None:
        record["validator_version"] = validator_version
    if dataset_version is not None:
        record["dataset_version"] = dataset_version
    return record


def should_recycle_rejected_example(result: ExampleValidationResult) -> bool:
    if result.category != "code_gen" or result.disposition != Disposition.REJECTED:
        return False
    if not result.example or not str(result.example.get("code", "")).strip():
        return False

    for field_result in result.compiler_results:
        compiler_result = field_result.compiler_result
        if not compiler_result.passed and any(
            (compiler_result.error_message, compiler_result.stderr, compiler_result.stdout)
        ):
            return True
    return False


def calculate_pass_rates(results: Sequence[ExampleValidationResult]) -> dict[str, float]:
    total = len(results)
    if total == 0:
        return {
            "json_parse_pass_rate": 0.0,
            "compiler_pass_rate": 0.0,
            "manual_review_pass_rate": 0.0,
        }

    json_schema_passes = sum(1 for result in results if result.json_schema_pass)
    compiler_results = [result for result in results if result.compiler_pass is not None]
    compiler_passes = sum(1 for result in compiler_results if result.compiler_pass)
    review_results = [result for result in results if result.disposition == Disposition.REVIEW]
    manual_review_passes = sum(1 for result in review_results if result.test_pass is True)

    return {
        "json_parse_pass_rate": json_schema_passes / total,
        "compiler_pass_rate": compiler_passes / len(compiler_results) if compiler_results else 0.0,
        "manual_review_pass_rate": manual_review_passes / len(review_results) if review_results else 0.0,
    }


def evaluate_batch_thresholds(results: Sequence[ExampleValidationResult]) -> list[str]:
    pass_rates = calculate_pass_rates(results)
    warnings = []
    compiler_pass_rate = pass_rates["compiler_pass_rate"]
    compiler_warning_threshold = PASS_RATE_THRESHOLDS["scaled_batch_compiler_warning"]
    if compiler_pass_rate < compiler_warning_threshold:
        warnings.append(
            f"compiler pass rate {compiler_pass_rate:.2f} "
            f"below scaled-batch warning threshold {compiler_warning_threshold:.2f}"
        )
    return warnings


def _validate_value(value: object, schema: Mapping[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")

    if expected_type == "array":
        if not isinstance(value, list):
            return [f"{path} must be an array"]
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                errors.extend(_validate_value(item, item_schema, f"item {index}"))
        return errors

    if expected_type == "object":
        if not isinstance(value, dict):
            return [f"{path} must be an object"]

        required = schema.get("required", [])
        for field_name in required:
            if field_name not in value:
                errors.append(f"{path} missing required field: {field_name}")

        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for field_name in value:
                if field_name not in properties:
                    errors.append(f"{path} has unexpected field: {field_name}")

        for field_name, field_schema in properties.items():
            if field_name in value and isinstance(field_schema, Mapping):
                errors.extend(_validate_value(value[field_name], field_schema, f"{path} field {field_name}"))
        return errors

    if expected_type == "string":
        if not isinstance(value, str):
            return [f"{path} must be a string"]
        if schema.get("minLength", 0) >= 1 and not value:
            errors.append(f"{path} must be a non-empty string")
        enum_values = schema.get("enum")
        if enum_values is not None and value not in enum_values:
            errors.append(f"{path} must be one of: {', '.join(enum_values)}")

    return errors


def _normalize_schema_error(error: str) -> str:
    return error.replace("batch ", "")
