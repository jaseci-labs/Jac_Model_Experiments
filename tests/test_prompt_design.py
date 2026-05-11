import json

from data_generation.foundation import SCRIPTED_CATEGORIES
from data_generation.prompt_design import (
    CATEGORY_PROMPT_TEMPLATES,
    CATEGORY_SCHEMAS,
    JSON_PARSER_EXPECTATIONS,
    PILOT_BATCH_SETTINGS,
    PROMPT_REVISION_LOG_REQUIRED_FIELDS,
    SCALE_UP_BATCH_SETTINGS,
    build_prompt_request,
    schema_for_category,
)


def test_all_scripted_categories_have_templates_schemas_and_pilot_settings():
    for category in SCRIPTED_CATEGORIES:
        assert category in CATEGORY_PROMPT_TEMPLATES
        assert category in CATEGORY_SCHEMAS
        assert PILOT_BATCH_SETTINGS[category]["requested_count"] == 5


def test_code_generation_schema_is_strict():
    schema = schema_for_category("code_gen")

    assert schema["type"] == "array"
    item_schema = schema["items"]
    assert item_schema["required"] == ["prompt", "code", "complexity"]
    assert item_schema["additionalProperties"] is False
    assert item_schema["properties"]["complexity"]["enum"] == ["simple", "medium", "hard"]


def test_debug_schema_uses_task1_error_type_enum():
    schema = schema_for_category("debug")

    assert schema["items"]["properties"]["error_type"]["enum"] == [
        "syntax",
        "type",
        "walker",
        "scope",
        "import",
        "semantic",
    ]


def test_explanation_schema_uses_task1_granularity_enum():
    schema = schema_for_category("explanation")

    assert schema["items"]["properties"]["granularity"]["enum"] == ["line", "block", "module"]


def test_conversion_schema_requires_python_jac_and_notes():
    schema = schema_for_category("conversion")

    assert schema["items"]["required"] == ["python_code", "jac_code", "conversion_notes"]


def test_prompt_request_includes_context_bundle_and_strict_json_schema():
    request = build_prompt_request(
        category="code_gen",
        context_bundle="Jac context goes here",
        requested_count=5,
    )

    assert request["prompt_version"] == "prompt-code_gen-v1"
    assert request["context_bundle_version"] == "jac-context-v1"
    assert "Jac is its own programming language" in request["system_prompt"]
    assert "[BEGIN JAC CONTEXT BUNDLE: jac-context-v1]" in request["system_prompt"]
    assert "Jac context goes here" in request["system_prompt"]
    assert "Return a JSON array" in request["user_prompt"]
    assert request["response_schema"] == CATEGORY_SCHEMAS["code_gen"]
    json.dumps(request["response_schema"])


def test_category_prompts_include_task4_compiler_failure_constraints():
    for category in SCRIPTED_CATEGORIES:
        request = build_prompt_request(
            category=category,
            context_bundle="Jac context",
            requested_count=5,
            prompt_version_number=2,
        )
        user_prompt = request["user_prompt"]

        assert "Use `def`, not `can`, for function-style declarations." in user_prompt
        assert "Never use `import:py`" in user_prompt
        assert "Use `root()` instead of bare `root`" in user_prompt
        assert "Keep each Jac code snippet small enough for fast compiler validation." in user_prompt


def test_parser_expectations_reject_markdown_and_extra_prose():
    assert JSON_PARSER_EXPECTATIONS["top_level"] == "array"
    assert JSON_PARSER_EXPECTATIONS["allow_markdown_fences"] is False
    assert JSON_PARSER_EXPECTATIONS["allow_extra_prose"] is False
    assert JSON_PARSER_EXPECTATIONS["allow_additional_properties"] is False


def test_scale_up_settings_match_task2():
    assert SCALE_UP_BATCH_SETTINGS["min_examples_per_call"] == 20
    assert SCALE_UP_BATCH_SETTINGS["max_examples_per_call"] == 50
    assert SCALE_UP_BATCH_SETTINGS["reduce_batch_before_removing_context"] is True


def test_revision_log_policy_has_required_fields():
    assert PROMPT_REVISION_LOG_REQUIRED_FIELDS == (
        "prompt_version",
        "previous_prompt_version",
        "category",
        "changed_fields",
        "reason_for_change",
        "batch_ids_affected",
        "observed_pass_rate_before",
        "observed_pass_rate_after",
        "changed_at",
        "notes",
    )
