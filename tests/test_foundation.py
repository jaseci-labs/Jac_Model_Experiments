import pytest

from data_generation.foundation import (
    ALLOWED_CATEGORIES,
    CONTEXT_BUNDLE_REQUIREMENTS,
    DATASET_STORAGE_PATHS,
    build_batch_id,
    build_example_id,
    build_prompt_version,
    validate_metadata,
)


def test_task1_categories_and_storage_paths_are_complete():
    assert ALLOWED_CATEGORIES == (
        "code_gen",
        "debug",
        "explanation",
        "conversion",
        "trajectory",
    )
    for area in ("raw_output", "clean_dataset", "rejected", "review"):
        for category in ALLOWED_CATEGORIES:
            assert DATASET_STORAGE_PATHS[area][category] == f"dataset/{area}/{category}"

    assert DATASET_STORAGE_PATHS["logs"]["generation"] == "dataset/logs/generation"
    assert DATASET_STORAGE_PATHS["releases"] == "dataset/releases"


def test_naming_helpers_match_task1_formats():
    assert build_batch_id("20260507", "code_gen", 1) == "20260507-code_gen-001"
    assert build_example_id("code_gen", "20260507", 1, 7) == "code_gen-20260507-001-0007"
    assert build_prompt_version("code_gen", 1) == "prompt-code_gen-v1"


def test_invalid_category_is_rejected_by_naming_helpers():
    with pytest.raises(ValueError, match="Unsupported category"):
        build_batch_id("20260507", "code-generation", 1)


def test_required_metadata_validation_accepts_clean_record():
    record = {
        "id": "code_gen-20260507-001-0007",
        "batch_id": "20260507-code_gen-001",
        "category": "code_gen",
        "complexity": "medium",
        "compiler_pass": True,
        "test_pass": True,
        "manually_reviewed": False,
        "generator": "openai-api",
        "generation_date": "2026-05-07",
        "source_prompt_version": "prompt-code_gen-v1",
        "context_bundle_version": "jac-context-v1",
        "validator_version": "validator-v1",
        "dataset_version": "jac-synth-v0.1.0",
    }

    assert validate_metadata(record) == []


def test_required_metadata_validation_reports_missing_and_invalid_fields():
    errors = validate_metadata({"category": "bad", "complexity": "huge", "generator": "manual"})

    assert "missing required field: id" in errors
    assert "invalid category: bad" in errors
    assert "invalid complexity: huge" in errors
    assert "invalid generator: manual" in errors


def test_context_bundle_requirements_include_task1_required_sources():
    assert CONTEXT_BUNDLE_REQUIREMENTS["initial_version"] == "jac-context-v1"
    required_sections = CONTEXT_BUNDLE_REQUIREMENTS["required_sections"]

    assert "jac://docs/cheatsheet" in required_sections
    assert "jac://guide/pitfalls" in required_sections
    assert "jac://guide/patterns" in required_sections
    assert "skills.md or equivalent Jac MCP guidance" in required_sections
