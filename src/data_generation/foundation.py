from __future__ import annotations

from collections.abc import Mapping
from typing import Any

ALLOWED_CATEGORIES = ("code_gen", "debug", "explanation", "conversion", "trajectory")
SCRIPTED_CATEGORIES = ("code_gen", "debug", "explanation", "conversion")
ALLOWED_COMPLEXITIES = ("simple", "medium", "hard")
ALLOWED_GENERATORS = (
    "openai-api",
    "cursor-jac-mcp",
    "codex-jac-mcp",
    "claude-code-jac-mcp",
)
ALLOWED_ERROR_TYPES = ("syntax", "type", "walker", "scope", "import", "semantic")
ALLOWED_GRANULARITIES = ("line", "block", "module")

REQUIRED_METADATA_FIELDS = (
    "id",
    "batch_id",
    "category",
    "complexity",
    "compiler_pass",
    "test_pass",
    "manually_reviewed",
    "generator",
    "generation_date",
    "source_prompt_version",
    "context_bundle_version",
    "validator_version",
    "dataset_version",
)

OPTIONAL_METADATA_FIELDS = (
    "error_type",
    "granularity",
    "trajectory_length_tokens",
    "dedup_hash",
    "reviewer",
    "review_status",
    "rejection_reason",
)

DATASET_STORAGE_PATHS = {
    "raw_output": {category: f"dataset/raw_output/{category}" for category in ALLOWED_CATEGORIES},
    "clean_dataset": {category: f"dataset/clean_dataset/{category}" for category in ALLOWED_CATEGORIES},
    "rejected": {category: f"dataset/rejected/{category}" for category in ALLOWED_CATEGORIES},
    "review": {category: f"dataset/review/{category}" for category in ALLOWED_CATEGORIES},
    "logs": {
        "generation": "dataset/logs/generation",
        "parsing": "dataset/logs/parsing",
        "compiler": "dataset/logs/compiler",
        "test": "dataset/logs/test",
        "retry": "dataset/logs/retry",
        "deduplication": "dataset/logs/deduplication",
    },
    "releases": "dataset/releases",
}

CONTEXT_BUNDLE_REQUIREMENTS = {
    "initial_version": "jac-context-v1",
    "required_sections": (
        "jac://docs/cheatsheet",
        "jac://guide/pitfalls",
        "jac://guide/patterns",
        "skills.md or equivalent Jac MCP guidance",
        "valid examples of walkers",
        "valid examples of nodes",
        "valid examples of edges",
        "valid examples of abilities",
        "valid examples of imports",
        "valid examples of type annotations",
        "standard library usage",
        "code organization patterns",
        "category output schema instructions",
    ),
}


def _ensure_category(category: str) -> None:
    if category not in ALLOWED_CATEGORIES:
        raise ValueError(f"Unsupported category: {category}")


def build_batch_id(date: str, category: str, sequence: int) -> str:
    _ensure_category(category)
    return f"{date}-{category}-{sequence:03d}"


def build_example_id(category: str, date: str, batch_sequence: int, example_sequence: int) -> str:
    _ensure_category(category)
    return f"{category}-{date}-{batch_sequence:03d}-{example_sequence:04d}"


def build_prompt_version(category: str, version: int) -> str:
    _ensure_category(category)
    return f"prompt-{category}-v{version}"


def build_context_bundle_version(version: int) -> str:
    return f"jac-context-v{version}"


def build_validator_version(version: int) -> str:
    return f"validator-v{version}"


def validate_metadata(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []

    for field in REQUIRED_METADATA_FIELDS:
        if field not in record:
            errors.append(f"missing required field: {field}")

    category = record.get("category")
    if category is not None and category not in ALLOWED_CATEGORIES:
        errors.append(f"invalid category: {category}")

    complexity = record.get("complexity")
    if complexity is not None and complexity not in ALLOWED_COMPLEXITIES:
        errors.append(f"invalid complexity: {complexity}")

    generator = record.get("generator")
    if generator is not None and generator not in ALLOWED_GENERATORS:
        errors.append(f"invalid generator: {generator}")

    error_type = record.get("error_type")
    if error_type is not None and error_type not in ALLOWED_ERROR_TYPES:
        errors.append(f"invalid error_type: {error_type}")

    granularity = record.get("granularity")
    if granularity is not None and granularity not in ALLOWED_GRANULARITIES:
        errors.append(f"invalid granularity: {granularity}")

    return errors
