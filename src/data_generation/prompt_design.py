from __future__ import annotations

from copy import deepcopy
from typing import Any

from data_generation.foundation import (
    ALLOWED_COMPLEXITIES,
    ALLOWED_ERROR_TYPES,
    ALLOWED_GRANULARITIES,
    SCRIPTED_CATEGORIES,
    build_prompt_version,
)

DEFAULT_CONTEXT_BUNDLE_VERSION = "jac-context-v1"
DEFAULT_PROMPT_VERSION = 1

PROMPT_VERSION_REGISTRY = {
    "code_gen": {
        1: {"status": "pilot-only"},
        2: {"status": "deprecated"},
        3: {"status": "deprecated"},
        4: {"status": "scale-approved"},
    },
    "debug": {
        1: {"status": "pilot-only"},
        2: {"status": "deprecated"},
        3: {"status": "deprecated"},
        4: {"status": "scale-approved"},
    },
    "explanation": {
        1: {"status": "pilot-only"},
        2: {"status": "deprecated"},
        3: {"status": "scale-approved"},
    },
    "conversion": {
        1: {"status": "pilot-only"},
        2: {"status": "deprecated"},
        3: {"status": "deprecated"},
        4: {"status": "scale-approved"},
    },
}

TASK4_COMPILER_FAILURE_CONSTRAINTS = """Observed compiler-failure guardrails for this prompt revision:
- Use `def`, not `can`, for function-style declarations.
- Never use `import:py`; prefer examples that do not require Python imports.
- Use `root()` instead of bare `root`.
- Use simple built-in edge connection syntax such as `++>` unless the context bundle shows a typed edge syntax you can reproduce exactly.
- Keep each Jac code snippet small enough for fast compiler validation.
"""

TASK4_RETRY_CONSTRAINTS = """Additional retry guardrails from prompt v2 validation:
- Use `not value`, not Python-style `!value`, for boolean negation.
- For debugging examples, broken_code must produce a compiler error, not only a warning.
- Avoid hard examples that require nested list mutation, boolean-list indexing, or long shortest-path implementations.
"""

TASK4_DEBUG_FAILURE_CONSTRAINTS = """Debugging-specific compiler-failure guardrails:
- The broken_code field must fail `jac check`; do not use warning-only cases such as missing imports that still pass.
- Prefer obvious syntax, type, and undefined-name errors that the compiler reports as errors.
- The fixed_code field must be the same small program with only the intended error corrected.
- Avoid import-based broken examples unless the missing import is known to be a hard compiler error.
"""

TASK4_CODE_GEN_SCALE_CONSTRAINTS = """Code-generation scale-failure guardrails:
- Annotate every local variable before arithmetic or boolean use; avoid relying on inferred `Unknown` values.
- Use `True` and `False`, never lowercase `true` or `false`.
- Prefer `+=`, `-=`, and `*=` updates over `x = x + y` in loops and counters.
- Avoid using `+` to concatenate lists or strings unless the context bundle shows that exact pattern.
- Avoid walker examples that read state from a spawned walker result; prefer pure functions, nodes, edges, and direct walker visits shown in context.
- Do not use repeated walker ability names in the same walker.
- Avoid Python-style helper methods such as `.keys()`, `.append()`, or free-form list/dict idioms unless shown in the Jac context bundle.
"""

SHARED_SYSTEM_PROMPT_TEMPLATE = """You generate synthetic supervised fine-tuning examples for the Jac programming language.

Jac is its own programming language. Do not treat Jac as Python, JavaScript, or pseudocode. Prefer idiomatic Jac constructs from the provided context, including walkers, nodes, edges, abilities, imports, type annotations, and graph-oriented patterns when they fit the task.

All Jac code fields in your response will be validated with the Jac compiler. Code that does not compile will be rejected. For debugging examples, broken_code must fail for exactly the intended reason and fixed_code must compile.

Return strict JSON only. Do not include markdown fences, comments outside JSON, prose before JSON, or prose after JSON. The top-level value must be a JSON object with exactly one key named `examples`. The `examples` value must be an array, and every object in that array must match the category schema exactly.

Use the provided Jac context bundle as the source of truth for syntax and idioms. If a requested example cannot be generated confidently from the context, choose a simpler valid Jac example rather than guessing syntax.

[BEGIN JAC CONTEXT BUNDLE: {context_bundle_version}]
{context_bundle}
[END JAC CONTEXT BUNDLE]
"""

PILOT_BATCH_SETTINGS = {
    "code_gen": {
        "requested_count": 5,
        "diversity": "Mix simple, medium, and hard tasks; cover at least two Jac construct families.",
    },
    "debug": {
        "requested_count": 5,
        "diversity": "Use exactly one error type per example and avoid duplicate error patterns.",
    },
    "explanation": {
        "requested_count": 5,
        "diversity": "Include line, block, and module explanation granularities.",
    },
    "conversion": {
        "requested_count": 5,
        "diversity": "Include at least one function conversion and one class or data-structure conversion.",
    },
}

SCALE_UP_BATCH_SETTINGS = {
    "min_examples_per_call": 20,
    "max_examples_per_call": 50,
    "reduce_batch_before_removing_context": True,
}

JSON_PARSER_EXPECTATIONS = {
    "top_level": "array",
    "allow_markdown_fences": False,
    "allow_extra_prose": False,
    "allow_additional_properties": False,
    "required_strings_must_be_non_empty": True,
    "controlled_fields_must_match_enums": True,
}

PROMPT_REVISION_LOG_REQUIRED_FIELDS = (
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


def _string_schema() -> dict[str, Any]:
    return {"type": "string", "minLength": 1}


CATEGORY_SCHEMAS = {
    "code_gen": {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["prompt", "code", "complexity"],
            "additionalProperties": False,
            "properties": {
                "prompt": _string_schema(),
                "code": _string_schema(),
                "complexity": {"type": "string", "enum": list(ALLOWED_COMPLEXITIES)},
            },
        },
    },
    "debug": {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["broken_code", "error_type", "error_message", "fixed_code", "fix_explanation"],
            "additionalProperties": False,
            "properties": {
                "broken_code": _string_schema(),
                "error_type": {"type": "string", "enum": list(ALLOWED_ERROR_TYPES)},
                "error_message": _string_schema(),
                "fixed_code": _string_schema(),
                "fix_explanation": _string_schema(),
            },
        },
    },
    "explanation": {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["code", "granularity", "explanation"],
            "additionalProperties": False,
            "properties": {
                "code": _string_schema(),
                "granularity": {"type": "string", "enum": list(ALLOWED_GRANULARITIES)},
                "explanation": _string_schema(),
            },
        },
    },
    "conversion": {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["python_code", "jac_code", "conversion_notes"],
            "additionalProperties": False,
            "properties": {
                "python_code": _string_schema(),
                "jac_code": _string_schema(),
                "conversion_notes": _string_schema(),
            },
        },
    },
}

CATEGORY_PROMPT_TEMPLATES = {
    "code_gen": """Generate {requested_count} Jac code generation examples.

Return a JSON object with exactly one key named `examples`. Each item in `examples` must contain exactly: prompt, code, complexity.

Requirements:
- The prompt must be a natural language task with one reasonable implementation.
- The code must be complete Jac code expected to compile.
- complexity must be one of: simple, medium, hard.
- Use this complexity distribution: {complexity_distribution}.
- Cover these Jac constructs when possible: {required_constructs}.
- Vary these domains: {domains}.
- Do not write Python-like Jac.
- Use context bundle {context_bundle_version}.
- This prompt version is {prompt_version}.
""",
    "debug": """Generate {requested_count} Jac debugging examples.

Return a JSON object with exactly one key named `examples`. Each item in `examples` must contain exactly: broken_code, error_type, error_message, fixed_code, fix_explanation.

Requirements:
- broken_code must contain exactly one intended error.
- error_type must be one of: syntax, type, walker, scope, import, semantic.
- broken_code must fail for the stated error.
- fixed_code must be complete Jac code expected to compile.
- fix_explanation must state what changed and why.
- Source valid examples policy: {source_valid_examples_policy}.
- Use these error types: {error_types}.
- Use context bundle {context_bundle_version}.
- This prompt version is {prompt_version}.
""",
    "explanation": """Generate {requested_count} Jac explanation examples.

Return a JSON object with exactly one key named `examples`. Each item in `examples` must contain exactly: code, granularity, explanation.

Requirements:
- code must be valid Jac expected to compile.
- granularity must be one of: line, block, module.
- explanation must describe Jac behavior accurately and specifically.
- Use this granularity distribution: {granularity_distribution}.
- Cover these Jac constructs when possible: {required_constructs}.
- Do not explain the code as if it were Python.
- Use context bundle {context_bundle_version}.
- This prompt version is {prompt_version}.
""",
    "conversion": """Generate {requested_count} Python-to-Jac conversion examples.

Return a JSON object with exactly one key named `examples`. Each item in `examples` must contain exactly: python_code, jac_code, conversion_notes.

Requirements:
- python_code must be clear, self-contained Python.
- jac_code must preserve behavior and be idiomatic Jac expected to compile.
- conversion_notes must explain meaningful design changes.
- Python source styles: {python_source_styles}.
- Complexity constraints: {complexity_constraints}.
- Required conversion patterns: {required_conversion_patterns}.
- Do not mechanically translate Python syntax into Jac.
- Use context bundle {context_bundle_version}.
- This prompt version is {prompt_version}.
""",
}

DEFAULT_TEMPLATE_VALUES = {
    "code_gen": {
        "complexity_distribution": "40% simple, 40% medium, 20% hard",
        "required_constructs": "walkers, nodes, edges, abilities, imports, type annotations, common control flow",
        "domains": "graph algorithms, data processing, web, utilities",
    },
    "debug": {
        "error_types": "syntax, type, walker, scope, import, semantic",
        "source_valid_examples_policy": "Prefer starting from compiler-verified valid Jac and inject one realistic error.",
    },
    "explanation": {
        "granularity_distribution": "include line, block, and module examples in pilot batches",
        "required_constructs": "walkers, nodes, edges, abilities, imports, type annotations, common control flow",
    },
    "conversion": {
        "python_source_styles": "functions, classes, data structures, graph-like algorithms, utilities",
        "complexity_constraints": "mix simple, medium, and hard examples when requested",
        "required_conversion_patterns": "function-to-ability, class-to-node, graph pattern conversion, walker/traversal rewrites",
    },
}


def schema_for_category(category: str) -> dict[str, Any]:
    if category not in CATEGORY_SCHEMAS:
        raise ValueError(f"Unsupported scripted category: {category}")
    return deepcopy(CATEGORY_SCHEMAS[category])


def latest_scale_prompt_version(category: str) -> int:
    if category not in PROMPT_VERSION_REGISTRY:
        raise ValueError(f"Unsupported scripted category: {category}")
    approved = [
        version
        for version, metadata in PROMPT_VERSION_REGISTRY[category].items()
        if metadata["status"] == "scale-approved"
    ]
    if not approved:
        raise ValueError(f"No scale-approved prompt version for {category}")
    return max(approved)


def _ensure_scale_approved_prompt(category: str, prompt_version_number: int) -> None:
    status = PROMPT_VERSION_REGISTRY.get(category, {}).get(prompt_version_number, {}).get("status")
    if status != "scale-approved":
        raise ValueError(f"Prompt {build_prompt_version(category, prompt_version_number)} is not scale-approved")


def build_prompt_request(
    *,
    category: str,
    context_bundle: str,
    requested_count: int | None = None,
    context_bundle_version: str = DEFAULT_CONTEXT_BUNDLE_VERSION,
    prompt_version_number: int | None = None,
    template_overrides: dict[str, str] | None = None,
    scale_mode: bool = False,
    complexity_target: str | None = None,
) -> dict[str, Any]:
    if category not in SCRIPTED_CATEGORIES:
        raise ValueError(f"Unsupported scripted category: {category}")

    if scale_mode and prompt_version_number is None:
        prompt_version_number = latest_scale_prompt_version(category)
    prompt_version_number = prompt_version_number or DEFAULT_PROMPT_VERSION
    if scale_mode:
        _ensure_scale_approved_prompt(category, prompt_version_number)

    prompt_version = build_prompt_version(category, prompt_version_number)
    values = {
        **DEFAULT_TEMPLATE_VALUES[category],
        "requested_count": requested_count or PILOT_BATCH_SETTINGS[category]["requested_count"],
        "context_bundle_version": context_bundle_version,
        "prompt_version": prompt_version,
    }
    if template_overrides:
        values.update(template_overrides)

    user_prompt = CATEGORY_PROMPT_TEMPLATES[category].format(**values)
    if complexity_target:
        if complexity_target not in ALLOWED_COMPLEXITIES:
            raise ValueError(f"Unsupported complexity target: {complexity_target}")
        user_prompt = f"{user_prompt}\nTarget complexity for every example: {complexity_target}."
    if prompt_version_number >= 2:
        user_prompt = f"{user_prompt}\n{TASK4_COMPILER_FAILURE_CONSTRAINTS}"
    if prompt_version_number >= 3:
        user_prompt = f"{user_prompt}\n{TASK4_RETRY_CONSTRAINTS}"
    if category == "debug" and prompt_version_number >= 4:
        user_prompt = f"{user_prompt}\n{TASK4_DEBUG_FAILURE_CONSTRAINTS}"
    if category == "code_gen" and prompt_version_number >= 4:
        user_prompt = f"{user_prompt}\n{TASK4_CODE_GEN_SCALE_CONSTRAINTS}"

    return {
        "category": category,
        "prompt_version": prompt_version,
        "context_bundle_version": context_bundle_version,
        "system_prompt": SHARED_SYSTEM_PROMPT_TEMPLATE.format(
            context_bundle_version=context_bundle_version,
            context_bundle=context_bundle,
        ),
        "user_prompt": user_prompt,
        "response_schema": schema_for_category(category),
        "parser_expectations": JSON_PARSER_EXPECTATIONS.copy(),
    }
