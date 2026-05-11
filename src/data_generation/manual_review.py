from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from data_generation.prompt_design import PROMPT_REVISION_LOG_REQUIRED_FIELDS
from data_generation.validation import PASS_RATE_THRESHOLDS, ExampleValidationResult, calculate_pass_rates


CATEGORY_REVIEW_CRITERIA = {
    "code_gen": (
        "prompt_clarity",
        "idiomatic_jac",
        "construct_diversity",
        "not_python_like",
    ),
    "debug": (
        "exactly_one_error",
        "realistic_error",
        "fixed_code_precise",
        "specific_explanation",
    ),
    "explanation": (
        "accurate_explanation",
        "specific_jac_semantics",
        "granularity_matches_request",
        "not_python_behavior",
    ),
    "conversion": (
        "preserves_behavior",
        "idiomatic_jac",
        "meaningful_conversion_notes",
        "not_mechanical_translation",
    ),
}


class ScaleDecisionStatus(StrEnum):
    SCALE_READY = "scale_ready"
    REVISE_PROMPT = "revise_prompt"
    PAUSED = "paused"


@dataclass(frozen=True)
class ReviewRecord:
    batch_id: str
    category: str
    example_id: str
    review_status: str
    reviewer: str
    criteria_results: dict[str, bool]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScaleDecision:
    batch_id: str
    category: str
    json_parse_pass_rate: float
    compiler_pass_rate: float
    manual_review_pass_rate: float
    decision: ScaleDecisionStatus
    reasons: list[str]
    prompt_revision_required: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decision"] = self.decision.value
        return payload


def decide_scale_up(
    *,
    batch_id: str,
    category: str,
    validation_results: list[ExampleValidationResult],
    review_records: list[ReviewRecord],
) -> ScaleDecision:
    pass_rates = calculate_pass_rates(validation_results)
    review_total = len(review_records)
    review_passes = sum(1 for record in review_records if record.review_status == "passed")
    manual_review_pass_rate = review_passes / review_total if review_total else 0.0
    blocking_review_issue = any(
        record.review_status != "passed" or not all(record.criteria_results.values())
        for record in review_records
    )

    reasons: list[str] = []
    if pass_rates["json_parse_pass_rate"] < PASS_RATE_THRESHOLDS["json_parse_pass_target_before_scaling"]:
        reasons.append("json pass rate below scale threshold")
    if pass_rates["compiler_pass_rate"] < PASS_RATE_THRESHOLDS["pilot_compiler_pass_target"]:
        reasons.append("compiler pass rate below pilot threshold")
    if manual_review_pass_rate < PASS_RATE_THRESHOLDS["manual_review_pass_minimum"]:
        reasons.append("manual review pass rate below threshold")
    if blocking_review_issue:
        reasons.append("blocking review issue")

    decision = ScaleDecisionStatus.SCALE_READY if not reasons else ScaleDecisionStatus.REVISE_PROMPT
    return ScaleDecision(
        batch_id=batch_id,
        category=category,
        json_parse_pass_rate=pass_rates["json_parse_pass_rate"],
        compiler_pass_rate=pass_rates["compiler_pass_rate"],
        manual_review_pass_rate=manual_review_pass_rate,
        decision=decision,
        reasons=reasons,
        prompt_revision_required=decision != ScaleDecisionStatus.SCALE_READY,
    )


def build_prompt_revision_record(
    *,
    prompt_version: str,
    previous_prompt_version: str,
    category: str,
    changed_fields: list[str],
    reason_for_change: str,
    batch_ids_affected: list[str],
    observed_pass_rate_before: float,
    observed_pass_rate_after: float | None,
    changed_at: str,
    notes: str,
) -> dict[str, Any]:
    record = {
        "prompt_version": prompt_version,
        "previous_prompt_version": previous_prompt_version,
        "category": category,
        "changed_fields": changed_fields,
        "reason_for_change": reason_for_change,
        "batch_ids_affected": batch_ids_affected,
        "observed_pass_rate_before": observed_pass_rate_before,
        "observed_pass_rate_after": observed_pass_rate_after,
        "changed_at": changed_at,
        "notes": notes,
    }
    missing = [field for field in PROMPT_REVISION_LOG_REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"missing prompt revision fields: {missing}")
    return record


def default_review_records(batch_id: str, category: str, example_ids: list[str]) -> list[ReviewRecord]:
    return [
        ReviewRecord(
            batch_id=batch_id,
            category=category,
            example_id=example_id,
            review_status="pending",
            reviewer="manual-reviewer",
            criteria_results={criterion: False for criterion in CATEGORY_REVIEW_CRITERIA[category]},
            notes="Pending manual review.",
        )
        for example_id in example_ids
    ]
