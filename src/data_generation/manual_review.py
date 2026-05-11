from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
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
    "trajectory": (
        "task_solved",
        "logical_mcp_tool_use",
        "compiler_recovery_present",
        "final_code_validated",
        "idiomatic_jac",
        "no_private_context",
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
    prompt_revision_count: int = 0,
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

    if reasons and prompt_revision_count >= 2:
        reasons.append("prompt revision retry limit reached")
        decision = ScaleDecisionStatus.PAUSED
    else:
        decision = ScaleDecisionStatus.SCALE_READY if not reasons else ScaleDecisionStatus.REVISE_PROMPT
    return ScaleDecision(
        batch_id=batch_id,
        category=category,
        json_parse_pass_rate=pass_rates["json_parse_pass_rate"],
        compiler_pass_rate=pass_rates["compiler_pass_rate"],
        manual_review_pass_rate=manual_review_pass_rate,
        decision=decision,
        reasons=reasons,
        prompt_revision_required=decision == ScaleDecisionStatus.REVISE_PROMPT,
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


VALID_REVIEW_STATUSES = ("pending", "passed", "failed", "waived")


def list_review_records(workspace_root: str | Path = ".", *, status: str | None = None) -> list[dict[str, Any]]:
    records = []
    for path, record, _index in _iter_review_records(Path(workspace_root)):
        if status and record.get("review_status") != status:
            continue
        records.append({**record, "_review_path": str(path)})
    return sorted(records, key=lambda record: str(record.get("example_id", "")))


def mark_review_record(
    workspace_root: str | Path = ".",
    *,
    example_id: str,
    status: str,
    reviewer: str,
    criteria_results: dict[str, bool],
    notes: str,
) -> dict[str, Any]:
    if status not in VALID_REVIEW_STATUSES:
        raise ValueError(f"invalid review status: {status}")
    if status in {"failed", "waived"} and not notes.strip():
        raise ValueError("notes are required for failed or waived reviews")

    root = Path(workspace_root)
    for path, record, index in _iter_review_records(root):
        if record.get("example_id") != example_id:
            continue
        category = str(record.get("category"))
        _validate_criteria(category, criteria_results)
        payload = _read_review_payload(path)
        updated = {
            **record,
            "review_status": status,
            "reviewer": reviewer,
            "criteria_results": criteria_results,
            "notes": notes,
        }
        payload[index] = updated
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return updated
    raise ValueError(f"review record not found: {example_id}")


def validate_review_files(workspace_root: str | Path = ".") -> dict[str, Any]:
    errors = []
    for path, record, _index in _iter_review_records(Path(workspace_root)):
        category = str(record.get("category", ""))
        status = record.get("review_status")
        if status not in VALID_REVIEW_STATUSES:
            errors.append({"path": str(path), "example_id": record.get("example_id"), "reason": f"invalid status: {status}"})
        try:
            _validate_criteria(category, record.get("criteria_results", {}))
        except ValueError as error:
            errors.append({"path": str(path), "example_id": record.get("example_id"), "reason": str(error)})
        if status in {"failed", "waived"} and not str(record.get("notes", "")).strip():
            errors.append({"path": str(path), "example_id": record.get("example_id"), "reason": "notes are required"})
    return {"status": "blocked" if errors else "complete", "errors": errors}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage Jac synthetic data manual reviews.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--status", choices=VALID_REVIEW_STATUSES, default=None)

    mark_parser = subparsers.add_parser("mark")
    mark_parser.add_argument("--id", required=True)
    mark_parser.add_argument("--status", choices=VALID_REVIEW_STATUSES, required=True)
    mark_parser.add_argument("--reviewer", required=True)
    mark_parser.add_argument("--criteria", action="append", default=[])
    mark_parser.add_argument("--notes", default="")

    subparsers.add_parser("validate")

    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps(list_review_records(".", status=args.status), indent=2, sort_keys=True))
        return 0
    if args.command == "mark":
        print(
            json.dumps(
                mark_review_record(
                    ".",
                    example_id=args.id,
                    status=args.status,
                    reviewer=args.reviewer,
                    criteria_results=_parse_criteria_args(args.criteria),
                    notes=args.notes,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate":
        print(json.dumps(validate_review_files("."), indent=2, sort_keys=True))
        return 0
    return 2


def _iter_review_records(root: Path) -> Any:
    for path in sorted((root / "dataset/review").glob("**/*.json")):
        payload = _read_review_payload(path)
        for index, record in enumerate(payload):
            if isinstance(record, dict) and "example_id" in record:
                yield path, record, index


def _read_review_payload(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("records", [])
    if not isinstance(payload, list):
        return []
    return payload


def _validate_criteria(category: str, criteria_results: dict[str, bool]) -> None:
    expected = set(CATEGORY_REVIEW_CRITERIA.get(category, ()))
    actual = set(criteria_results)
    if actual != expected:
        raise ValueError(f"invalid review criteria for {category}: {sorted(actual ^ expected)}")
    if not all(isinstance(value, bool) for value in criteria_results.values()):
        raise ValueError("criteria values must be booleans")


def _parse_criteria_args(values: list[str]) -> dict[str, bool]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid criteria argument: {value}")
        key, raw = value.split("=", 1)
        parsed[key] = raw.lower() == "true"
    return parsed


if __name__ == "__main__":
    sys.exit(main())
