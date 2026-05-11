from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import random
import re
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from data_generation.artifacts import ensure_dataset_tree, write_json, write_jsonl
from data_generation.foundation import (
    ALLOWED_CATEGORIES,
    DATASET_STORAGE_PATHS,
    SCRIPTED_CATEGORIES,
    validate_metadata,
)
from data_generation.manual_review import CATEGORY_REVIEW_CRITERIA
from data_generation.prompt_design import CATEGORY_SCHEMAS
from data_generation.validation import PASS_RATE_THRESHOLDS, VALIDATION_LOG_REQUIRED_FIELDS, _COMPILER_FIELD_POLICIES

RELEASE_TOTAL_RANGE = (10_000, 15_000)
CATEGORY_TARGET_RANGES = {
    "code_gen": (3_000, 5_000),
    "debug": (2_000, 3_000),
    "explanation": (1_000, 2_000),
    "conversion": (1_000, 2_000),
    "trajectory": (2_000, 3_000),
}
TARGET_HARD_RATIO = 0.20
NEAR_DUPLICATE_THRESHOLD = 0.90


def audit_prerequisites(workspace_root: str | Path = ".", *, trajectory_batch_id: str | None = None) -> dict[str, Any]:
    root = Path(workspace_root)
    tasks = {
        "task1": _audit_task1_storage(root),
        "task2": _audit_task2_prompts(),
        "task3": _audit_task3_validation(),
        "task4": _audit_task4_artifacts(root),
        "task5": _audit_task5_artifacts(root, trajectory_batch_id=trajectory_batch_id),
    }
    return {"overall_status": _rollup_status(task["status"] for task in tasks.values()), "tasks": tasks}


def load_clean_candidates(workspace_root: str | Path = ".") -> dict[str, list[dict[str, Any]]]:
    root = Path(workspace_root)
    candidates = {category: [] for category in ALLOWED_CATEGORIES}
    for category in ALLOWED_CATEGORIES:
        clean_dir = root / "dataset" / "clean_dataset" / category
        for path in sorted(clean_dir.glob("*.jsonl")):
            for line_number, line in enumerate(path.read_text().splitlines(), start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record["_source_path"] = str(path)
                record["_source_line"] = line_number
                candidates.setdefault(category, []).append(record)
    return candidates


def audit_candidates(workspace_root: str | Path, candidates: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    root = Path(workspace_root)
    failures: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    validation_index = _validation_log_index(root)
    review_index = _review_index(root)

    for category, records in candidates.items():
        for record in records:
            record_id = str(record.get("id", "<missing id>"))
            for error in validate_metadata(record):
                failures.append({"id": record_id, "category": category, "reason": error})

            if record.get("compiler_pass") is False:
                failures.append({"id": record_id, "category": category, "reason": "compiler_pass is false"})

            if record.get("test_pass") is False and not _has_review_exception(record, review_index):
                failures.append({"id": record_id, "category": category, "reason": "test_pass is false without review exception"})
            elif record.get("test_pass") is None and not _has_review_exception(record, review_index):
                warnings.append({"id": record_id, "category": category, "reason": "test_pass is null; manual review evidence required"})

            batch_id = record.get("batch_id")
            if record_id not in validation_index.get(str(batch_id), set()):
                failures.append({"id": record_id, "category": category, "reason": "missing validation log entry"})

    status = "blocked" if failures else "warning" if warnings else "complete"
    return {"status": status, "failures": failures, "warnings": warnings}


def repair_clean_metadata(workspace_root: str | Path = ".") -> dict[str, Any]:
    root = Path(workspace_root)
    validation_records = _validation_record_index(root)
    review_index = _review_index(root)
    updated: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for category in ALLOWED_CATEGORIES:
        clean_dir = root / "dataset/clean_dataset" / category
        for path in sorted(clean_dir.glob("*.jsonl")):
            changed = False
            repaired_records = []
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                record_id = str(record.get("id", ""))
                validation = validation_records.get(record_id)
                if validation is None:
                    skipped.append({"id": record_id or "<missing id>", "reason": "missing validation log entry"})
                    repaired_records.append(record)
                    continue
                repaired = _repair_record(record, category=category, validation=validation, review=review_index.get(record_id))
                changed = changed or repaired != record
                repaired_records.append(repaired)
                if repaired != record:
                    updated.append({"id": record_id, "path": str(path)})
            if changed:
                write_jsonl(path, repaired_records)

    return {"updated_count": len(updated), "updated": updated, "skipped": skipped}


def build_exact_duplicate_clusters(records_by_category: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    clusters = []
    for category, records in records_by_category.items():
        buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for index, record in enumerate(records):
            for key_type, value in _exact_key_values(record):
                digest = _sha256(_normalize_jsonish(value))
                buckets.setdefault((category, key_type, digest), []).append({"record": record, "source_order": index})

        for (cluster_category, key_type, digest), members in sorted(buckets.items()):
            if len(members) > 1:
                clusters.append(
                    {
                        "category": cluster_category,
                        "key_type": key_type,
                        "dedup_hash": digest,
                        "records": members,
                    }
                )
    return clusters


def deduplicate_records(
    workspace_root: str | Path,
    records_by_category: dict[str, list[dict[str, Any]]],
    *,
    version: str,
    write_logs: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    root = Path(workspace_root)
    kept_by_category = {category: [deepcopy(record) for record in records] for category, records in records_by_category.items()}
    removed_ids: set[str] = set()
    removed_duplicates: list[dict[str, Any]] = []

    for cluster in build_exact_duplicate_clusters(records_by_category):
        active_members = [member for member in cluster["records"] if member["record"].get("id") not in removed_ids]
        if len(active_members) < 2:
            continue
        kept = select_best_record(active_members)
        kept_id = kept["record"]["id"]
        for member in active_members:
            record = member["record"]
            if record["id"] == kept_id:
                continue
            removed_ids.add(record["id"])
            removed_duplicates.append(
                {
                    "category": cluster["category"],
                    "duplicate_key_type": cluster["key_type"],
                    "dedup_hash": cluster["dedup_hash"],
                    "kept_id": kept_id,
                    "removed_id": record["id"],
                    "source_batch_id": record.get("batch_id"),
                    "removal_reason": "exact duplicate with lower quality signals",
                }
            )

    for category, records in kept_by_category.items():
        filtered = []
        for record in records:
            if record.get("id") in removed_ids:
                continue
            release_record = _strip_internal_fields(record)
            release_record["dedup_hash"] = _record_dedup_hash(release_record)
            filtered.append(release_record)
        kept_by_category[category] = filtered

    summary = {
        "version": version,
        "removed_count": len(removed_duplicates),
        "removed_duplicates": removed_duplicates,
    }
    if write_logs:
        ensure_dataset_tree(workspace_root)
        write_json(root / "dataset/logs/deduplication" / f"{version}-exact.json", summary)
    return kept_by_category, summary


def select_best_record(members: list[dict[str, Any]]) -> dict[str, Any]:
    def score(member: dict[str, Any]) -> tuple[int, int, int, int, int]:
        record = member["record"]
        complexity_rank = {"simple": 0, "medium": 1, "hard": 2}.get(record.get("complexity"), 0)
        return (
            int(record.get("manually_reviewed") is True),
            int(record.get("review_status") == "passed"),
            int(record.get("test_pass") is True),
            complexity_rank,
            -member.get("source_order", 0),
        )

    return max(members, key=score)


def build_near_duplicate_report(
    workspace_root: str | Path,
    records_by_category: dict[str, list[dict[str, Any]]],
    *,
    version: str,
    write_logs: bool = True,
) -> dict[str, Any]:
    clusters = []
    resolutions = _near_duplicate_resolutions(Path(workspace_root), version)
    for category, records in records_by_category.items():
        for left_index, left in enumerate(records):
            for right in records[left_index + 1 :]:
                left_key = _near_key(left)
                right_key = _near_key(right)
                if not left_key or not right_key:
                    continue
                score = difflib.SequenceMatcher(None, left_key, right_key).ratio()
                if score >= NEAR_DUPLICATE_THRESHOLD:
                    cluster_id = _near_duplicate_cluster_id(category, [left["id"], right["id"]])
                    resolution = resolutions.get(cluster_id)
                    clusters.append(
                        {
                            "cluster_id": cluster_id,
                            "category": category,
                            "record_ids": [left["id"], right["id"]],
                            "similarity": round(score, 4),
                            "reason": "high normalized content similarity",
                            "action": "manual_review_required",
                            "resolution_status": resolution.get("action", "pending") if resolution else "pending",
                            "resolution": resolution,
                        }
                    )

    unresolved_count = sum(1 for cluster in clusters if cluster["resolution_status"] == "pending")
    report = {"version": version, "flagged_count": len(clusters), "unresolved_count": unresolved_count, "clusters": clusters}
    if write_logs:
        ensure_dataset_tree(workspace_root)
        write_json(Path(workspace_root) / "dataset/logs/deduplication" / f"{version}-near.json", report)
    return report


def resolve_near_duplicate_cluster(
    workspace_root: str | Path = ".",
    *,
    version: str,
    cluster_id: str,
    action: str,
    reviewer: str,
    notes: str,
) -> dict[str, Any]:
    if action not in {"remove_duplicate", "keep_distinct", "waived"}:
        raise ValueError(f"invalid near-duplicate resolution action: {action}")
    if not notes.strip():
        raise ValueError("notes are required for near-duplicate resolution")
    root = Path(workspace_root)
    resolutions = _near_duplicate_resolutions(root, version)
    resolution = {
        "cluster_id": cluster_id,
        "action": action,
        "reviewer": reviewer,
        "notes": notes,
    }
    resolutions[cluster_id] = resolution
    write_json(root / "dataset/logs/deduplication" / f"{version}-near-resolutions.json", {"version": version, "resolutions": resolutions})
    return resolution


def build_review_sample(
    workspace_root: str | Path,
    records_by_category: dict[str, list[dict[str, Any]]],
    *,
    version: str,
    sample_rate: float = 0.1,
    write_sample: bool = True,
) -> dict[str, Any]:
    rng = random.Random(version)
    categories: dict[str, Any] = {}
    for category, records in records_by_category.items():
        if not records:
            categories[category] = {"sample_size": 0, "records": []}
            continue
        sample_size = max(1, math.ceil(len(records) * sample_rate))
        selected = _stratified_sample(records, sample_size, rng)
        categories[category] = {
            "sample_size": len(selected),
            "records": [
                {
                    "id": record["id"],
                    "category": category,
                    "complexity": record.get("complexity"),
                    "batch_id": record.get("batch_id"),
                    "source_path": record.get("_source_path"),
                    "criteria": list(CATEGORY_REVIEW_CRITERIA[category]),
                }
                for record in selected
            ],
        }

    sample = {"version": version, "sample_rate": sample_rate, "categories": categories}
    if write_sample:
        ensure_dataset_tree(workspace_root)
        write_json(Path(workspace_root) / "dataset/review" / f"{version}-manual-review-sample.json", sample)
    return sample


def summarize_manual_review(workspace_root: str | Path, sample: dict[str, Any]) -> dict[str, Any]:
    review_index = _review_index(Path(workspace_root))
    categories = {}
    blocked_pending = False
    blocked_failed = False
    threshold = PASS_RATE_THRESHOLDS["manual_review_pass_minimum"]

    for category, category_sample in sample["categories"].items():
        sampled = category_sample["records"]
        if not sampled:
            categories[category] = {"sample_size": 0, "reviewed_count": 0, "passed_count": 0, "pass_rate": None}
            continue

        reviewed_count = 0
        passed_count = 0
        pending_ids = []
        for item in sampled:
            review = review_index.get(item["id"])
            if review is None or review.get("review_status") == "pending":
                pending_ids.append(item["id"])
                continue
            reviewed_count += 1
            if review.get("review_status") == "passed" and all(review.get("criteria_results", {}).values()):
                passed_count += 1

        pass_rate = passed_count / reviewed_count if reviewed_count else 0.0
        if pending_ids:
            blocked_pending = True
        elif pass_rate < threshold:
            blocked_failed = True
        categories[category] = {
            "sample_size": len(sampled),
            "reviewed_count": reviewed_count,
            "passed_count": passed_count,
            "pass_rate": pass_rate,
            "pending_ids": pending_ids,
        }

    if blocked_pending:
        status = "blocked_manual_review_pending"
    elif blocked_failed:
        status = "blocked_manual_review_failed"
    else:
        status = "complete"
    return {"status": status, "threshold": threshold, "categories": categories}


def build_manifest(
    version: str,
    records_by_category: dict[str, list[dict[str, Any]]],
    dedup_summary: dict[str, Any],
    review_summary: dict[str, Any],
    *,
    known_limitations: list[str] | None = None,
) -> dict[str, Any]:
    all_records = [record for records in records_by_category.values() for record in records]
    dates = sorted(str(record.get("generation_date", "")) for record in all_records if record.get("generation_date"))
    return {
        "dataset_version": version,
        "generation_date_range": {"start": dates[0] if dates else None, "end": dates[-1] if dates else None},
        "category_counts": {category: len(records_by_category.get(category, [])) for category in ALLOWED_CATEGORIES},
        "total_count": len(all_records),
        "prompt_versions": sorted({str(record.get("source_prompt_version")) for record in all_records if record.get("source_prompt_version")}),
        "context_bundle_versions": sorted({str(record.get("context_bundle_version")) for record in all_records if record.get("context_bundle_version")}),
        "validator_versions": sorted({str(record.get("validator_version")) for record in all_records if record.get("validator_version")}),
        "deduplication_summary": dedup_summary,
        "manual_review_summary": review_summary,
        "known_limitations": known_limitations or [],
    }


def audit_release_readiness(
    workspace_root: str | Path = ".",
    *,
    version: str,
    allow_pilot_release: bool = False,
    write_artifacts: bool = False,
    write_report: bool = False,
) -> dict[str, Any]:
    root = Path(workspace_root)
    candidates = load_clean_candidates(workspace_root)
    candidate_audit = audit_candidates(workspace_root, candidates)
    deduped, dedup_summary = deduplicate_records(workspace_root, candidates, version=version, write_logs=write_artifacts)
    near_report = build_near_duplicate_report(workspace_root, deduped, version=version, write_logs=write_artifacts)
    sample = build_review_sample(workspace_root, deduped, version=version, write_sample=write_artifacts)
    review_summary = summarize_manual_review(workspace_root, sample)
    manifest = build_manifest(version, deduped, dedup_summary, review_summary)
    preflight = audit_prerequisites(workspace_root)
    count_summary = _count_summary(deduped)
    hard_ratios = _hard_ratios(deduped)
    status = _release_status(
        preflight,
        candidate_audit,
        review_summary,
        count_summary,
        allow_pilot_release,
        hard_ratios=hard_ratios,
        near_report=near_report,
    )
    limitations = _known_limitations(preflight, candidate_audit, review_summary, count_summary, hard_ratios, near_report)

    audit = {
        "version": version,
        "status": status,
        "preflight": preflight,
        "candidate_audit": candidate_audit,
        "deduplication_summary": dedup_summary,
        "near_duplicate_summary": near_report,
        "manual_review_sample": sample,
        "manual_review_summary": review_summary,
        "count_summary": count_summary,
        "hard_example_ratios": hard_ratios,
        "manifest": {**manifest, "known_limitations": limitations},
        "known_limitations": limitations,
    }
    audit["fingerprint"] = _audit_fingerprint(audit)
    if write_report:
        write_json(root / "dataset/logs/audit" / f"{version}-audit.json", audit)
    return audit


def write_audit_snapshot(workspace_root: str | Path, audit: dict[str, Any]) -> str:
    root = Path(workspace_root)
    path = root / "dataset/logs/audit" / f"{audit['version']}-audit-snapshot.json"
    write_json(path, audit)
    return str(path)


def build_readiness_summary(
    workspace_root: str | Path = ".",
    *,
    version: str,
    allow_pilot_release: bool = False,
    write_report: bool = False,
) -> dict[str, Any]:
    root = Path(workspace_root)
    audit = audit_release_readiness(
        root,
        version=version,
        allow_pilot_release=allow_pilot_release,
        write_artifacts=False,
        write_report=False,
    )
    readiness = {
        "version": version,
        "status": audit["status"],
        "blockers": list(audit["known_limitations"]),
        "counts": _readiness_counts(audit["count_summary"]),
        "hard_example_ratios": audit["hard_example_ratios"],
        "manual_review": _readiness_manual_review(audit["manual_review_summary"]),
        "near_duplicates": {
            "unresolved_count": audit["near_duplicate_summary"]["flagged_count"],
            "resolved_count": 0,
            "waived_count": 0,
        },
        "latest_scale_decisions": _latest_scale_decisions(root),
        "prompt_versions": audit["manifest"]["prompt_versions"],
        "validation": {
            "candidate_audit_status": audit["candidate_audit"]["status"],
            "candidate_audit_failure_count": len(audit["candidate_audit"]["failures"]),
            "candidate_audit_warning_count": len(audit["candidate_audit"]["warnings"]),
            "candidate_audit_warning_counts": _reason_counts(audit["candidate_audit"]["warnings"]),
        },
        "release_status_reasons": _release_status_reasons(audit),
    }
    if write_report:
        write_json(root / "dataset/logs/audit" / f"{version}-readiness.json", readiness)
    return readiness


def freeze_release(
    workspace_root: str | Path = ".",
    *,
    version: str,
    allow_pilot_release: bool = False,
    force: bool = False,
    training_runs: list[str] | None = None,
    audit_snapshot_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(workspace_root)
    release_dir = root / "dataset/releases" / version
    if release_dir.exists() and any(release_dir.iterdir()) and not force:
        raise FileExistsError(f"release already exists: {release_dir}")
    if release_dir.exists() and force:
        shutil.rmtree(release_dir)

    if audit_snapshot_path:
        audit = json.loads(Path(audit_snapshot_path).read_text())
        current_audit = audit_release_readiness(root, version=audit["version"], allow_pilot_release=allow_pilot_release, write_artifacts=False)
        if current_audit["fingerprint"] != audit.get("fingerprint"):
            raise ValueError("audit snapshot is stale; re-run audit before freezing")
    else:
        audit = audit_release_readiness(root, version=version, allow_pilot_release=allow_pilot_release, write_artifacts=True)
    if audit["status"] == "blocked" or (audit["status"] == "pilot_only_not_volume_complete" and not allow_pilot_release):
        raise ValueError(f"release is not ready: {audit['status']}")

    release_dir.mkdir(parents=True, exist_ok=True)
    clean_dir = release_dir / "clean_dataset"
    clean_dir.mkdir(parents=True, exist_ok=True)
    deduped = deduplicate_records(root, load_clean_candidates(root), version=version)[0]
    for category, records in deduped.items():
        write_jsonl(clean_dir / f"{category}.jsonl", [_strip_internal_fields(record) for record in records])

    write_json(release_dir / "manifest.json", audit["manifest"])
    write_json(release_dir / "audit.json", audit)
    write_json(release_dir / "manual_review_sample.json", audit["manual_review_sample"])
    write_json(release_dir / "deduplication_summary.json", audit["deduplication_summary"])
    write_json(release_dir / "training_runs.json", training_runs or [])
    checksums = _release_checksums(release_dir)
    write_json(
        release_dir / "IMMUTABLE_RELEASE.json",
        {
            "version": version,
            "status": audit["status"],
            "audit_fingerprint": audit["fingerprint"],
            "checksums": checksums,
        },
    )
    return {"version": version, "status": audit["status"], "release_path": str(release_dir), "audit": audit}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit and freeze Jac synthetic dataset releases.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--version", required=True)
    audit_parser.add_argument("--allow-pilot-release", action="store_true")
    audit_parser.add_argument("--write-report", action="store_true")

    readiness_parser = subparsers.add_parser("readiness")
    readiness_parser.add_argument("--version", required=True)
    readiness_parser.add_argument("--allow-pilot-release", action="store_true")
    readiness_parser.add_argument("--write-report", action="store_true")

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--version", required=True)
    freeze_parser.add_argument("--allow-pilot-release", action="store_true")
    freeze_parser.add_argument("--force", action="store_true")
    freeze_parser.add_argument("--training-run", action="append", default=[])

    subparsers.add_parser("repair-metadata")

    args = parser.parse_args(argv)
    if args.command == "audit":
        print(
            json.dumps(
                audit_release_readiness(
                    ".",
                    version=args.version,
                    allow_pilot_release=args.allow_pilot_release,
                    write_report=args.write_report,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "readiness":
        print(
            json.dumps(
                build_readiness_summary(
                    ".",
                    version=args.version,
                    allow_pilot_release=args.allow_pilot_release,
                    write_report=args.write_report,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "freeze":
        print(
            json.dumps(
                freeze_release(
                    ".",
                    version=args.version,
                    allow_pilot_release=args.allow_pilot_release,
                    force=args.force,
                    training_runs=args.training_run,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "repair-metadata":
        print(json.dumps(repair_clean_metadata("."), indent=2, sort_keys=True))
        return 0
    return 2


def _audit_task1_storage(root: Path) -> dict[str, Any]:
    missing = []
    for value in DATASET_STORAGE_PATHS.values():
        if isinstance(value, dict):
            paths = value.values()
        else:
            paths = (value,)
        for path in paths:
            if not (root / path).exists():
                missing.append(path)
    return _status_payload(missing)


def _audit_task2_prompts() -> dict[str, Any]:
    missing = [category for category in SCRIPTED_CATEGORIES if category not in CATEGORY_SCHEMAS]
    return _status_payload(missing)


def _audit_task3_validation() -> dict[str, Any]:
    missing = [field for field in VALIDATION_LOG_REQUIRED_FIELDS if not field]
    missing.extend(category for category in ALLOWED_CATEGORIES if category not in _COMPILER_FIELD_POLICIES)
    return _status_payload(missing)


def _audit_task4_artifacts(root: Path) -> dict[str, Any]:
    missing = []
    for category in SCRIPTED_CATEGORIES:
        checks = {
            "raw_output": root / "dataset/raw_output" / category,
            "clean_dataset": root / "dataset/clean_dataset" / category,
            "validation": root / "dataset/logs/validation",
            "review": root / "dataset/review" / category,
            "scale_decisions": root / "dataset/logs/scale_decisions",
        }
        if not any(checks["raw_output"].glob("*.json")):
            missing.append(f"{category}: raw output")
        if not any(checks["clean_dataset"].glob("*.jsonl")):
            missing.append(f"{category}: clean candidates")
        if not any(checks["validation"].glob(f"*-{category}-*.jsonl")):
            missing.append(f"{category}: validation log")
        if not any(checks["review"].glob("*.json")):
            missing.append(f"{category}: review file")
        if not any(checks["scale_decisions"].glob(f"*-{category}-*.json")):
            missing.append(f"{category}: scale decision")
    return _status_payload(missing)


def _audit_task5_artifacts(root: Path, *, trajectory_batch_id: str | None) -> dict[str, Any]:
    batch_id = trajectory_batch_id or _latest_batch_id(root / "dataset/clean_dataset/trajectory", suffix=".jsonl")
    if not batch_id:
        return {"status": "blocked", "reasons": ["trajectory: clean candidates"]}
    checks = {
        "raw": root / "dataset/raw_output/trajectory" / f"{batch_id}.json",
        "clean": root / "dataset/clean_dataset/trajectory" / f"{batch_id}.jsonl",
        "validation": root / "dataset/logs/validation" / f"{batch_id}.jsonl",
        "review": root / "dataset/review/trajectory" / f"{batch_id}-review.json",
        "generation": root / "dataset/logs/generation" / f"{batch_id}.json",
    }
    missing = [name for name, path in checks.items() if not path.exists()]
    return _status_payload(missing, extra={"batch_id": batch_id})


def _status_payload(missing: list[str], *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {"status": "blocked" if missing else "complete", "reasons": missing}
    if extra:
        payload.update(extra)
    return payload


def _rollup_status(statuses: Any) -> str:
    statuses = list(statuses)
    if any(status == "blocked" for status in statuses):
        return "blocked"
    if any(status == "warning" for status in statuses):
        return "warning"
    return "complete"


def _latest_batch_id(directory: Path, *, suffix: str) -> str | None:
    candidates = sorted(directory.glob(f"*{suffix}"))
    if not candidates:
        return None
    return candidates[-1].name.removesuffix(suffix)


def _validation_log_index(root: Path) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for path in sorted((root / "dataset/logs/validation").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            index.setdefault(str(record.get("batch_id")), set()).add(str(record.get("example_id")))
    return index


def _validation_record_index(root: Path) -> dict[str, dict[str, Any]]:
    index = {}
    for path in sorted((root / "dataset/logs/validation").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("example_id"):
                index[str(record["example_id"])] = record
    return index


def _review_index(root: Path) -> dict[str, dict[str, Any]]:
    index = {}
    for path in sorted((root / "dataset/review").glob("**/*.json")):
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            payload = payload.get("records", [])
        if not isinstance(payload, list):
            continue
        for record in payload:
            if isinstance(record, dict) and "example_id" in record:
                index[str(record["example_id"])] = record
    return index


def _repair_record(
    record: dict[str, Any],
    *,
    category: str,
    validation: dict[str, Any],
    review: dict[str, Any] | None,
) -> dict[str, Any]:
    batch_id = str(validation.get("batch_id") or record.get("batch_id") or _batch_id_from_record_id(str(record.get("id", "")), category))
    review_passed = bool(review and review.get("review_status") == "passed" and all(review.get("criteria_results", {}).values()))
    repaired = dict(record)
    repaired["batch_id"] = batch_id
    repaired["category"] = category
    repaired["complexity"] = _record_complexity(category, repaired)
    repaired["compiler_pass"] = _validation_compiler_pass(validation)
    repaired["test_pass"] = validation.get("test_result")
    repaired["manually_reviewed"] = review_passed
    repaired["generator"] = "cursor-jac-mcp" if category == "trajectory" else "openai-api"
    repaired["generation_date"] = repaired.get("generation_date") or _generation_date_from_batch_id(batch_id)
    repaired["source_prompt_version"] = validation.get("prompt_version") or repaired.get("source_prompt_version") or f"prompt-{category}-v1"
    repaired["context_bundle_version"] = validation.get("context_bundle_version") or repaired.get("context_bundle_version") or "jac-context-v1"
    repaired["validator_version"] = validation.get("validator_version") or repaired.get("validator_version") or "validator-v1"
    repaired["dataset_version"] = validation.get("dataset_version") or repaired.get("dataset_version") or "jac-synth-v0.1.0"
    repaired["review_status"] = review.get("review_status") if review else repaired.get("review_status", "pending")
    repaired["rejection_reason"] = validation.get("rejection_reason")
    return repaired


def _record_complexity(category: str, record: dict[str, Any]) -> str:
    if record.get("complexity"):
        return str(record["complexity"])
    if category == "explanation":
        granularity = record.get("granularity")
        return "simple" if granularity == "line" else "hard" if granularity == "module" else "medium"
    if category == "trajectory":
        return "medium"
    return "medium"


def _validation_compiler_pass(validation: dict[str, Any]) -> bool | None:
    results = validation.get("compiler_result")
    if isinstance(results, list) and results:
        return all(
            bool(result.get("passed")) is bool(result.get("expected_to_compile", True))
            for result in results
            if isinstance(result, dict)
        )
    return None


def _batch_id_from_record_id(record_id: str, category: str) -> str:
    parts = record_id.removeprefix(f"{category}-").split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{category}-{parts[1]}"
    return ""


def _generation_date_from_batch_id(batch_id: str) -> str:
    date = batch_id.split("-", 1)[0]
    if len(date) == 8:
        return f"{date[0:4]}-{date[4:6]}-{date[6:8]}T00:00:00Z"
    return ""


def _has_review_exception(record: dict[str, Any], review_index: dict[str, dict[str, Any]]) -> bool:
    review = review_index.get(str(record.get("id")))
    return bool(review and review.get("review_status") == "passed")


def _exact_key_values(record: dict[str, Any]) -> list[tuple[str, Any]]:
    category = record.get("category")
    keys: list[tuple[str, Any]] = []
    if category == "code_gen":
        keys.extend([("prompt", record.get("prompt")), ("code", record.get("code"))])
    elif category == "debug":
        keys.extend([("broken_code", record.get("broken_code")), ("fixed_code", record.get("fixed_code"))])
    elif category == "explanation":
        keys.append(("code", record.get("code")))
    elif category == "conversion":
        keys.extend([("python_code", record.get("python_code")), ("jac_code", record.get("jac_code"))])
    elif category == "trajectory":
        keys.extend([("final_output.code", record.get("final_output", {}).get("code")), ("turns", record.get("turns"))])
    return [(key, value) for key, value in keys if value]


def _record_dedup_hash(record: dict[str, Any]) -> str:
    return _sha256(_normalize_jsonish({key: value for key, value in _exact_key_values(record)}))


def _near_duplicate_cluster_id(category: str, record_ids: list[str]) -> str:
    return _sha256(_normalize_jsonish({"category": category, "record_ids": sorted(record_ids)}))[:16]


def _near_duplicate_resolutions(root: Path, version: str) -> dict[str, dict[str, Any]]:
    path = root / "dataset/logs/deduplication" / f"{version}-near-resolutions.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    resolutions = payload.get("resolutions", {})
    return resolutions if isinstance(resolutions, dict) else {}


def _near_key(record: dict[str, Any]) -> str:
    category = record.get("category")
    if category == "code_gen":
        parts = [record.get("prompt", ""), record.get("code", "")]
    elif category == "debug":
        parts = [record.get("broken_code", ""), record.get("fixed_code", ""), record.get("error_type", "")]
    elif category == "explanation":
        parts = [record.get("code", ""), record.get("granularity", "")]
    elif category == "conversion":
        parts = [record.get("python_code", ""), record.get("jac_code", "")]
    elif category == "trajectory":
        roles = " ".join(turn.get("role", "") for turn in record.get("turns", []))
        parts = [record.get("task", {}).get("prompt", ""), record.get("final_output", {}).get("code", ""), roles]
    else:
        parts = []
    return _normalize_identifiers("\n".join(str(part) for part in parts))


def _normalize_jsonish(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _normalize_identifiers(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"\b[a-z_][a-z0-9_]*\b", "id", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _audit_fingerprint(audit: dict[str, Any]) -> str:
    payload = deepcopy(audit)
    payload.pop("fingerprint", None)
    volatile = payload.get("near_duplicate_summary")
    if isinstance(volatile, dict):
        volatile.pop("clusters", None)
    return _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _release_checksums(release_dir: Path) -> dict[str, str]:
    checksums = {}
    for path in sorted(release_dir.glob("**/*")):
        if path.is_file() and path.name != "IMMUTABLE_RELEASE.json":
            checksums[str(path.relative_to(release_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return checksums


def _strip_internal_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _stratified_sample(records: list[dict[str, Any]], sample_size: int, rng: random.Random) -> list[dict[str, Any]]:
    groups = {
        "hard": [record for record in records if record.get("complexity") == "hard"],
        "medium": [record for record in records if record.get("complexity") == "medium"],
        "simple": [record for record in records if record.get("complexity") == "simple"],
    }
    selected = []
    for complexity in ("hard", "medium", "simple"):
        if len(selected) >= sample_size:
            break
        group = sorted(groups[complexity], key=lambda record: record["id"])
        if group:
            selected.append(rng.choice(group))
    remaining = [record for record in records if record not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, sample_size - len(selected))])
    return sorted(selected[:sample_size], key=lambda record: record["id"])


def _count_summary(records_by_category: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    by_category = {category: len(records_by_category.get(category, [])) for category in ALLOWED_CATEGORIES}
    return {
        "by_category": by_category,
        "total": sum(by_category.values()),
        "target_total_range": RELEASE_TOTAL_RANGE,
        "category_target_ranges": CATEGORY_TARGET_RANGES,
    }


def _hard_ratios(records_by_category: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    ratios = {}
    for category, records in records_by_category.items():
        hard_count = sum(1 for record in records if record.get("complexity") == "hard")
        ratios[category] = {
            "hard_count": hard_count,
            "total": len(records),
            "ratio": hard_count / len(records) if records else None,
            "target_ratio": TARGET_HARD_RATIO,
        }
    return ratios


def _readiness_counts(count_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_total": count_summary["total"],
        "target_total_range": list(count_summary["target_total_range"]),
        "by_category": {
            category: {
                "current": count_summary["by_category"].get(category, 0),
                "target_range": list(count_summary["category_target_ranges"][category]),
            }
            for category in ALLOWED_CATEGORIES
        },
    }


def _readiness_manual_review(review_summary: dict[str, Any]) -> dict[str, Any]:
    categories = {}
    for category, summary in review_summary["categories"].items():
        categories[category] = {
            "sample_size": summary.get("sample_size", 0),
            "reviewed_count": summary.get("reviewed_count", 0),
            "passed_count": summary.get("passed_count", 0),
            "pending_count": len(summary.get("pending_ids", [])),
            "pass_rate": summary.get("pass_rate"),
        }
    return {
        "status": review_summary["status"],
        "threshold": review_summary["threshold"],
        "categories": categories,
    }


def _reason_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        reason = str(item.get("reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _latest_scale_decisions(root: Path) -> dict[str, dict[str, Any]]:
    decisions = {}
    for category in SCRIPTED_CATEGORIES:
        paths = sorted((root / "dataset/logs/scale_decisions").glob(f"*-{category}-*.json"))
        if not paths:
            decisions[category] = {"status": "missing"}
            continue
        payload = json.loads(paths[-1].read_text())
        decisions[category] = {
            "batch_id": payload.get("batch_id"),
            "decision": payload.get("decision"),
            "reasons": payload.get("reasons", []),
        }
    return decisions


def _release_status_reasons(audit: dict[str, Any]) -> list[str]:
    reasons = []
    if audit["preflight"]["overall_status"] != "complete":
        reasons.append("preflight incomplete")
    if audit["candidate_audit"]["status"] == "blocked":
        reasons.append("candidate audit blocked")
    if audit["manual_review_summary"]["status"].startswith("blocked"):
        reasons.append(audit["manual_review_summary"]["status"])
    if audit["count_summary"]["total"] < RELEASE_TOTAL_RANGE[0]:
        reasons.append("below release volume minimum")
    if audit["near_duplicate_summary"]["flagged_count"]:
        reasons.append("near duplicates unresolved")
    return reasons


def _release_status(
    preflight: dict[str, Any],
    candidate_audit: dict[str, Any],
    review_summary: dict[str, Any],
    count_summary: dict[str, Any],
    allow_pilot_release: bool,
    hard_ratios: dict[str, Any] | None = None,
    near_report: dict[str, Any] | None = None,
) -> str:
    if preflight["overall_status"] == "blocked" or candidate_audit["status"] == "blocked" or review_summary["status"].startswith("blocked"):
        return "blocked"
    total = count_summary["total"]
    in_total_range = RELEASE_TOTAL_RANGE[0] <= total <= RELEASE_TOTAL_RANGE[1]
    category_counts = count_summary["by_category"]
    in_category_ranges = all(
        CATEGORY_TARGET_RANGES[category][0] <= category_counts.get(category, 0) <= CATEGORY_TARGET_RANGES[category][1]
        for category in ALLOWED_CATEGORIES
    )
    if in_total_range and in_category_ranges:
        if _hard_ratio_blocked(hard_ratios or {}):
            return "blocked"
        if (near_report or {}).get("unresolved_count", 0):
            return "blocked"
        return "ready"
    return "pilot_only_not_volume_complete" if allow_pilot_release or total < RELEASE_TOTAL_RANGE[0] else "blocked"


def _hard_ratio_blocked(hard_ratios: dict[str, Any]) -> bool:
    for ratio in hard_ratios.values():
        value = ratio.get("ratio")
        if value is not None and abs(value - TARGET_HARD_RATIO) > 0.15:
            return True
    return False


def _known_limitations(
    preflight: dict[str, Any],
    candidate_audit: dict[str, Any],
    review_summary: dict[str, Any],
    count_summary: dict[str, Any],
    hard_ratios: dict[str, Any],
    near_report: dict[str, Any],
) -> list[str]:
    limitations = []
    if preflight["overall_status"] != "complete":
        limitations.append("Tasks 1-5 preflight is not complete.")
    if candidate_audit["warnings"]:
        limitations.append("Some clean candidates have nullable test results that require review context.")
    if review_summary["status"] != "complete":
        limitations.append("Manual review sample is not fully passed.")
    if count_summary["total"] < RELEASE_TOTAL_RANGE[0]:
        limitations.append("Clean example count is below the 10,000 example release minimum.")
    for category, ratio in hard_ratios.items():
        value = ratio["ratio"]
        if value is not None and abs(value - TARGET_HARD_RATIO) > 0.15:
            limitations.append(f"{category} hard-example ratio is outside the expected band.")
    if near_report["flagged_count"]:
        limitations.append("Near-duplicate clusters require manual review.")
    return limitations


if __name__ == "__main__":
    sys.exit(main())
