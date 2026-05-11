import json
from pathlib import Path

import pytest

from data_generation.artifacts import ensure_dataset_tree, write_json, write_jsonl
from data_generation.foundation import ALLOWED_CATEGORIES, SCRIPTED_CATEGORIES
from data_generation.release import (
    audit_candidates,
    audit_prerequisites,
    audit_release_readiness,
    build_exact_duplicate_clusters,
    build_manifest,
    build_near_duplicate_report,
    build_review_sample,
    deduplicate_records,
    freeze_release,
    load_clean_candidates,
    main,
    summarize_manual_review,
)


def metadata_record(
    *,
    record_id: str,
    batch_id: str,
    category: str = "code_gen",
    complexity: str = "simple",
    compiler_pass: bool = True,
    test_pass: bool | None = True,
    manually_reviewed: bool = False,
    review_status: str = "pending",
    extra: dict | None = None,
) -> dict:
    record = {
        "id": record_id,
        "batch_id": batch_id,
        "category": category,
        "complexity": complexity,
        "compiler_pass": compiler_pass,
        "test_pass": test_pass,
        "manually_reviewed": manually_reviewed,
        "generator": "cursor-jac-mcp" if category == "trajectory" else "openai-api",
        "generation_date": "2026-05-11T00:00:00Z",
        "source_prompt_version": "trajectory-prompt-v1" if category == "trajectory" else f"prompt-{category}-v1",
        "context_bundle_version": "jac-context-v1",
        "validator_version": "validator-v1",
        "dataset_version": "jac-synth-v0.1.0",
        "review_status": review_status,
    }
    if category == "code_gen":
        record.update({"prompt": "Build a Jac walker.", "code": "node Item { has value: int; }"})
    elif category == "debug":
        record.update(
            {
                "broken_code": "node Item { has value: int BROKEN }",
                "error_type": "syntax",
                "error_message": "syntax error",
                "fixed_code": "node Item { has value: int; }",
                "fix_explanation": "Added the missing separator.",
            }
        )
    elif category == "explanation":
        record.update(
            {
                "code": "node Item { has value: int; }",
                "granularity": "block",
                "explanation": "Defines a node with an integer field.",
            }
        )
    elif category == "conversion":
        record.update(
            {
                "python_code": "class Item:\n    pass",
                "jac_code": "node Item {}",
                "conversion_notes": "Converted a Python class to a Jac node.",
            }
        )
    elif category == "trajectory":
        record.update(
            {
                "trajectory_length_tokens": 120,
                "task": {"prompt": "Build a Jac walker.", "difficulty_reason": "Uses walkers."},
                "final_output": {
                    "language": "jac",
                    "code": "node Item { has value: int; }",
                    "validation_tool": "user-jac.validate_jac",
                    "validation_result": {"passed": True},
                },
                "turns": [
                    {"role": "user", "content": "Build a Jac walker."},
                    {"role": "assistant", "content": "I will use Jac MCP."},
                    {"role": "tool_call", "content": "user-jac.validate_jac({})"},
                    {"role": "tool_result", "content": '{"passed": true}'},
                ],
            }
        )
    if extra:
        record.update(extra)
    return record


def write_batch(
    root: Path,
    *,
    category: str,
    date: str = "20260511",
    sequence: int = 1,
    records: list[dict] | None = None,
    scale_ready: bool = True,
) -> str:
    ensure_dataset_tree(root)
    batch_id = f"{date}-{category}-{sequence:03d}"
    records = records or [
        metadata_record(
            record_id=f"{category}-{date}-{sequence:03d}-0001",
            batch_id=batch_id,
            category=category,
            complexity="hard" if category == "trajectory" else "simple",
            test_pass=None if category in {"explanation", "trajectory"} else True,
        )
    ]
    write_json(root / f"dataset/raw_output/{category}/{batch_id}.json", {"batch_id": batch_id, "examples": records})
    write_jsonl(root / f"dataset/clean_dataset/{category}/{batch_id}.jsonl", records)
    write_jsonl(
        root / f"dataset/logs/validation/{batch_id}.jsonl",
        [
            {
                "batch_id": batch_id,
                "prompt_version": record["source_prompt_version"],
                "context_bundle_version": record["context_bundle_version"],
                "category": category,
                "example_id": record["id"],
                "json_schema_result": True,
                "compiler_result": [{"passed": True}],
                "test_result": record["test_pass"],
                "rejection_reason": None,
                "retry_count": 0,
                "final_disposition": "clean",
                "validator_version": record["validator_version"],
                "dataset_version": record["dataset_version"],
            }
            for record in records
        ],
    )
    write_json(
        root / f"dataset/review/{category}/{batch_id}-review.json",
        [
            {
                "batch_id": batch_id,
                "category": category,
                "example_id": record["id"],
                "review_status": "passed",
                "reviewer": "manual-reviewer",
                "criteria_results": {"idiomatic_jac": True},
                "notes": "ok",
            }
            for record in records
        ],
    )
    if category in SCRIPTED_CATEGORIES:
        write_json(root / f"dataset/logs/scale_decisions/{batch_id}.json", {"batch_id": batch_id, "decision": "scale_ready" if scale_ready else "revise_prompt"})
    else:
        write_json(
            root / f"dataset/logs/generation/{batch_id}.json",
            {"batch_id": batch_id, "ready_for_volume": True, "clean_count": len(records), "rejected_count": 0},
        )
    return batch_id


def test_audit_prerequisites_reports_tasks_1_to_5_complete(tmp_path):
    ensure_dataset_tree(tmp_path)
    for category in ALLOWED_CATEGORIES:
        write_batch(tmp_path, category=category)

    audit = audit_prerequisites(tmp_path)

    assert audit["overall_status"] == "complete"
    assert audit["tasks"]["task1"]["status"] == "complete"
    assert audit["tasks"]["task2"]["status"] == "complete"
    assert audit["tasks"]["task3"]["status"] == "complete"
    assert audit["tasks"]["task4"]["status"] == "complete"
    assert audit["tasks"]["task5"]["status"] == "complete"


def test_load_clean_candidates_and_audit_metadata_and_validation_logs(tmp_path):
    batch_id = write_batch(tmp_path, category="code_gen")
    missing_metadata = metadata_record(
        record_id="code_gen-20260511-001-0002",
        batch_id=batch_id,
        category="code_gen",
        extra={"compiler_pass": False},
    )
    del missing_metadata["generator"]
    write_jsonl(tmp_path / f"dataset/clean_dataset/code_gen/{batch_id}.jsonl", [missing_metadata])

    candidates = load_clean_candidates(tmp_path)
    candidate_audit = audit_candidates(tmp_path, candidates)

    assert candidates["code_gen"][0]["id"] == "code_gen-20260511-001-0002"
    assert candidate_audit["status"] == "blocked"
    assert any("missing required field: generator" in failure["reason"] for failure in candidate_audit["failures"])
    assert any("compiler_pass is false" in failure["reason"] for failure in candidate_audit["failures"])


def test_exact_deduplication_keeps_highest_quality_record_and_logs_removed_duplicate(tmp_path):
    batch_id = "20260511-code_gen-001"
    low_quality = metadata_record(record_id="code_gen-20260511-001-0001", batch_id=batch_id, category="code_gen")
    high_quality = metadata_record(
        record_id="code_gen-20260511-001-0002",
        batch_id=batch_id,
        category="code_gen",
        complexity="hard",
        manually_reviewed=True,
        review_status="passed",
    )
    records = {"code_gen": [low_quality, high_quality]}

    clusters = build_exact_duplicate_clusters(records)
    deduped, summary = deduplicate_records(tmp_path, records, version="jac-synth-v0.1.0")

    assert len(clusters) == 2
    assert deduped["code_gen"][0]["id"] == "code_gen-20260511-001-0002"
    assert "dedup_hash" in deduped["code_gen"][0]
    assert summary["removed_count"] == 1
    exact_log = tmp_path / "dataset/logs/deduplication/jac-synth-v0.1.0-exact.json"
    assert json.loads(exact_log.read_text())["removed_duplicates"][0]["removed_id"] == "code_gen-20260511-001-0001"


def test_near_duplicate_report_flags_trivial_prompt_rewrites(tmp_path):
    first = metadata_record(
        record_id="code_gen-20260511-001-0001",
        batch_id="20260511-code_gen-001",
        category="code_gen",
        extra={"prompt": "Create a walker that sums alpha values.", "code": "node Alpha { has value: int; }"},
    )
    second = metadata_record(
        record_id="code_gen-20260511-001-0002",
        batch_id="20260511-code_gen-001",
        category="code_gen",
        extra={"prompt": "Create a walker that sums beta values.", "code": "node Beta { has value: int; }"},
    )

    report = build_near_duplicate_report(tmp_path, {"code_gen": [first, second]}, version="jac-synth-v0.1.0")

    assert report["flagged_count"] == 1
    assert report["clusters"][0]["action"] == "manual_review_required"
    assert (tmp_path / "dataset/logs/deduplication/jac-synth-v0.1.0-near.json").exists()


def test_review_sample_is_deterministic_and_summary_blocks_missing_completed_review(tmp_path):
    records = {
        "code_gen": [
            metadata_record(
                record_id=f"code_gen-20260511-001-{index:04d}",
                batch_id="20260511-code_gen-001",
                category="code_gen",
                complexity="hard" if index == 10 else "simple",
            )
            for index in range(1, 11)
        ]
    }

    sample = build_review_sample(tmp_path, records, version="jac-synth-v0.1.0", sample_rate=0.1)
    repeat_sample = build_review_sample(tmp_path, records, version="jac-synth-v0.1.0", sample_rate=0.1)
    summary = summarize_manual_review(tmp_path, sample)

    assert sample == repeat_sample
    assert sample["categories"]["code_gen"]["sample_size"] == 1
    assert "criteria" in sample["categories"]["code_gen"]["records"][0]
    assert summary["status"] == "blocked_manual_review_pending"


def test_manifest_and_audit_report_pilot_only_when_counts_are_below_release_target(tmp_path):
    for category in ALLOWED_CATEGORIES:
        write_batch(tmp_path, category=category)
    candidates = load_clean_candidates(tmp_path)
    deduped, dedup_summary = deduplicate_records(tmp_path, candidates, version="jac-synth-v0.1.0")
    sample = build_review_sample(tmp_path, deduped, version="jac-synth-v0.1.0")
    review_summary = summarize_manual_review(tmp_path, sample)

    manifest = build_manifest("jac-synth-v0.1.0", deduped, dedup_summary, review_summary)
    audit = audit_release_readiness(tmp_path, version="jac-synth-v0.1.0")

    assert manifest["dataset_version"] == "jac-synth-v0.1.0"
    assert manifest["category_counts"]["trajectory"] == 1
    assert audit["status"] == "pilot_only_not_volume_complete"
    assert audit["count_summary"]["total"] == 5


def test_freeze_release_writes_immutable_pilot_release_and_cli_audit(tmp_path, monkeypatch, capsys):
    for category in ALLOWED_CATEGORIES:
        write_batch(tmp_path, category=category)

    monkeypatch.chdir(tmp_path)
    assert main(["audit", "--version", "jac-synth-v0.1.0"]) == 0
    assert '"status": "pilot_only_not_volume_complete"' in capsys.readouterr().out
    assert not (tmp_path / "dataset/logs/deduplication/jac-synth-v0.1.0-exact.json").exists()
    assert not (tmp_path / "dataset/review/jac-synth-v0.1.0-manual-review-sample.json").exists()

    release = freeze_release(tmp_path, version="jac-synth-v0.1.0", allow_pilot_release=True)

    assert release["status"] == "pilot_only_not_volume_complete"
    assert (tmp_path / "dataset/releases/jac-synth-v0.1.0/manifest.json").exists()
    assert (tmp_path / "dataset/releases/jac-synth-v0.1.0/clean_dataset/code_gen.jsonl").exists()
    assert json.loads((tmp_path / "dataset/releases/jac-synth-v0.1.0/training_runs.json").read_text()) == []
    with pytest.raises(FileExistsError):
        freeze_release(tmp_path, version="jac-synth-v0.1.0", allow_pilot_release=True)
