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
    build_readiness_summary,
    build_review_sample,
    deduplicate_records,
    freeze_release,
    load_clean_candidates,
    main,
    repair_clean_metadata,
    resolve_near_duplicate_cluster,
    write_audit_snapshot,
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


def test_candidate_audit_allows_null_test_pass_with_completed_review_evidence(tmp_path):
    record = metadata_record(
        record_id="code_gen-20260511-001-0001",
        batch_id="20260511-code_gen-001",
        category="code_gen",
        test_pass=None,
        manually_reviewed=True,
        review_status="passed",
    )
    write_batch(tmp_path, category="code_gen", records=[record])
    candidates = load_clean_candidates(tmp_path)

    candidate_audit = audit_candidates(tmp_path, candidates)

    assert candidate_audit["status"] == "complete"
    assert candidate_audit["warnings"] == []


def test_repair_clean_metadata_backfills_legacy_clean_records_from_validation_logs(tmp_path):
    ensure_dataset_tree(tmp_path)
    batch_id = "20260511-code_gen-004"
    legacy = {"id": "code_gen-20260511-004-0001", "prompt": "Say hi", "code": "valid jac", "complexity": "simple"}
    write_jsonl(tmp_path / f"dataset/clean_dataset/code_gen/{batch_id}.jsonl", [legacy])
    write_jsonl(
        tmp_path / f"dataset/logs/validation/{batch_id}.jsonl",
        [
            {
                "batch_id": batch_id,
                "prompt_version": "prompt-code_gen-v3",
                "context_bundle_version": "jac-context-v1",
                "category": "code_gen",
                "example_id": legacy["id"],
                "json_schema_result": True,
                "compiler_result": [{"passed": True}],
                "test_result": None,
                "rejection_reason": None,
                "retry_count": 0,
                "final_disposition": "clean",
                "validator_version": "validator-v1",
                "dataset_version": "jac-synth-v0.1.0",
            }
        ],
    )
    write_json(
        tmp_path / f"dataset/review/code_gen/{batch_id}-review.json",
        [
            {
                "batch_id": batch_id,
                "category": "code_gen",
                "example_id": legacy["id"],
                "review_status": "passed",
                "reviewer": "manual-reviewer",
                "criteria_results": {"idiomatic_jac": True},
                "notes": "Reviewed.",
            }
        ],
    )

    summary = repair_clean_metadata(tmp_path)
    repaired = json.loads((tmp_path / f"dataset/clean_dataset/code_gen/{batch_id}.jsonl").read_text().splitlines()[0])
    candidate_audit = audit_candidates(tmp_path, load_clean_candidates(tmp_path))

    assert summary["updated_count"] == 1
    assert repaired["batch_id"] == batch_id
    assert repaired["category"] == "code_gen"
    assert repaired["compiler_pass"] is True
    assert repaired["generator"] == "openai-api"
    assert repaired["source_prompt_version"] == "prompt-code_gen-v3"
    assert repaired["manually_reviewed"] is True
    assert repaired["review_status"] == "passed"
    assert candidate_audit["status"] == "complete"
    assert candidate_audit["failures"] == []


def test_repair_clean_metadata_treats_expected_debug_broken_code_failure_as_clean(tmp_path):
    ensure_dataset_tree(tmp_path)
    batch_id = "20260511-debug-004"
    legacy = {
        "id": "debug-20260511-004-0001",
        "broken_code": "BROKEN jac",
        "error_type": "syntax",
        "error_message": "syntax error",
        "fixed_code": "valid jac",
        "fix_explanation": "Removed the invalid token.",
        "compiler_pass": False,
    }
    write_jsonl(tmp_path / f"dataset/clean_dataset/debug/{batch_id}.jsonl", [legacy])
    write_jsonl(
        tmp_path / f"dataset/logs/validation/{batch_id}.jsonl",
        [
            {
                "batch_id": batch_id,
                "prompt_version": "prompt-debug-v3",
                "context_bundle_version": "jac-context-v1",
                "category": "debug",
                "example_id": legacy["id"],
                "json_schema_result": True,
                "compiler_result": [
                    {"field_name": "broken_code", "expected_to_compile": False, "passed": False},
                    {"field_name": "fixed_code", "expected_to_compile": True, "passed": True},
                ],
                "test_result": None,
                "rejection_reason": None,
                "retry_count": 0,
                "final_disposition": "clean",
                "validator_version": "validator-v1",
                "dataset_version": "jac-synth-v0.1.0",
            }
        ],
    )

    repair_clean_metadata(tmp_path)

    repaired = json.loads((tmp_path / f"dataset/clean_dataset/debug/{batch_id}.jsonl").read_text().splitlines()[0])
    candidate_audit = audit_candidates(tmp_path, load_clean_candidates(tmp_path))
    assert repaired["compiler_pass"] is True
    assert candidate_audit["failures"] == []


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
    assert report["clusters"][0]["cluster_id"]
    assert report["clusters"][0]["resolution_status"] == "pending"
    assert report["clusters"][0]["action"] == "manual_review_required"
    assert (tmp_path / "dataset/logs/deduplication/jac-synth-v0.1.0-near.json").exists()


def test_near_duplicate_resolution_marks_cluster_resolved_and_unblocks_summary(tmp_path):
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
    cluster_id = report["clusters"][0]["cluster_id"]

    resolution = resolve_near_duplicate_cluster(
        tmp_path,
        version="jac-synth-v0.1.0",
        cluster_id=cluster_id,
        action="keep_distinct",
        reviewer="ayush",
        notes="Same structure but intentionally different entity naming exercise.",
    )
    resolved_report = build_near_duplicate_report(tmp_path, {"code_gen": [first, second]}, version="jac-synth-v0.1.0")

    assert resolution["action"] == "keep_distinct"
    assert resolved_report["unresolved_count"] == 0
    assert resolved_report["clusters"][0]["resolution_status"] == "keep_distinct"


def test_readiness_summary_counts_only_unresolved_near_duplicates(tmp_path):
    records = [
        metadata_record(
            record_id="code_gen-20260511-001-0001",
            batch_id="20260511-code_gen-001",
            category="code_gen",
            extra={"prompt": "Create a walker that sums alpha values.", "code": "node Alpha { has value: int; }"},
        ),
        metadata_record(
            record_id="code_gen-20260511-001-0002",
            batch_id="20260511-code_gen-001",
            category="code_gen",
            extra={"prompt": "Create a walker that sums beta values.", "code": "node Beta { has value: int; }"},
        ),
    ]
    write_batch(tmp_path, category="code_gen", records=records)
    for category in set(ALLOWED_CATEGORIES) - {"code_gen"}:
        write_batch(tmp_path, category=category)
    report = build_near_duplicate_report(tmp_path, {"code_gen": records}, version="jac-synth-v0.1.0")
    resolve_near_duplicate_cluster(
        tmp_path,
        version="jac-synth-v0.1.0",
        cluster_id=report["clusters"][0]["cluster_id"],
        action="keep_distinct",
        reviewer="ayush",
        notes="Same structure but intentionally different entity naming exercise.",
    )

    readiness = build_readiness_summary(tmp_path, version="jac-synth-v0.1.0")

    assert readiness["near_duplicates"]["unresolved_count"] == 0
    assert not any("near duplicate" in reason.lower() for reason in readiness["release_status_reasons"])


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


def test_readiness_summary_lists_blockers_targets_and_latest_scale_decisions(tmp_path):
    for category in ALLOWED_CATEGORIES:
        write_batch(tmp_path, category=category, scale_ready=category != "debug")

    readiness = build_readiness_summary(tmp_path, version="jac-synth-v0.1.0")

    assert readiness["version"] == "jac-synth-v0.1.0"
    assert readiness["status"] == "pilot_only_not_volume_complete"
    assert readiness["counts"]["current_total"] == 5
    assert readiness["counts"]["target_total_range"] == [10_000, 15_000]
    assert readiness["counts"]["by_category"]["code_gen"]["current"] == 1
    assert readiness["manual_review"]["status"] == "complete"
    assert readiness["near_duplicates"]["unresolved_count"] == 0
    assert readiness["latest_scale_decisions"]["debug"]["decision"] == "revise_prompt"
    assert any("Clean example count is below" in blocker for blocker in readiness["blockers"])
    assert "candidate_audit_warning_counts" in readiness["validation"]


def test_readiness_summary_does_not_list_candidate_warnings_as_blockers(tmp_path):
    for category in ALLOWED_CATEGORIES:
        write_batch(tmp_path, category=category)
    batch_id = "20260511-explanation-001"
    review_path = tmp_path / f"dataset/review/explanation/{batch_id}-review.json"
    review_payload = json.loads(review_path.read_text())
    review_payload[0]["review_status"] = "failed"
    review_path.write_text(json.dumps(review_payload))

    readiness = build_readiness_summary(tmp_path, version="jac-synth-v0.1.0")

    assert readiness["validation"]["candidate_audit_status"] == "warning"
    assert "Some clean candidates have nullable test results that require review context." not in readiness["blockers"]


def test_full_release_status_blocks_when_hard_ratios_are_outside_band(tmp_path):
    records = {
        category: [
            metadata_record(
                record_id=f"{category}-20260511-001-{index:04d}",
                batch_id=f"20260511-{category}-001",
                category=category,
                complexity="simple",
            )
            for index in range(1, 3)
        ]
        for category in ALLOWED_CATEGORIES
    }

    audit = {
        "preflight": {"overall_status": "complete"},
        "candidate_audit": {"status": "complete"},
        "manual_review_summary": {"status": "complete"},
        "count_summary": {
            "total": 10_000,
            "by_category": {
                "code_gen": 3_000,
                "debug": 2_000,
                "explanation": 1_000,
                "conversion": 1_000,
                "trajectory": 3_000,
            },
        },
        "hard_example_ratios": {
            category: {"ratio": 0.0, "target_ratio": 0.2}
            for category in ALLOWED_CATEGORIES
        },
    }

    from data_generation.release import _release_status

    assert _release_status(
        audit["preflight"],
        audit["candidate_audit"],
        audit["manual_review_summary"],
        audit["count_summary"],
        False,
        hard_ratios=audit["hard_example_ratios"],
    ) == "blocked"


def test_full_release_status_blocks_unresolved_near_duplicates():
    from data_generation.release import _release_status

    assert _release_status(
        {"overall_status": "complete"},
        {"status": "complete"},
        {"status": "complete"},
        {
            "total": 10_000,
            "by_category": {
                "code_gen": 3_000,
                "debug": 2_000,
                "explanation": 1_000,
                "conversion": 1_000,
                "trajectory": 3_000,
            },
        },
        False,
        hard_ratios={category: {"ratio": 0.2} for category in ALLOWED_CATEGORIES},
        near_report={"unresolved_count": 1},
    ) == "blocked"


def test_known_limitations_skip_hard_ratios_until_volume_is_complete_and_resolved_near_duplicates():
    from data_generation.release import _known_limitations

    limitations = _known_limitations(
        {"overall_status": "complete"},
        {"warnings": []},
        {"status": "complete"},
        {"total": 166, "by_category": {category: 1 for category in ALLOWED_CATEGORIES}},
        {category: {"ratio": 0.0} for category in ALLOWED_CATEGORIES},
        {"flagged_count": 2, "unresolved_count": 0},
    )

    assert limitations == ["Clean example count is below the 10,000 example release minimum."]


def test_readiness_and_audit_cli_write_reports_only_when_requested(tmp_path, monkeypatch, capsys):
    for category in ALLOWED_CATEGORIES:
        write_batch(tmp_path, category=category)
    monkeypatch.chdir(tmp_path)

    assert main(["audit", "--version", "jac-synth-v0.1.0"]) == 0
    assert '"status": "pilot_only_not_volume_complete"' in capsys.readouterr().out
    assert not (tmp_path / "dataset/logs/audit/jac-synth-v0.1.0-audit.json").exists()

    assert main(["audit", "--version", "jac-synth-v0.1.0", "--write-report"]) == 0
    assert (tmp_path / "dataset/logs/audit/jac-synth-v0.1.0-audit.json").exists()

    assert main(["readiness", "--version", "jac-synth-v0.1.0", "--write-report"]) == 0
    readiness_output = capsys.readouterr().out

    assert '"current_total": 5' in readiness_output
    assert (tmp_path / "dataset/logs/audit/jac-synth-v0.1.0-readiness.json").exists()


def test_resolve_near_duplicate_cli_records_decision(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    assert (
        main(
            [
                "resolve-near-duplicate",
                "--version",
                "jac-synth-v0.1.0",
                "--cluster-id",
                "cluster-1",
                "--action",
                "keep_distinct",
                "--reviewer",
                "ayush",
                "--notes",
                "Different task intent.",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "keep_distinct"
    resolutions = json.loads((tmp_path / "dataset/logs/deduplication/jac-synth-v0.1.0-near-resolutions.json").read_text())
    assert resolutions["resolutions"]["cluster-1"]["notes"] == "Different task intent."


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


def test_freeze_release_from_snapshot_writes_checksums_and_refuses_drift(tmp_path):
    for category in ALLOWED_CATEGORIES:
        write_batch(tmp_path, category=category)
    audit = audit_release_readiness(tmp_path, version="jac-synth-v0.1.0", allow_pilot_release=True)
    snapshot_path = write_audit_snapshot(tmp_path, audit)

    release = freeze_release(
        tmp_path,
        version="jac-synth-v0.1.0",
        allow_pilot_release=True,
        audit_snapshot_path=snapshot_path,
    )
    lock = json.loads((tmp_path / "dataset/releases/jac-synth-v0.1.0/IMMUTABLE_RELEASE.json").read_text())

    assert release["status"] == audit["status"]
    assert lock["audit_fingerprint"] == audit["fingerprint"]
    assert "manifest.json" in lock["checksums"]

    write_batch(tmp_path, category="code_gen", sequence=99)
    with pytest.raises(ValueError, match="audit snapshot is stale"):
        freeze_release(
            tmp_path,
            version="jac-synth-v0.1.1",
            allow_pilot_release=True,
            audit_snapshot_path=snapshot_path,
        )


def test_repair_metadata_cli_updates_legacy_clean_files(tmp_path, monkeypatch, capsys):
    ensure_dataset_tree(tmp_path)
    batch_id = "20260511-conversion-004"
    legacy = {
        "id": "conversion-20260511-004-0001",
        "python_code": "print('hi')",
        "jac_code": "valid jac",
        "conversion_notes": "Uses Jac.",
    }
    write_jsonl(tmp_path / f"dataset/clean_dataset/conversion/{batch_id}.jsonl", [legacy])
    write_jsonl(
        tmp_path / f"dataset/logs/validation/{batch_id}.jsonl",
        [
            {
                "batch_id": batch_id,
                "prompt_version": "prompt-conversion-v3",
                "context_bundle_version": "jac-context-v1",
                "category": "conversion",
                "example_id": legacy["id"],
                "json_schema_result": True,
                "compiler_result": [{"passed": True}],
                "test_result": None,
                "rejection_reason": None,
                "retry_count": 0,
                "final_disposition": "clean",
                "validator_version": "validator-v1",
                "dataset_version": "jac-synth-v0.1.0",
            }
        ],
    )
    monkeypatch.chdir(tmp_path)

    assert main(["repair-metadata"]) == 0

    assert '"updated_count": 1' in capsys.readouterr().out
    repaired = json.loads((tmp_path / f"dataset/clean_dataset/conversion/{batch_id}.jsonl").read_text().splitlines()[0])
    assert repaired["category"] == "conversion"
    assert repaired["source_prompt_version"] == "prompt-conversion-v3"
