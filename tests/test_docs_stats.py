import json
from pathlib import Path

from data_generation.artifacts import ensure_dataset_tree, write_json, write_jsonl
from data_generation.docs_stats import build_stats_snapshot, main, render_stats_markdown


def metadata_record(
    *,
    record_id: str,
    batch_id: str,
    category: str,
    complexity: str = "simple",
    test_pass: bool | None = True,
) -> dict:
    record = {
        "id": record_id,
        "batch_id": batch_id,
        "category": category,
        "complexity": complexity,
        "compiler_pass": True,
        "test_pass": test_pass,
        "manually_reviewed": False,
        "generator": "cursor-jac-mcp" if category == "trajectory" else "openai-api",
        "generation_date": "2026-05-11T00:00:00Z",
        "source_prompt_version": "trajectory-prompt-v1" if category == "trajectory" else f"prompt-{category}-v1",
        "context_bundle_version": "jac-context-v1",
        "validator_version": "jac-mcp-validate-v1" if category == "trajectory" else "validator-v1",
        "dataset_version": "jac-synth-v0.1.0",
        "review_status": "pending",
    }
    if category == "code_gen":
        record.update({"prompt": f"Build a Jac node for {record_id}.", "code": f"node Item_{record_id.replace('-', '_')} {{ has value: int; }}"})
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
        record.update({"code": "node Item { has value: int; }", "granularity": "block", "explanation": "Defines a node."})
    elif category == "conversion":
        record.update({"python_code": "class Item:\n    pass", "jac_code": "node Item {}", "conversion_notes": "Converted class to node."})
    elif category == "trajectory":
        record.update(
            {
                "trajectory_length_tokens": 100,
                "task": {"prompt": "Build a Jac node.", "difficulty_reason": "Small model."},
                "final_output": {"language": "jac", "code": "node Item {}", "validation_tool": "user-jac.validate_jac", "validation_result": {"passed": True}},
                "turns": [
                    {"role": "user", "content": "Build a Jac node."},
                    {"role": "assistant", "content": "Done."},
                    {"role": "tool_call", "content": "user-jac.validate_jac({})"},
                    {"role": "tool_result", "content": '{"passed": true}'},
                ],
            }
        )
    return record


def write_batch(root: Path, *, category: str, clean_count: int = 1, rejected_count: int = 0) -> None:
    ensure_dataset_tree(root)
    batch_id = f"20260511-{category}-001"
    clean_records = [
        metadata_record(
            record_id=f"{category}-20260511-001-{index + 1:04d}",
            batch_id=batch_id,
            category=category,
            complexity="hard" if index == 0 else "simple",
            test_pass=None if category in {"explanation", "trajectory"} else True,
        )
        for index in range(clean_count)
    ]
    write_json(root / f"dataset/raw_output/{category}/{batch_id}.json", {"batch_id": batch_id, "examples": clean_records})
    write_jsonl(root / f"dataset/clean_dataset/{category}/{batch_id}.jsonl", clean_records)
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
                "compiler_result": [{"passed": True, "expected_to_compile": True}],
                "test_result": record["test_pass"],
                "rejection_reason": None,
                "retry_count": 0,
                "final_disposition": "clean",
                "validator_version": record["validator_version"],
                "dataset_version": record["dataset_version"],
            }
            for record in clean_records
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
                "criteria_results": {},
                "notes": "ok",
            }
            for record in clean_records
        ],
    )
    if rejected_count:
        rejected = [
            {
                **metadata_record(
                    record_id=f"{category}-20260511-001-r{index + 1:04d}",
                    batch_id=batch_id,
                    category=category,
                ),
                "compiler_pass": False,
                "review_status": "rejected",
                "rejection_reason": "compiler failure",
            }
            for index in range(rejected_count)
        ]
        write_jsonl(root / f"dataset/rejected/{category}/{batch_id}.jsonl", rejected)
    if category != "trajectory":
        write_json(root / f"dataset/logs/scale_decisions/{batch_id}.json", {"batch_id": batch_id, "decision": "scale_ready", "reasons": []})
    else:
        write_json(root / f"dataset/logs/generation/{batch_id}.json", {"batch_id": batch_id, "ready_for_volume": True})


def test_build_stats_snapshot_counts_dataset_artifacts(tmp_path):
    write_batch(tmp_path, category="code_gen", clean_count=2, rejected_count=1)
    write_batch(tmp_path, category="debug", clean_count=1)

    snapshot = build_stats_snapshot(tmp_path, version="jac-synth-v0.1.0")

    assert snapshot["version"] == "jac-synth-v0.1.0"
    assert snapshot["clean_total"] == 3
    assert snapshot["rejected_total"] == 1
    assert snapshot["categories"]["code_gen"]["clean_count"] == 2
    assert snapshot["categories"]["code_gen"]["rejected_count"] == 1
    assert snapshot["categories"]["debug"]["review_file_count"] == 1
    assert snapshot["readiness"]["counts"]["current_total"] == 3


def test_render_stats_markdown_is_deterministic_and_links_docs(tmp_path):
    write_batch(tmp_path, category="code_gen", clean_count=1)
    snapshot = build_stats_snapshot(tmp_path, version="jac-synth-v0.1.0", generated_at="2026-05-11T00:00:00Z")

    markdown = render_stats_markdown(snapshot)

    assert markdown.startswith("# Dataset Stats\n")
    assert "Generated from repository artifacts at `2026-05-11T00:00:00Z`." in markdown
    assert "| `code_gen` | 1 | 0 | 1 | 1 | 3,000-5,000 |" in markdown
    assert "[release readiness](operations.md#release-readiness)" in markdown
    assert markdown.endswith("\n")


def test_main_writes_stats_markdown(tmp_path):
    write_batch(tmp_path, category="code_gen", clean_count=1)
    output = tmp_path / "docs" / "stats.md"

    exit_code = main(["--workspace-root", str(tmp_path), "--version", "jac-synth-v0.1.0", "--output", str(output), "--generated-at", "2026-05-11T00:00:00Z"])

    assert exit_code == 0
    assert output.exists()
    assert "`jac-synth-v0.1.0`" in output.read_text()
