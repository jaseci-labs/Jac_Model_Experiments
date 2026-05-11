import json

import pytest

from data_generation.trajectory_generation import (
    INITIAL_SFT_CONTEXT_LIMIT_TOKENS,
    TrajectoryPilotRunner,
    build_candidate_task_bank,
    build_trajectory_scale_plan,
    build_trajectory_record,
    ingest_normalized_transcript,
    main,
    count_consecutive_failed_validation_calls,
    estimate_trajectory_tokens,
    normalize_transcript_turns,
    select_pilot_tasks,
)
from data_generation.validation import CompilerResult


def passing_compiler(_code: str) -> CompilerResult:
    return CompilerResult(passed=True, stdout="validation passed")


def failing_compiler(_code: str) -> CompilerResult:
    return CompilerResult(passed=False, error_message="synthetic validation failure")


def valid_turns() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "Build a walker that sums values."},
        {"role": "assistant", "content": "I will consult Jac docs, implement, and validate."},
        {"role": "tool_call", "content": 'user-jac.understand_jac_and_jaseci({})'},
        {"role": "tool_result", "content": "knowledge map"},
        {"role": "tool_call", "content": 'user-jac.get_resource({\"uri\":\"jac://guide/pitfalls\"})'},
        {"role": "tool_result", "content": "pitfalls"},
        {"role": "tool_call", "content": 'user-jac.get_resource({\"uri\":\"jac://guide/patterns\"})'},
        {"role": "tool_result", "content": "patterns"},
        {"role": "assistant", "content": "Here is the Jac code."},
        {"role": "tool_call", "content": 'user-jac.validate_jac({\"code\":\"valid jac\",\"filename\":\"trajectory.jac\"})'},
        {"role": "tool_result", "content": "{\"passed\": true}"},
        {"role": "assistant", "content": "Final answer with validated Jac code."},
    ]


def test_task_bank_has_required_distribution_and_pilot_selection():
    task_bank = build_candidate_task_bank()

    assert len(task_bank) == 15
    assert sum(1 for task in task_bank if task.complexity == "simple") == 5
    assert sum(1 for task in task_bank if task.complexity == "medium") == 7
    assert sum(1 for task in task_bank if task.complexity == "hard") == 3

    selected = select_pilot_tasks(task_bank)

    assert [task.complexity for task in selected] == ["simple", "medium", "hard"]


def test_normalize_transcript_turns_preserves_allowed_roles():
    turns = normalize_transcript_turns(valid_turns())

    assert [turn.role for turn in turns[:4]] == ["user", "assistant", "tool_call", "tool_result"]
    assert turns[0].content == "Build a walker that sums values."


def test_normalize_transcript_turns_rejects_invalid_roles():
    with pytest.raises(ValueError, match="invalid trajectory turn role"):
        normalize_transcript_turns([{"role": "system", "content": "private workspace context"}])


def test_build_trajectory_record_populates_task5_metadata():
    task = select_pilot_tasks(build_candidate_task_bank())[0]
    record = build_trajectory_record(
        task=task,
        turns=normalize_transcript_turns(valid_turns()),
        final_code="valid jac",
        validation_result={"passed": True},
        date="20260511",
        batch_sequence=1,
        example_sequence=1,
        generation_date="2026-05-11T00:00:00Z",
    )

    assert record["id"] == "trajectory-20260511-001-0001"
    assert record["batch_id"] == "20260511-trajectory-001"
    assert record["category"] == "trajectory"
    assert record["generator"] == "cursor-jac-mcp"
    assert record["final_output"]["validation_tool"] == "user-jac.validate_jac"
    assert record["turns"][0]["role"] == "user"


def test_token_estimate_counts_all_turn_content():
    turns = normalize_transcript_turns([{"role": "user", "content": "abcd" * 10}])

    assert estimate_trajectory_tokens(turns) == 10
    assert estimate_trajectory_tokens(turns) < INITIAL_SFT_CONTEXT_LIMIT_TOKENS


def test_counts_consecutive_failed_validation_calls():
    turns = normalize_transcript_turns(
        [
            {"role": "tool_call", "content": 'user-jac.validate_jac({"code":"bad"})'},
            {"role": "tool_result", "content": '{"passed": false}'},
            {"role": "tool_call", "content": 'user-jac.validate_jac({"code":"still bad"})'},
            {"role": "tool_result", "content": "error: failed"},
            {"role": "tool_call", "content": 'user-jac.validate_jac({"code":"valid"})'},
            {"role": "tool_result", "content": '{"passed": true}'},
        ]
    )

    assert count_consecutive_failed_validation_calls(turns) == 2


def test_runner_writes_clean_rejected_validation_generation_and_review_artifacts(tmp_path):
    tasks = select_pilot_tasks(build_candidate_task_bank())
    runner = TrajectoryPilotRunner(workspace_root=tmp_path, compiler=passing_compiler)
    records = [
        {
            "task": tasks[0],
            "turns": valid_turns(),
            "final_code": "valid jac",
            "validation_result": {"passed": True},
        },
        {
            "task": tasks[1],
            "turns": valid_turns(),
            "final_code": "valid jac",
            "validation_result": {"passed": False},
        },
    ]

    summary = runner.write_pilot_batch(
        date="20260511",
        sequence=1,
        raw_sessions=records,
        generation_date="2026-05-11T00:00:00Z",
    )

    assert summary.clean_count == 1
    assert summary.rejected_count == 1
    raw = tmp_path / "dataset/raw_output/trajectory/20260511-trajectory-001.json"
    clean = tmp_path / "dataset/clean_dataset/trajectory/20260511-trajectory-001.jsonl"
    rejected = tmp_path / "dataset/rejected/trajectory/20260511-trajectory-001.jsonl"
    validation = tmp_path / "dataset/logs/validation/20260511-trajectory-001.jsonl"
    generation = tmp_path / "dataset/logs/generation/20260511-trajectory-001.json"
    review = tmp_path / "dataset/review/trajectory/20260511-trajectory-001-review.json"

    assert raw.exists()
    assert clean.exists()
    assert rejected.exists()
    assert validation.exists()
    assert generation.exists()
    assert review.exists()
    assert json.loads(generation.read_text())["ready_for_volume"] is False
    assert json.loads(generation.read_text())["artifact_paths"]["generation"].endswith(
        "dataset/logs/generation/20260511-trajectory-001.json"
    )


def test_runner_rejects_missing_required_mcp_docs(tmp_path):
    task = select_pilot_tasks(build_candidate_task_bank())[0]
    turns = [turn for turn in valid_turns() if "jac://guide/patterns" not in turn["content"]]
    runner = TrajectoryPilotRunner(workspace_root=tmp_path, compiler=passing_compiler)

    summary = runner.write_pilot_batch(
        date="20260511",
        sequence=1,
        raw_sessions=[
            {
                "task": task,
                "turns": turns,
                "final_code": 'with entry { print("hi"); }',
                "validation_result": {"passed": True},
            }
        ],
        generation_date="2026-05-11T00:00:00Z",
    )

    assert summary.clean_count == 0
    assert summary.rejected_count == 1
    rejected = tmp_path / "dataset/rejected/trajectory/20260511-trajectory-001.jsonl"
    record = json.loads(rejected.read_text().splitlines()[0])
    assert record["rejection_reason"] == "missing required Jac MCP context fetch"


def test_runner_rejects_more_than_three_failed_validation_attempts(tmp_path):
    task = select_pilot_tasks(build_candidate_task_bank())[0]
    turns = [
        {"role": "user", "content": "Build Jac."},
        {"role": "assistant", "content": "I will validate."},
        {"role": "tool_call", "content": 'user-jac.understand_jac_and_jaseci({})'},
        {"role": "tool_result", "content": "knowledge map"},
        {"role": "tool_call", "content": 'user-jac.get_resource({\"uri\":\"jac://guide/pitfalls\"})'},
        {"role": "tool_result", "content": "pitfalls"},
        {"role": "tool_call", "content": 'user-jac.get_resource({\"uri\":\"jac://guide/patterns\"})'},
        {"role": "tool_result", "content": "patterns"},
    ]
    for index in range(4):
        turns.extend(
            [
                {"role": "tool_call", "content": f'user-jac.validate_jac({{\"code\":\"bad {index}\"}})'},
                {"role": "tool_result", "content": '{"passed": false}'},
            ]
        )

    runner = TrajectoryPilotRunner(workspace_root=tmp_path, compiler=passing_compiler)
    summary = runner.write_pilot_batch(
        date="20260511",
        sequence=1,
        raw_sessions=[
            {
                "task": task,
                "turns": turns,
                "final_code": 'with entry { print("hi"); }',
                "validation_result": {"passed": True},
            }
        ],
        generation_date="2026-05-11T00:00:00Z",
    )

    assert summary.clean_count == 0
    rejected = tmp_path / "dataset/rejected/trajectory/20260511-trajectory-001.jsonl"
    record = json.loads(rejected.read_text().splitlines()[0])
    assert record["rejection_reason"] == "more than three consecutive compiler failures"


def test_ingest_normalized_transcript_writes_validated_trajectory_batch(tmp_path):
    task = select_pilot_tasks(build_candidate_task_bank())[0].to_dict()
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "task": task,
                "turns": valid_turns(),
                "final_code": 'with entry { print("hi"); }',
                "validation_result": {"passed": True},
            }
        )
    )

    summary = ingest_normalized_transcript(
        tmp_path,
        input_path=transcript_path,
        date="20260511",
        sequence=2,
        generation_date="2026-05-11T00:00:00Z",
        compiler=passing_compiler,
    )

    assert summary.clean_count == 1
    assert (tmp_path / "dataset/clean_dataset/trajectory/20260511-trajectory-002.jsonl").exists()


def test_ingest_normalized_transcript_rejects_secret_like_content(tmp_path):
    task = select_pilot_tasks(build_candidate_task_bank())[0].to_dict()
    transcript_path = tmp_path / "transcript.json"
    turns = valid_turns()
    turns.append({"role": "assistant", "content": "OPENAI_API_KEY=sk-secret"})
    transcript_path.write_text(
        json.dumps(
            {
                "task": task,
                "turns": turns,
                "final_code": "valid jac",
                "validation_result": {"passed": True},
            }
        )
    )

    summary = ingest_normalized_transcript(
        tmp_path,
        input_path=transcript_path,
        date="20260511",
        sequence=2,
        generation_date="2026-05-11T00:00:00Z",
        compiler=passing_compiler,
    )

    assert summary.clean_count == 0
    rejected = tmp_path / "dataset/rejected/trajectory/20260511-trajectory-002.jsonl"
    assert json.loads(rejected.read_text().splitlines()[0])["rejection_reason"] == "trajectory contains secret-like content"


def test_trajectory_scale_plan_rotates_tasks_and_preserves_hard_ratio():
    plan = build_trajectory_scale_plan(target_count=10, existing_count=3)

    assert plan["missing_count"] == 7
    assert len(plan["tasks"]) == 7
    assert 0.15 <= plan["hard_ratio"] <= 0.35


def test_trajectory_cli_plan_and_ingest(tmp_path, monkeypatch, capsys):
    task = select_pilot_tasks(build_candidate_task_bank())[0].to_dict()
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "task": task,
                "turns": valid_turns(),
                "final_code": 'with entry { print("hi"); }',
                "validation_result": {"passed": True},
            }
        )
    )
    monkeypatch.chdir(tmp_path)

    assert main(["plan", "--target-count", "10", "--existing-count", "3"]) == 0
    assert '"missing_count": 7' in capsys.readouterr().out
    assert (
        main(
            [
                "ingest",
                "--input",
                str(transcript_path),
                "--date",
                "20260511",
                "--sequence",
                "3",
                "--generation-date",
                "2026-05-11T00:00:00Z",
            ]
        )
        == 0
    )
    assert '"clean_count": 1' in capsys.readouterr().out
