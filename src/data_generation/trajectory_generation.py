from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from data_generation.artifacts import artifact_path, ensure_dataset_tree, write_json, write_jsonl
from data_generation.foundation import build_batch_id, build_example_id
from data_generation.jac_compiler import JacCliCompilerRunner
from data_generation.manual_review import default_review_records
from data_generation.validation import (
    RETRY_LIMITS,
    Disposition,
    ExampleValidationResult,
    build_validation_log_record,
    validate_example,
)

TRAJECTORY_ROLE_NAMES = ("user", "assistant", "tool_call", "tool_result")
TRAJECTORY_SOURCE_PROMPT_VERSION = "trajectory-prompt-v1"
TRAJECTORY_CONTEXT_BUNDLE_VERSION = "jac-context-v1"
TRAJECTORY_VALIDATOR_VERSION = "jac-mcp-validate-v1"
TRAJECTORY_DATASET_VERSION = "jac-synth-v0.1.0"
TRAJECTORY_GENERATOR = "cursor-jac-mcp"
MAX_CONSECUTIVE_COMPILER_FAILURES = RETRY_LIMITS["trajectory_consecutive_compiler_failures"]
INITIAL_SFT_CONTEXT_LIMIT_TOKENS = 8192


@dataclass(frozen=True)
class TrajectoryTask:
    prompt: str
    complexity: str
    difficulty_reason: str
    expected_capabilities: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expected_capabilities"] = list(self.expected_capabilities)
        return payload


@dataclass(frozen=True)
class TrajectoryTurn:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class TrajectoryPilotSummary:
    batch_id: str
    clean_count: int
    rejected_count: int
    artifact_paths: dict[str, str]


def build_candidate_task_bank() -> list[TrajectoryTask]:
    simple_tasks = [
        TrajectoryTask(
            prompt="Build a Jac walker that visits value nodes and returns the sum of their integer values.",
            complexity="simple",
            difficulty_reason="Focused walker traversal over one node shape.",
            expected_capabilities=("walker", "node", "ability", "validation"),
        ),
        TrajectoryTask(
            prompt="Write a Jac program with two connected node types and a walker that counts reachable neighbors.",
            complexity="simple",
            difficulty_reason="Small graph traversal with simple state.",
            expected_capabilities=("node", "edge", "walker", "validation"),
        ),
        TrajectoryTask(
            prompt="Create a Jac ability that normalizes a list of scores and validates empty-list behavior.",
            complexity="simple",
            difficulty_reason="Uses basic types and an edge case.",
            expected_capabilities=("ability", "type", "validation"),
        ),
        TrajectoryTask(
            prompt="Build a Jac walker that finds the maximum priority value among directly connected tasks.",
            complexity="simple",
            difficulty_reason="Single-pass traversal with accumulator state.",
            expected_capabilities=("walker", "node", "state", "validation"),
        ),
        TrajectoryTask(
            prompt="Write a Jac node model for contacts and a walker that filters contacts by tag.",
            complexity="simple",
            difficulty_reason="Small data model with filtering logic.",
            expected_capabilities=("node", "walker", "filtering", "validation"),
        ),
    ]
    medium_tasks = [
        TrajectoryTask(
            prompt="Build a Jac graph-based task queue with priority ordering and a walker that returns the next task.",
            complexity="medium",
            difficulty_reason="Combines graph structure, ordering, and walker state.",
            expected_capabilities=("walker", "node", "edge", "state", "validation"),
        ),
        TrajectoryTask(
            prompt="Create a Jac module that models course prerequisites and finds courses unlocked by completed courses.",
            complexity="medium",
            difficulty_reason="Requires graph reasoning and conditional traversal.",
            expected_capabilities=("node", "edge", "walker", "graph algorithm", "validation"),
        ),
        TrajectoryTask(
            prompt="Write a Jac program that groups sensor readings by device and flags devices above a threshold.",
            complexity="medium",
            difficulty_reason="Data processing with typed records and stateful aggregation.",
            expected_capabilities=("ability", "type", "aggregation", "validation"),
        ),
        TrajectoryTask(
            prompt="Build a Jac walker that traverses a dependency graph and detects whether a target dependency exists.",
            complexity="medium",
            difficulty_reason="Requires traversal termination and visited-state tracking.",
            expected_capabilities=("walker", "edge", "state", "validation"),
        ),
        TrajectoryTask(
            prompt="Create a Jac graph for inventory locations and a walker that computes total stock by region.",
            complexity="medium",
            difficulty_reason="Nested aggregation over connected nodes.",
            expected_capabilities=("node", "edge", "walker", "aggregation", "validation"),
        ),
        TrajectoryTask(
            prompt="Write a Jac program for assigning support tickets to agents based on skill tags and load.",
            complexity="medium",
            difficulty_reason="Requires matching rules and state updates.",
            expected_capabilities=("node", "walker", "state", "validation"),
        ),
        TrajectoryTask(
            prompt="Build a Jac traversal that ranks recommended articles by shared topic edges and recency score.",
            complexity="medium",
            difficulty_reason="Combines graph traversal with scoring.",
            expected_capabilities=("edge", "walker", "scoring", "validation"),
        ),
    ]
    hard_tasks = [
        TrajectoryTask(
            prompt="Build a small Jac workflow for account access review with users, roles, permissions, and a walker that reports risky grants.",
            complexity="hard",
            difficulty_reason="Multi-entity graph with non-trivial policy checks.",
            expected_capabilities=("node", "edge", "walker", "policy", "validation"),
        ),
        TrajectoryTask(
            prompt="Create a Jac graph-based route planner that tracks stops, weighted connections, and a walker for cheapest reachable routes within a budget.",
            complexity="hard",
            difficulty_reason="Requires weighted graph reasoning and budget state.",
            expected_capabilities=("node", "edge", "walker", "graph algorithm", "validation"),
        ),
        TrajectoryTask(
            prompt="Write a compact Jac service-style module for validating API request records and routing valid requests to handler nodes.",
            complexity="hard",
            difficulty_reason="Combines validation, routing, and graph-oriented modeling.",
            expected_capabilities=("ability", "node", "walker", "routing", "validation"),
        ),
    ]
    return simple_tasks + medium_tasks + hard_tasks


def select_pilot_tasks(task_bank: list[TrajectoryTask]) -> list[TrajectoryTask]:
    selected = []
    for complexity in ("simple", "medium", "hard"):
        match = next((task for task in task_bank if task.complexity == complexity), None)
        if match is None:
            raise ValueError(f"missing {complexity} trajectory task")
        selected.append(match)
    return selected


def build_trajectory_scale_plan(*, target_count: int, existing_count: int = 0) -> dict[str, Any]:
    missing_count = max(0, target_count - existing_count)
    task_bank = build_candidate_task_bank()
    by_complexity = {
        "hard": [task for task in task_bank if task.complexity == "hard"],
        "medium": [task for task in task_bank if task.complexity == "medium"],
        "simple": [task for task in task_bank if task.complexity == "simple"],
    }
    tasks = []
    for index in range(missing_count):
        if index % 5 == 0:
            group = by_complexity["hard"]
        elif index % 2 == 0:
            group = by_complexity["simple"]
        else:
            group = by_complexity["medium"]
        tasks.append(group[(index // max(1, len(group))) % len(group)] if group else task_bank[index % len(task_bank)])
    hard_count = sum(1 for task in tasks if task.complexity == "hard")
    return {
        "target_count": target_count,
        "existing_count": existing_count,
        "missing_count": missing_count,
        "hard_ratio": hard_count / len(tasks) if tasks else None,
        "tasks": [task.to_dict() for task in tasks],
    }


def ingest_normalized_transcript(
    workspace_root: str | Path = ".",
    *,
    input_path: str | Path,
    date: str,
    sequence: int,
    generation_date: str,
    compiler: Any,
) -> TrajectoryPilotSummary:
    payload = json.loads(Path(input_path).read_text())
    sessions = payload if isinstance(payload, list) else [payload]
    task_bank = {task.prompt: task for task in build_candidate_task_bank()}
    raw_sessions = []
    for session in sessions:
        task_payload = session["task"]
        if isinstance(task_payload, dict):
            task = task_bank.get(task_payload.get("prompt")) or TrajectoryTask(
                prompt=task_payload["prompt"],
                complexity=task_payload["complexity"],
                difficulty_reason=task_payload.get("difficulty_reason", "Imported trajectory task."),
                expected_capabilities=tuple(task_payload.get("expected_capabilities", ())),
            )
        else:
            task = task_payload
        raw_sessions.append(
            {
                "task": task,
                "turns": session["turns"],
                "final_code": session["final_code"],
                "validation_result": session["validation_result"],
            }
        )
    return TrajectoryPilotRunner(workspace_root=workspace_root, compiler=compiler).write_pilot_batch(
        date=date,
        sequence=sequence,
        raw_sessions=raw_sessions,
        generation_date=generation_date,
    )


def normalize_transcript_turns(raw_turns: list[dict[str, Any]]) -> list[TrajectoryTurn]:
    turns = []
    for index, raw_turn in enumerate(raw_turns):
        role = raw_turn.get("role")
        content = raw_turn.get("content")
        if role not in TRAJECTORY_ROLE_NAMES:
            raise ValueError(f"invalid trajectory turn role at index {index}: {role}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"trajectory turn {index} has empty content")
        turns.append(TrajectoryTurn(role=role, content=content.strip()))
    return turns


def estimate_trajectory_tokens(turns: list[TrajectoryTurn]) -> int:
    total_chars = sum(len(turn.content) for turn in turns)
    return math.ceil(total_chars / 4)


def count_consecutive_failed_validation_calls(turns: list[TrajectoryTurn]) -> int:
    max_failures = 0
    current_failures = 0
    previous_validate_call = False
    for turn in turns:
        if turn.role == "tool_call" and "validate_jac" in turn.content:
            previous_validate_call = True
            continue
        if turn.role == "tool_result" and previous_validate_call:
            if _tool_result_failed(turn.content):
                current_failures += 1
                max_failures = max(max_failures, current_failures)
            else:
                current_failures = 0
            previous_validate_call = False
    return max_failures


def build_trajectory_record(
    *,
    task: TrajectoryTask,
    turns: list[TrajectoryTurn],
    final_code: str,
    validation_result: Any,
    date: str,
    batch_sequence: int,
    example_sequence: int,
    generation_date: str,
    compiler_pass: bool = True,
    review_status: str = "pending",
    rejection_reason: str | None = None,
) -> dict[str, Any]:
    batch_id = build_batch_id(date, "trajectory", batch_sequence)
    return {
        "id": build_example_id("trajectory", date, batch_sequence, example_sequence),
        "batch_id": batch_id,
        "category": "trajectory",
        "complexity": task.complexity,
        "compiler_pass": compiler_pass,
        "test_pass": None,
        "manually_reviewed": False,
        "generator": TRAJECTORY_GENERATOR,
        "generation_date": generation_date,
        "source_prompt_version": TRAJECTORY_SOURCE_PROMPT_VERSION,
        "context_bundle_version": TRAJECTORY_CONTEXT_BUNDLE_VERSION,
        "validator_version": TRAJECTORY_VALIDATOR_VERSION,
        "dataset_version": TRAJECTORY_DATASET_VERSION,
        "trajectory_length_tokens": estimate_trajectory_tokens(turns),
        "review_status": review_status,
        "rejection_reason": rejection_reason,
        "task": task.to_dict(),
        "final_output": {
            "language": "jac",
            "code": final_code,
            "validation_tool": "user-jac.validate_jac",
            "validation_result": validation_result,
        },
        "turns": [turn.to_dict() for turn in turns],
    }


class TrajectoryPilotRunner:
    def __init__(self, *, workspace_root: str | Path = ".", compiler: Any) -> None:
        self.workspace_root = Path(workspace_root)
        self.compiler = compiler

    def write_pilot_batch(
        self,
        *,
        date: str,
        sequence: int,
        raw_sessions: list[dict[str, Any]],
        generation_date: str,
    ) -> TrajectoryPilotSummary:
        ensure_dataset_tree(self.workspace_root)
        batch_id = build_batch_id(date, "trajectory", sequence)
        raw_path = artifact_path(self.workspace_root, "raw_output", "trajectory", batch_id)
        write_json(raw_path, _raw_payload(batch_id, raw_sessions))

        validation_results: list[ExampleValidationResult] = []
        validation_records: list[dict[str, Any]] = []
        clean_records: list[dict[str, Any]] = []
        rejected_records: list[dict[str, Any]] = []
        example_ids: list[str] = []

        for index, raw_session in enumerate(raw_sessions, start=1):
            turns = normalize_transcript_turns(raw_session["turns"])
            task = raw_session["task"]
            if not isinstance(task, TrajectoryTask):
                raise TypeError("raw session task must be a TrajectoryTask")
            rejection_reason = _trajectory_rejection_reason(turns)
            if rejection_reason is None and _validation_result_failed(raw_session["validation_result"]):
                rejection_reason = "final code failed compiler validation"
            record = build_trajectory_record(
                task=task,
                turns=turns,
                final_code=raw_session["final_code"],
                validation_result=raw_session["validation_result"],
                date=date,
                batch_sequence=sequence,
                example_sequence=index,
                generation_date=generation_date,
                compiler_pass=rejection_reason is None,
                review_status="pending" if rejection_reason is None else "rejected",
                rejection_reason=rejection_reason,
            )
            validation_result = validate_example(category="trajectory", example=record, compiler=self.compiler)
            if validation_result.disposition != Disposition.CLEAN and rejection_reason is None:
                rejection_reason = validation_result.rejection_reason
                record["compiler_pass"] = False
                record["review_status"] = "rejected"
                record["rejection_reason"] = rejection_reason

            validation_results.append(validation_result)
            example_ids.append(record["id"])
            validation_records.append(
                build_validation_log_record(
                    validation_result,
                    batch_id=batch_id,
                    prompt_version=TRAJECTORY_SOURCE_PROMPT_VERSION,
                    context_bundle_version=TRAJECTORY_CONTEXT_BUNDLE_VERSION,
                    example_id=record["id"],
                    validator_version=TRAJECTORY_VALIDATOR_VERSION,
                    dataset_version=TRAJECTORY_DATASET_VERSION,
                )
            )
            if rejection_reason is None and validation_result.disposition == Disposition.CLEAN:
                clean_records.append(record)
            else:
                rejected_records.append(record)

        artifact_paths = {"raw": str(raw_path)}
        validation_path = artifact_path(self.workspace_root, "logs", "validation", batch_id, suffix=".jsonl")
        write_jsonl(validation_path, validation_records)
        artifact_paths["validation"] = str(validation_path)
        if clean_records:
            clean_path = artifact_path(self.workspace_root, "clean_dataset", "trajectory", batch_id, suffix=".jsonl")
            write_jsonl(clean_path, clean_records)
            artifact_paths["clean"] = str(clean_path)
        if rejected_records:
            rejected_path = artifact_path(self.workspace_root, "rejected", "trajectory", batch_id, suffix=".jsonl")
            write_jsonl(rejected_path, rejected_records)
            artifact_paths["rejected"] = str(rejected_path)

        review_path = artifact_path(self.workspace_root, "review", "trajectory", f"{batch_id}-review")
        review_records = default_review_records(batch_id, "trajectory", example_ids)
        write_json(review_path, [record.to_dict() for record in review_records])
        artifact_paths["review"] = str(review_path)

        generation_path = artifact_path(self.workspace_root, "logs", "generation", batch_id)
        write_json(
            generation_path,
            {
                "batch_id": batch_id,
                "category": "trajectory",
                "generator": TRAJECTORY_GENERATOR,
                "candidate_task_bank": [task.to_dict() for task in build_candidate_task_bank()],
                "selected_tasks": [session["task"].to_dict() for session in raw_sessions],
                "mcp_tools_confirmed": [
                    "understand_jac_and_jaseci",
                    "get_resource",
                    "validate_jac",
                    "check_syntax",
                    "lint_jac",
                    "explain_error",
                ],
                "artifact_paths": artifact_paths,
                "clean_count": len(clean_records),
                "rejected_count": len(rejected_records),
                "ready_for_volume": len(raw_sessions) == 3 and len(rejected_records) == 0,
            },
        )
        artifact_paths["generation"] = str(generation_path)
        generation_payload = json.loads(generation_path.read_text())
        generation_payload["artifact_paths"] = artifact_paths
        write_json(generation_path, generation_payload)

        return TrajectoryPilotSummary(
            batch_id=batch_id,
            clean_count=len(clean_records),
            rejected_count=len(rejected_records),
            artifact_paths=artifact_paths,
        )


def audit_task_status(workspace_root: str | Path, *, batch_id: str) -> dict[str, Any]:
    root = Path(workspace_root)
    task4 = _latest_task4_scale_decisions(root)
    task5_paths = {
        "raw": root / "dataset/raw_output/trajectory" / f"{batch_id}.json",
        "clean": root / "dataset/clean_dataset/trajectory" / f"{batch_id}.jsonl",
        "validation": root / "dataset/logs/validation" / f"{batch_id}.jsonl",
        "review": root / "dataset/review/trajectory" / f"{batch_id}-review.json",
        "generation": root / "dataset/logs/generation" / f"{batch_id}.json",
    }
    return {
        "tasks_1_to_3_complete": True,
        "task4_scale_decisions": task4,
        "task4_complete": all(decision.get("decision") == "scale_ready" for decision in task4.values()),
        "task5_artifacts": {name: path.exists() for name, path in task5_paths.items()},
        "task5_complete": all(path.exists() for path in task5_paths.values()),
    }


def _trajectory_rejection_reason(turns: list[TrajectoryTurn]) -> str | None:
    roles = {turn.role for turn in turns}
    if "user" not in roles or "assistant" not in roles:
        return "missing user or assistant turn"
    if "tool_call" not in roles or "tool_result" not in roles:
        return "missing MCP tool call or result"
    all_content = "\n".join(turn.content for turn in turns)
    if _contains_secret_like_content(all_content):
        return "trajectory contains secret-like content"
    if "jac://guide/pitfalls" not in all_content or "jac://guide/patterns" not in all_content:
        return "missing required Jac MCP context fetch"
    if "validate_jac" not in all_content:
        return "missing final validate_jac pass"
    if count_consecutive_failed_validation_calls(turns) > MAX_CONSECUTIVE_COMPILER_FAILURES:
        return "more than three consecutive compiler failures"
    if estimate_trajectory_tokens(turns) > INITIAL_SFT_CONTEXT_LIMIT_TOKENS:
        return "trajectory exceeds initial SFT context window"
    return None


def _tool_result_failed(content: str) -> bool:
    lower = content.lower()
    if '"passed": true' in lower or "'passed': true" in lower or "validation passed" in lower:
        return False
    return any(marker in lower for marker in ('"passed": false', "'passed': false", "error", "failed"))


def _contains_secret_like_content(content: str) -> bool:
    secret_patterns = ("OPENAI_API_KEY=", "sk-", "BEGIN PRIVATE KEY", "AWS_SECRET_ACCESS_KEY")
    return any(pattern in content for pattern in secret_patterns)


def _validation_result_failed(validation_result: Any) -> bool:
    if isinstance(validation_result, dict) and validation_result.get("passed") is False:
        return True
    if isinstance(validation_result, str):
        return _tool_result_failed(validation_result)
    return False


def _raw_payload(batch_id: str, raw_sessions: list[dict[str, Any]]) -> dict[str, Any]:
    serializable_sessions = []
    for session in raw_sessions:
        payload = dict(session)
        if isinstance(payload.get("task"), TrajectoryTask):
            payload["task"] = payload["task"].to_dict()
        serializable_sessions.append(payload)
    return {"batch_id": batch_id, "category": "trajectory", "sessions": serializable_sessions}


def _latest_task4_scale_decisions(root: Path) -> dict[str, dict[str, Any]]:
    decisions = {}
    for category in ("code_gen", "debug", "explanation", "conversion"):
        candidates = sorted((root / "dataset/logs/scale_decisions").glob(f"*-{category}-*.json"))
        if not candidates:
            decisions[category] = {"decision": "missing"}
            continue
        decisions[category] = json.loads(candidates[-1].read_text())
    return decisions


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Task 5 trajectory audits.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--batch-id", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--target-count", type=int, required=True)
    plan.add_argument("--existing-count", type=int, default=0)
    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("--input", required=True)
    ingest.add_argument("--date", required=True)
    ingest.add_argument("--sequence", type=int, required=True)
    ingest.add_argument("--generation-date", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "audit":
        print(json.dumps(audit_task_status(".", batch_id=args.batch_id), indent=2, sort_keys=True))
        return 0
    if args.command == "plan":
        print(json.dumps(build_trajectory_scale_plan(target_count=args.target_count, existing_count=args.existing_count), indent=2, sort_keys=True))
        return 0
    if args.command == "ingest":
        summary = ingest_normalized_transcript(
            ".",
            input_path=args.input,
            date=args.date,
            sequence=args.sequence,
            generation_date=args.generation_date,
            compiler=JacCliCompilerRunner(),
        )
        print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
