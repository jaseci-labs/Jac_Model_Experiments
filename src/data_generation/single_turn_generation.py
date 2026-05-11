from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_generation.artifacts import artifact_path, ensure_dataset_tree, write_json, write_jsonl
from data_generation.context_bundle import DEFAULT_CONTEXT_BUNDLE_VERSION, load_context_bundle, write_context_bundle
from data_generation.foundation import SCRIPTED_CATEGORIES, build_batch_id, build_example_id, build_validator_version
from data_generation.jac_compiler import JacCliCompilerRunner
from data_generation.manual_review import ScaleDecisionStatus, build_prompt_revision_record, decide_scale_up, default_review_records
from data_generation.openai_generation import GenerationResult, OpenAIGenerationClient, OpenAISettings
from data_generation.prompt_design import build_prompt_request
from data_generation.validation import (
    Disposition,
    build_validation_log_record,
    parse_json_batch,
    validate_batch,
)


@dataclass(frozen=True)
class PilotSummary:
    batch_id: str
    category: str
    clean_count: int
    rejected_count: int
    review_count: int
    discarded_count: int
    artifact_paths: dict[str, str]


class FakeGenerationClient:
    def __init__(self, responses: dict[str, list[dict[str, Any]] | str]) -> None:
        self.responses = responses

    def generate_batch(self, prompt_request: dict[str, Any]) -> GenerationResult:
        response = self.responses[prompt_request["category"]]
        return GenerationResult(
            examples=response,
            raw_response={"category": prompt_request["category"], "response": response},
        )


class SingleTurnPilotRunner:
    def __init__(
        self,
        *,
        workspace_root: str | Path = ".",
        generation_client: Any,
        compiler: Any,
        context_bundle: str,
        model: str = "gpt-5.5",
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.generation_client = generation_client
        self.compiler = compiler
        self.context_bundle = context_bundle
        self.model = model

    def run_category_pilot(
        self,
        *,
        category: str,
        date: str,
        sequence: int,
        prompt_version_number: int = 1,
        context_bundle_version: str = DEFAULT_CONTEXT_BUNDLE_VERSION,
    ) -> PilotSummary:
        ensure_dataset_tree(self.workspace_root)
        batch_id = build_batch_id(date, category, sequence)
        prompt_request = build_prompt_request(
            category=category,
            context_bundle=self.context_bundle,
            context_bundle_version=context_bundle_version,
            prompt_version_number=prompt_version_number,
        )
        generation_result = self.generation_client.generate_batch(prompt_request)

        raw_path = artifact_path(self.workspace_root, "raw_output", category, batch_id)
        write_json(raw_path, generation_result.raw_response)

        examples = generation_result.examples
        if isinstance(examples, str):
            parsed, errors = parse_json_batch(examples)
            if errors:
                retry_path = artifact_path(self.workspace_root, "logs", "retry", batch_id)
                write_json(retry_path, {"batch_id": batch_id, "category": category, "errors": errors, "retry_count": 1})
                self._write_scale_decision(batch_id, category, [], [])
                return PilotSummary(batch_id, category, 0, 0, 0, 0, {"raw": str(raw_path), "retry": str(retry_path)})
            examples = parsed or []

        validation_results = validate_batch(category=category, examples=examples, compiler=self.compiler)
        example_ids = [
            build_example_id(category, date, sequence, index + 1)
            for index in range(len(validation_results))
        ]

        validation_records = [
            build_validation_log_record(
                result,
                batch_id=batch_id,
                prompt_version=prompt_request["prompt_version"],
                context_bundle_version=context_bundle_version,
                example_id=example_ids[index],
                validator_version=build_validator_version(1),
                dataset_version="jac-synth-v0.1.0",
            )
            for index, result in enumerate(validation_results)
        ]
        validation_path = artifact_path(self.workspace_root, "logs", "validation", batch_id, suffix=".jsonl")
        write_jsonl(validation_path, validation_records)

        clean_records = self._records_for_disposition(validation_results, example_ids, Disposition.CLEAN)
        rejected_records = self._records_for_disposition(validation_results, example_ids, Disposition.REJECTED)
        review_records = self._records_for_disposition(validation_results, example_ids, Disposition.REVIEW)
        discarded_records = self._records_for_disposition(validation_results, example_ids, Disposition.DISCARDED)

        artifact_paths = {"raw": str(raw_path), "validation": str(validation_path)}
        self._write_disposition_file(artifact_paths, "clean", "clean_dataset", category, batch_id, clean_records)
        self._write_disposition_file(artifact_paths, "rejected", "rejected", category, batch_id, rejected_records)
        self._write_disposition_file(artifact_paths, "review", "review", category, batch_id, review_records)
        self._write_disposition_file(artifact_paths, "discarded", "rejected", category, f"{batch_id}-discarded", discarded_records)

        review_file_records = default_review_records(batch_id, category, example_ids)
        review_path = artifact_path(self.workspace_root, "review", category, f"{batch_id}-review")
        write_json(review_path, [record.to_dict() for record in review_file_records])
        artifact_paths["manual_review"] = str(review_path)
        artifact_paths["scale_decision"] = self._write_scale_decision(
            batch_id,
            category,
            validation_results,
            review_file_records,
            prompt_request["prompt_version"],
            prompt_version_number,
        )

        generation_log_path = artifact_path(self.workspace_root, "logs", "generation", batch_id)
        write_json(
            generation_log_path,
            {
                "batch_id": batch_id,
                "category": category,
                "model": self.model,
                "prompt_version": prompt_request["prompt_version"],
                "example_count": len(examples),
            },
        )
        artifact_paths["generation"] = str(generation_log_path)

        return PilotSummary(
            batch_id=batch_id,
            category=category,
            clean_count=len(clean_records),
            rejected_count=len(rejected_records),
            review_count=len(review_records),
            discarded_count=len(discarded_records),
            artifact_paths=artifact_paths,
        )

    @staticmethod
    def _records_for_disposition(
        results: list[Any],
        example_ids: list[str],
        disposition: Disposition,
    ) -> list[dict[str, Any]]:
        records = []
        for index, result in enumerate(results):
            if result.disposition == disposition and result.example is not None:
                records.append({"id": example_ids[index], **dict(result.example), "disposition": disposition.value})
        return records

    def _write_disposition_file(
        self,
        artifact_paths: dict[str, str],
        key: str,
        area: str,
        category: str,
        batch_id: str,
        records: list[dict[str, Any]],
    ) -> None:
        if not records:
            return
        path = artifact_path(self.workspace_root, area, category, batch_id, suffix=".jsonl")
        write_jsonl(path, records)
        artifact_paths[key] = str(path)

    def _write_scale_decision(
        self,
        batch_id: str,
        category: str,
        validation_results: list[Any],
        review_records: list[Any],
        prompt_version: str = "",
        prompt_version_number: int = 1,
    ) -> str:
        decision = decide_scale_up(
            batch_id=batch_id,
            category=category,
            validation_results=validation_results,
            review_records=review_records,
        )
        path = artifact_path(self.workspace_root, "logs", "scale_decisions", batch_id)
        write_json(path, decision.to_dict())
        if decision.decision != ScaleDecisionStatus.SCALE_READY and prompt_version:
            revision_path = artifact_path(self.workspace_root, "logs", "prompt_revisions", batch_id)
            write_json(
                revision_path,
                build_prompt_revision_record(
                    prompt_version=f"prompt-{category}-v{prompt_version_number + 1}",
                    previous_prompt_version=prompt_version,
                    category=category,
                    changed_fields=["user_prompt"],
                    reason_for_change="Pilot batch did not meet Task 4 validation or manual review scale gates.",
                    batch_ids_affected=[batch_id],
                    observed_pass_rate_before=decision.compiler_pass_rate,
                    observed_pass_rate_after=None,
                    changed_at=self._batch_timestamp(batch_id),
                    notes="Inspect validation and review artifacts before rerunning this category.",
                ),
            )
        return str(path)

    @staticmethod
    def _batch_timestamp(batch_id: str) -> str:
        date = batch_id.split("-", 1)[0]
        return f"{date[0:4]}-{date[4:6]}-{date[6:8]}T00:00:00Z"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Task 4 single-turn generation pilots.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-pilots")
    run.add_argument("--date", required=True)
    run.add_argument("--start-sequence", type=int, default=1)
    run.add_argument("--model", default=None)
    run.add_argument("--category", choices=SCRIPTED_CATEGORIES, default=None)
    run.add_argument("--prompt-version-number", type=int, default=1)
    run.add_argument("--timeout-seconds", type=float, default=90.0)
    run.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command != "run-pilots":
        return 2

    root = Path(".")
    ensure_dataset_tree(root)
    write_context_bundle(root)
    context_bundle = load_context_bundle(root)
    model = args.model or "gpt-5.5"
    categories = [args.category] if args.category else list(SCRIPTED_CATEGORIES)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "model": model,
                    "categories": categories,
                    "prompt_version_number": args.prompt_version_number,
                }
            )
        )
        return 0

    settings = OpenAISettings.from_env(model=model)
    client = OpenAIGenerationClient(model=settings.model, api_key=settings.api_key, timeout_seconds=args.timeout_seconds)
    runner = SingleTurnPilotRunner(
        workspace_root=root,
        generation_client=client,
        compiler=JacCliCompilerRunner(),
        context_bundle=context_bundle,
        model=settings.model,
    )
    summaries = [
        runner.run_category_pilot(
            category=category,
            date=args.date,
            sequence=args.start_sequence,
            prompt_version_number=args.prompt_version_number,
        )
        for category in categories
    ]
    print(json.dumps([summary.__dict__ for summary in summaries], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
