from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_generation.artifacts import artifact_path, ensure_dataset_tree, write_json, write_jsonl
from data_generation.context_bundle import DEFAULT_CONTEXT_BUNDLE_VERSION, load_context_bundle, write_context_bundle
from data_generation.foundation import SCRIPTED_CATEGORIES, build_batch_id, build_example_id, build_prompt_version, build_validator_version
from data_generation.jac_compiler import JacCliCompilerRunner
from data_generation.manual_review import ScaleDecisionStatus, build_prompt_revision_record, decide_scale_up, default_review_records
from data_generation.openai_generation import GenerationResult, OpenAIGenerationClient, OpenAISettings
from data_generation.prompt_design import build_prompt_request, latest_scale_prompt_version
from data_generation.release import CATEGORY_TARGET_RANGES, load_clean_candidates
from data_generation.validation import (
    Disposition,
    build_validation_log_record,
    calculate_pass_rates,
    parse_json_batch,
    validate_batch,
)

DEFAULT_SCALE_BATCH_SIZE = 50


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
        requested_count: int | None = None,
        scale_mode: bool = False,
        complexity_target: str | None = None,
        parse_retry_limit: int = 0,
    ) -> PilotSummary:
        ensure_dataset_tree(self.workspace_root)
        batch_id = build_batch_id(date, category, sequence)
        raw_path = artifact_path(self.workspace_root, "raw_output", category, batch_id)
        if raw_path.exists():
            raise FileExistsError(f"generation artifacts already exist for batch: {batch_id}")
        prompt_request = build_prompt_request(
            category=category,
            context_bundle=self.context_bundle,
            requested_count=requested_count,
            context_bundle_version=context_bundle_version,
            prompt_version_number=prompt_version_number,
            scale_mode=scale_mode,
            complexity_target=complexity_target,
        )
        generation_result = self.generation_client.generate_batch(prompt_request)

        write_json(raw_path, generation_result.raw_response)

        examples = generation_result.examples
        if isinstance(examples, str):
            parsed, errors = parse_json_batch(examples)
            if errors and parse_retry_limit > 0:
                retry_path = artifact_path(self.workspace_root, "logs", "retry", batch_id)
                write_json(retry_path, {"batch_id": batch_id, "category": category, "errors": errors, "retry_count": 1})
                generation_result = self.generation_client.generate_batch(prompt_request)
                examples = generation_result.examples
                if isinstance(examples, str):
                    parsed, errors = parse_json_batch(examples)
                    if not errors:
                        examples = parsed or []
            if isinstance(examples, str) and errors:
                retry_path = artifact_path(self.workspace_root, "logs", "retry", batch_id)
                write_json(retry_path, {"batch_id": batch_id, "category": category, "errors": errors, "retry_count": 1})
                self._write_scale_decision(batch_id, category, [], [])
                return PilotSummary(batch_id, category, 0, 0, 0, 0, {"raw": str(raw_path), "retry": str(retry_path)})
            elif isinstance(examples, str):
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

        metadata_context = {
            "batch_id": batch_id,
            "category": category,
            "generation_date": self._batch_timestamp(batch_id),
            "source_prompt_version": prompt_request["prompt_version"],
            "context_bundle_version": context_bundle_version,
            "validator_version": build_validator_version(1),
            "dataset_version": "jac-synth-v0.1.0",
        }
        clean_records = self._records_for_disposition(validation_results, example_ids, Disposition.CLEAN, metadata_context)
        rejected_records = self._records_for_disposition(validation_results, example_ids, Disposition.REJECTED, metadata_context)
        review_records = self._records_for_disposition(validation_results, example_ids, Disposition.REVIEW, metadata_context)
        discarded_records = self._records_for_disposition(validation_results, example_ids, Disposition.DISCARDED, metadata_context)

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
        generation_metadata = generation_result.raw_response.get("generation_metadata", {})
        write_json(
            generation_log_path,
            {
                "batch_id": batch_id,
                "category": category,
                "model": self.model,
                "prompt_version": prompt_request["prompt_version"],
                "requested_count": requested_count or len(examples),
                "example_count": len(examples),
                "retry_count": generation_metadata.get("retry_count", 0),
                "latency_seconds": generation_metadata.get("latency_seconds"),
                "usage": generation_metadata.get("usage"),
                "finish_reason": generation_metadata.get("finish_reason"),
                "refusal": generation_metadata.get("refusal"),
                "prompt_hash": _sha256(prompt_request["user_prompt"]),
                "context_bundle_hash": _sha256(self.context_bundle),
                "pass_rates": calculate_pass_rates(validation_results),
                "rejection_reasons": _rejection_reason_counts(validation_results),
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
        metadata_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        records = []
        for index, result in enumerate(results):
            if result.disposition == disposition and result.example is not None:
                example = dict(result.example)
                records.append(
                    {
                        "id": example_ids[index],
                        **example,
                        "batch_id": metadata_context["batch_id"],
                        "category": metadata_context["category"],
                        "complexity": _record_complexity(metadata_context["category"], example),
                        "compiler_pass": result.compiler_pass,
                        "test_pass": result.test_pass,
                        "manually_reviewed": False,
                        "generator": "openai-api",
                        "generation_date": metadata_context["generation_date"],
                        "source_prompt_version": metadata_context["source_prompt_version"],
                        "context_bundle_version": metadata_context["context_bundle_version"],
                        "validator_version": metadata_context["validator_version"],
                        "dataset_version": metadata_context["dataset_version"],
                        "review_status": "pending" if disposition in {Disposition.CLEAN, Disposition.REVIEW} else "rejected",
                        "rejection_reason": result.rejection_reason,
                        "disposition": disposition.value,
                    }
                )
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
            next_prompt_version_number = _prompt_version_number(prompt_version) + 1
            write_json(
                revision_path,
                build_prompt_revision_record(
                    prompt_version=f"prompt-{category}-v{next_prompt_version_number}",
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


def build_scale_plan(
    workspace_root: str | Path = ".",
    *,
    version: str,
    date: str,
    target_total: int,
    category: str | None = None,
    batch_size: int = DEFAULT_SCALE_BATCH_SIZE,
    max_batches: int | None = None,
) -> dict[str, Any]:
    root = Path(workspace_root)
    candidates = load_clean_candidates(root)
    categories = [category] if category else list(SCRIPTED_CATEGORIES)
    category_targets = _category_scale_targets(target_total)
    planned_batches: list[dict[str, Any]] = []
    category_summaries: dict[str, Any] = {}

    for selected_category in categories:
        current_count = len(candidates.get(selected_category, []))
        target_count = category_targets[selected_category]
        missing_count = max(0, target_count - current_count)
        next_sequence = _next_sequence(root, date=date, category=selected_category)
        prompt_revision_count = _prompt_revision_count(root, selected_category)
        status = "paused" if prompt_revision_count >= 2 else "planned"
        category_summaries[selected_category] = {
            "current_count": current_count,
            "target_count": target_count,
            "missing_count": missing_count,
            "next_sequence": next_sequence,
            "hard_ratio": _hard_ratio(candidates.get(selected_category, [])),
            "complexity_target": _complexity_target(candidates.get(selected_category, [])),
            "prompt_revision_count": prompt_revision_count,
            "status": status,
        }
        if status == "paused":
            continue

        remaining = missing_count
        sequence = next_sequence
        while remaining > 0 and (max_batches is None or len(planned_batches) < max_batches):
            requested_count = min(batch_size, remaining)
            planned_batches.append(
                {
                    "category": selected_category,
                    "date": date,
                    "sequence": sequence,
                    "batch_id": build_batch_id(date, selected_category, sequence),
                    "requested_count": requested_count,
                    "complexity_target": category_summaries[selected_category]["complexity_target"],
                }
            )
            remaining -= requested_count
            sequence += 1

        if max_batches is not None and len(planned_batches) >= max_batches:
            break

    return {
        "dry_run": True,
        "version": version,
        "target_total": target_total,
        "batch_size": batch_size,
        "max_batches": max_batches,
        "categories": category_summaries,
        "batches": planned_batches,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Task 4 single-turn generation pilots.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-pilots")
    run.add_argument("--date", required=True)
    run.add_argument("--start-sequence", type=int, default=1)
    run.add_argument("--model", default=None)
    run.add_argument("--category", choices=SCRIPTED_CATEGORIES, default=None)
    run.add_argument("--prompt-version-number", type=int, default=1)
    run.add_argument("--requested-count", type=int, default=None)
    run.add_argument("--timeout-seconds", type=float, default=90.0)
    run.add_argument("--max-retries", type=int, default=0)
    run.add_argument("--dry-run", action="store_true")

    scale = subparsers.add_parser("scale")
    scale.add_argument("--version", required=True)
    scale.add_argument("--date", required=True)
    scale.add_argument("--target-total", type=int, required=True)
    scale.add_argument("--category", choices=SCRIPTED_CATEGORIES, default=None)
    scale.add_argument("--batch-size", type=int, default=DEFAULT_SCALE_BATCH_SIZE)
    scale.add_argument("--max-batches", type=int, default=None)
    scale.add_argument("--prompt-version-number", type=int, default=None)
    scale.add_argument("--timeout-seconds", type=float, default=90.0)
    scale.add_argument("--max-retries", type=int, default=0)
    scale.add_argument("--dry-run", action="store_true")
    scale.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    root = Path(".")
    ensure_dataset_tree(root)
    write_context_bundle(root)
    context_bundle = load_context_bundle(root)
    model = getattr(args, "model", None) or "gpt-5.5"

    if args.command == "scale":
        plan = build_scale_plan(
            root,
            version=args.version,
            date=args.date,
            target_total=args.target_total,
            category=args.category,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
        )
        if args.dry_run:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        settings = OpenAISettings.from_env(model=model)
        client = OpenAIGenerationClient(
            model=settings.model,
            api_key=settings.api_key,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        )
        runner = SingleTurnPilotRunner(
            workspace_root=root,
            generation_client=client,
            compiler=JacCliCompilerRunner(),
            context_bundle=context_bundle,
            model=settings.model,
        )
        summaries = _execute_scale_plan(
            runner,
            plan,
            prompt_version_number=args.prompt_version_number,
        )
        run_manifest = _write_scale_run_manifest(root, args=args, plan=plan, summaries=summaries)
        print(json.dumps({"plan": plan, "summaries": summaries, "scale_run": run_manifest}, indent=2, sort_keys=True))
        return 0

    if args.command != "run-pilots":
        return 2

    categories = [args.category] if args.category else list(SCRIPTED_CATEGORIES)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "model": model,
                    "categories": categories,
                    "prompt_version_number": args.prompt_version_number,
                    "requested_count": args.requested_count,
                }
            )
        )
        return 0

    settings = OpenAISettings.from_env(model=model)
    client = OpenAIGenerationClient(
        model=settings.model,
        api_key=settings.api_key,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )
    runner = SingleTurnPilotRunner(
        workspace_root=root,
        generation_client=client,
        compiler=JacCliCompilerRunner(),
        context_bundle=context_bundle,
        model=settings.model,
    )
    summaries = []
    for category in categories:
        print(f"running category {category}", file=sys.stderr, flush=True)
        summary = runner.run_category_pilot(
            category=category,
            date=args.date,
            sequence=args.start_sequence,
            prompt_version_number=args.prompt_version_number,
            requested_count=args.requested_count,
        )
        print(f"finished category {category}: clean={summary.clean_count} rejected={summary.rejected_count}", file=sys.stderr, flush=True)
        summaries.append(summary)
    print(json.dumps([summary.__dict__ for summary in summaries], indent=2))
    return 0


def _record_complexity(category: str, example: dict[str, Any]) -> str:
    if category == "code_gen":
        return str(example.get("complexity", "medium"))
    return "medium"


def _prompt_version_number(prompt_version: str) -> int:
    suffix = prompt_version.rsplit("-v", 1)[-1]
    return int(suffix) if suffix.isdigit() else 1


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rejection_reason_counts(results: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        if not result.rejection_reason:
            continue
        counts[result.rejection_reason] = counts.get(result.rejection_reason, 0) + 1
    return counts


def _hard_ratio(records: list[dict[str, Any]]) -> float | None:
    if not records:
        return None
    return sum(1 for record in records if record.get("complexity") == "hard") / len(records)


def _complexity_target(records: list[dict[str, Any]]) -> str | None:
    ratio = _hard_ratio(records)
    if ratio is None or ratio < 0.15:
        return "hard"
    return None


def _category_scale_targets(target_total: int) -> dict[str, int]:
    return {category: CATEGORY_TARGET_RANGES[category][0] for category in SCRIPTED_CATEGORIES}


def _next_sequence(root: Path, *, date: str, category: str) -> int:
    sequence = 1
    for area in ("raw_output", "clean_dataset", "rejected", "review"):
        base = root / "dataset" / area / category
        for path in base.glob(f"{date}-{category}-*.*"):
            maybe_sequence = _sequence_from_batch_name(path.name, category)
            if maybe_sequence is not None:
                sequence = max(sequence, maybe_sequence + 1)
    for log_area in ("validation", "generation", "scale_decisions"):
        base = root / "dataset/logs" / log_area
        for path in base.glob(f"{date}-{category}-*.*"):
            maybe_sequence = _sequence_from_batch_name(path.name, category)
            if maybe_sequence is not None:
                sequence = max(sequence, maybe_sequence + 1)
    return sequence


def _prompt_revision_count(root: Path, category: str) -> int:
    current_prompt = build_prompt_version(category, latest_scale_prompt_version(category))
    count = 0
    for path in (root / "dataset/logs/prompt_revisions").glob(f"*-{category}-*.json"):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            count += 1
            continue
        if payload.get("previous_prompt_version") != current_prompt:
            continue
        reason = str(payload.get("reason_for_change", "")).lower()
        observed = payload.get("observed_pass_rate_before")
        if "compiler" in reason or (isinstance(observed, int | float) and observed < 0.7):
            count += 1
    return count


def _sequence_from_batch_name(name: str, category: str) -> int | None:
    stem = name.split(".", 1)[0]
    marker = f"-{category}-"
    if marker not in stem:
        return None
    suffix = stem.rsplit(marker, 1)[-1]
    suffix = suffix.removesuffix("-review").removesuffix("-discarded")
    return int(suffix) if suffix.isdigit() else None


def _execute_scale_plan(
    runner: SingleTurnPilotRunner,
    plan: dict[str, Any],
    *,
    prompt_version_number: int | None,
) -> list[dict[str, Any]]:
    summaries = []
    for batch in plan["batches"]:
        summary = runner.run_category_pilot(
            category=batch["category"],
            date=batch["date"],
            sequence=batch["sequence"],
            prompt_version_number=prompt_version_number,
            requested_count=batch["requested_count"],
            scale_mode=True,
            complexity_target=batch.get("complexity_target"),
        )
        summaries.append(summary.__dict__)
    return summaries


def _write_scale_run_manifest(
    root: Path,
    *,
    args: argparse.Namespace,
    plan: dict[str, Any],
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    run_id = f"{args.date}-scale-{len(list((root / 'dataset/logs/scale_runs').glob('*.json'))) + 1:03d}"
    manifest = {
        "run_id": run_id,
        "version": args.version,
        "target_total": args.target_total,
        "cli_args": vars(args),
        "planned_batch_count": len(plan["batches"]),
        "completed_batch_count": len(summaries),
        "summaries": summaries,
        "stop_reason": "completed_plan",
    }
    write_json(root / "dataset/logs/scale_runs" / f"{run_id}.json", manifest)
    return manifest


if __name__ == "__main__":
    sys.exit(main())
