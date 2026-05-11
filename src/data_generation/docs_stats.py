from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data_generation.foundation import ALLOWED_CATEGORIES
from data_generation.release import CATEGORY_TARGET_RANGES, build_readiness_summary


def build_stats_snapshot(
    workspace_root: str | Path = ".",
    *,
    version: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    root = Path(workspace_root)
    categories = {}
    for category in ALLOWED_CATEGORIES:
        categories[category] = {
            "clean_count": _jsonl_record_count(root / "dataset/clean_dataset" / category),
            "rejected_count": _jsonl_record_count(root / "dataset/rejected" / category),
            "raw_file_count": _file_count(root / "dataset/raw_output" / category, "*.json"),
            "review_file_count": _file_count(root / "dataset/review" / category, "*.json"),
            "target_range": list(CATEGORY_TARGET_RANGES[category]),
        }

    readiness = build_readiness_summary(root, version=version)
    prompt_versions = readiness.get("prompt_versions", [])
    return {
        "version": version,
        "generated_at": generated_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "clean_total": sum(item["clean_count"] for item in categories.values()),
        "rejected_total": sum(item["rejected_count"] for item in categories.values()),
        "raw_file_total": sum(item["raw_file_count"] for item in categories.values()),
        "review_file_total": sum(item["review_file_count"] for item in categories.values()),
        "log_file_total": _file_count(root / "dataset/logs", "**/*"),
        "categories": categories,
        "readiness": readiness,
        "prompt_versions": prompt_versions,
    }


def render_stats_markdown(snapshot: dict[str, Any]) -> str:
    readiness = snapshot["readiness"]
    lines = [
        "# Dataset Stats",
        "",
        f"Generated from repository artifacts at `{snapshot['generated_at']}`.",
        "",
        f"- Dataset version: `{snapshot['version']}`",
        f"- Release status: `{readiness['status']}`",
        f"- Clean examples: `{snapshot['clean_total']}`",
        f"- Rejected examples: `{snapshot['rejected_total']}`",
        f"- Raw output files: `{snapshot['raw_file_total']}`",
        f"- Review files: `{snapshot['review_file_total']}`",
        f"- Log files: `{snapshot['log_file_total']}`",
        "",
        "For command details, see [release readiness](operations.md#release-readiness) and the [scale runbook](scale_runbook.md).",
        "",
        "## Category Progress",
        "",
        "| Category | Clean | Rejected | Raw files | Review files | Target range |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for category, stats in snapshot["categories"].items():
        target_min, target_max = stats["target_range"]
        lines.append(
            f"| `{category}` | {stats['clean_count']} | {stats['rejected_count']} | "
            f"{stats['raw_file_count']} | {stats['review_file_count']} | {target_min:,}-{target_max:,} |"
        )

    lines.extend(
        [
            "",
            "## Readiness",
            "",
            f"- Current total: `{readiness['counts']['current_total']}`",
            f"- Target total range: `{readiness['counts']['target_total_range'][0]:,}-{readiness['counts']['target_total_range'][1]:,}`",
            f"- Manual review status: `{readiness['manual_review']['status']}`",
            f"- Near duplicates unresolved: `{readiness['near_duplicates']['unresolved_count']}`",
            f"- Near duplicates resolved: `{readiness['near_duplicates']['resolved_count']}`",
            f"- Candidate audit status: `{readiness['validation']['candidate_audit_status']}`",
            f"- Candidate audit failures: `{readiness['validation']['candidate_audit_failure_count']}`",
            f"- Candidate audit warnings: `{readiness['validation']['candidate_audit_warning_count']}`",
            "",
            "## Blockers",
            "",
        ]
    )
    blockers = readiness.get("blockers", [])
    if blockers:
        lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        lines.append("- No blocking readiness limitations reported.")

    lines.extend(
        [
            "",
            "## Prompt And Context Versions",
            "",
            f"- Prompt versions: {_format_inline_list(snapshot.get('prompt_versions', []))}",
            f"- Context bundle versions are recorded per example; see `dataset/context/` and `docs/context.md`.",
            "",
            "## Interpretation Notes",
            "",
            "- Clean counts include records currently present in `dataset/clean_dataset/` before any final release freeze.",
            "- Rejected counts include records kept for inspection or possible recycling; they are not training-ready examples.",
            "- Readiness status comes from the release audit logic and can be blocked by volume, manual review, validation, or unresolved near duplicates.",
            "- This file is generated; update it with `python -m data_generation.docs_stats --version jac-synth-v0.1.0 --output docs/stats.md`.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Markdown dataset stats for project docs.")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generated-at", default=None)
    parser.add_argument("--json", action="store_true", help="Write the raw stats snapshot as JSON instead of Markdown.")
    args = parser.parse_args(argv)

    snapshot = build_stats_snapshot(args.workspace_root, version=args.version, generated_at=args.generated_at)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.json:
        output.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    else:
        output.write_text(render_stats_markdown(snapshot))
    return 0


def _jsonl_record_count(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in sorted(directory.glob("*.jsonl")) for line in path.read_text().splitlines() if line.strip())


def _file_count(directory: Path, pattern: str) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.glob(pattern) if path.is_file())


def _format_inline_list(values: list[Any]) -> str:
    if not values:
        return "`none`"
    return ", ".join(f"`{value}`" for value in values)


if __name__ == "__main__":
    sys.exit(main())
