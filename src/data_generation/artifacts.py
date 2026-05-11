from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from data_generation.foundation import DATASET_STORAGE_PATHS


def ensure_dataset_tree(workspace_root: str | Path = ".") -> None:
    root = Path(workspace_root)
    for area, value in DATASET_STORAGE_PATHS.items():
        if isinstance(value, dict):
            for path in value.values():
                (root / path).mkdir(parents=True, exist_ok=True)
        else:
            (root / value).mkdir(parents=True, exist_ok=True)


def artifact_path(
    workspace_root: str | Path,
    area: str,
    category_or_log_type: str,
    batch_id: str,
    *,
    suffix: str = ".json",
) -> Path:
    root = Path(workspace_root)
    value = DATASET_STORAGE_PATHS[area]
    if not isinstance(value, dict):
        base_path = root / value
    else:
        base_path = root / value[category_or_log_type]
    return base_path / f"{batch_id}{suffix}"


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{destination.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp_path, destination)


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{destination.name}.tmp")
    temp_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    os.replace(temp_path, destination)
