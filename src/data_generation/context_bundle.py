from __future__ import annotations

from pathlib import Path

from data_generation.artifacts import ensure_dataset_tree
from data_generation.prompt_design import CATEGORY_SCHEMAS


DEFAULT_CONTEXT_BUNDLE_VERSION = "jac-context-v1"
DEFAULT_JAC_GUIDANCE = {
    "jac://guide/pitfalls": "Jac is NOT Python. Semicolons, braces, has declarations, walkers, nodes, and edges matter.",
    "jac://guide/patterns": "Use idiomatic Jac patterns for walkers, nodes, edges, abilities, imports, and tests.",
}


def _read_if_exists(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def build_default_context_bundle(
    *,
    docs_context: str,
    dataset_foundation: str,
    jac_guidance: dict[str, str] | None = None,
    version: str = DEFAULT_CONTEXT_BUNDLE_VERSION,
) -> str:
    guidance = jac_guidance or DEFAULT_JAC_GUIDANCE
    schema_names = ", ".join(CATEGORY_SCHEMAS)
    sections = [
        f"# Jac Context Bundle: {version}",
        "## Core Warning",
        "Jac is NOT Python. Generated code must be idiomatic Jac and must compile.",
        "## Dataset Strategy",
        docs_context,
        "## Dataset Foundation",
        dataset_foundation,
        "## Scripted Category Schemas",
        f"Available scripted schemas: {schema_names}.",
        *[
            f"## {uri}\n{content}"
            for uri, content in guidance.items()
        ],
    ]
    return "\n\n".join(section.strip() for section in sections if section.strip()) + "\n"


def write_context_bundle(
    workspace_root: str | Path = ".",
    *,
    content: str | None = None,
    version: str = DEFAULT_CONTEXT_BUNDLE_VERSION,
) -> Path:
    root = Path(workspace_root)
    ensure_dataset_tree(root)
    bundle = content
    if bundle is None:
        bundle = build_default_context_bundle(
            docs_context=_read_if_exists(root / "docs/context.md"),
            dataset_foundation=_read_if_exists(root / "docs/dataset_foundation.md"),
            version=version,
        )
    path = root / "dataset" / "context" / f"{version}.md"
    path.write_text(bundle)
    return path


def load_context_bundle(workspace_root: str | Path = ".", *, version: str = DEFAULT_CONTEXT_BUNDLE_VERSION) -> str:
    path = Path(workspace_root) / "dataset" / "context" / f"{version}.md"
    if not path.exists():
        write_context_bundle(workspace_root, version=version)
    return path.read_text()
