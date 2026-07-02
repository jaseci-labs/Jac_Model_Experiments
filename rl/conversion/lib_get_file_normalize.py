def _normalize_rel(rel: str) -> str:
    return rel.lstrip("/").replace("\\", "/")
