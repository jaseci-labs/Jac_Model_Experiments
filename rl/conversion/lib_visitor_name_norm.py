def name_norm(name: str) -> str:
    return name.strip()[:40] or "anonymous"
