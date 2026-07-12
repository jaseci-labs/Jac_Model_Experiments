def _search_filter(contents: list, query: str) -> list:
    q = query.lower().strip()
    return [c for c in contents if q in c.lower()]
