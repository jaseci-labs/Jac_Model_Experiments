def _blank(rel: str, err: str) -> dict:
    return {
        "path": rel, "lang": "", "html": "", "lines": 0,
        "raw": "", "truncated": False, "error": err,
    }
