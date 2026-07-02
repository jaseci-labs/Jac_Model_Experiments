def _status_mark(status: str) -> str:
    return "ok  " if status == "ok" else ("skip" if status == "skip" else "FAIL")
