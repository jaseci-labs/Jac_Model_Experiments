def ok_count(results: list) -> int:
    return len([r for r in results if r["status"] == "ok"])
