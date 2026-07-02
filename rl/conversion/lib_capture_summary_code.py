def _summary_code(results: list) -> int:
    oks = len([r for r in results if r["status"] == "ok"])
    return 0 if oks > 0 else 1
