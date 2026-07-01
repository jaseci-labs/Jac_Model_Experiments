def shade(c: int, pct: int) -> int:
    v = c * pct // 100
    return v if v < 255 else 255
