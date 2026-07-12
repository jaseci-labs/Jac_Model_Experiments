# _one_dp(x) -> str and BenchResult (has avg: float, max: float, frames: int) provided elsewhere.
def _fmt_row(label: str, d) -> str:
    if d is None:
        return "  " + label.ljust(6) + "n/a".rjust(14) + "n/a".rjust(14) + "n/a".rjust(12)
    return "  " + label.ljust(6) + _one_dp(d.avg).rjust(14) + _one_dp(d.max).rjust(14) + str(d.frames).rjust(12)
