# rand01() -> float in [0,1) is provided elsewhere.
def rand_range(lo: float, hi: float) -> float:
    return lo + rand01() * (hi - lo)
