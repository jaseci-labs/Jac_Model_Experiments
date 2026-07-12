# jcos(a) -> float (range-reduced Taylor cosine) provided elsewhere.
def jsin(a: float) -> float:
    return jcos(a - 1.5707963267948796)
