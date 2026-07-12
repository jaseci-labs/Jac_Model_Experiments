"""RL driver (synthetic): round a positive float to 2 decimals without round()."""

def round2(x: float) -> float {
    __HOLE__
}

with entry {
    for x in [3.14159, 2.71828, 1.41421, 0.5, 9.999] {
        print(round2(x));
    }
}
