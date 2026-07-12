"""RL driver (synthetic): clamp an integer into the inclusive range [lo, hi]."""

def clamp(x: int, lo: int, hi: int) -> int {
    __HOLE__
}

with entry {
    print(clamp(5, 0, 10));
    print(clamp(-3, 0, 10));
    print(clamp(15, 0, 10));
    print(clamp(0, 0, 10));
    print(clamp(10, 0, 10));
}
