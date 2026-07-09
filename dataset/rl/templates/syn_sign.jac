"""RL driver (synthetic): sign of an integer (-1, 0, or 1)."""

def sign(n: int) -> int {
    __HOLE__
}

with entry {
    for x in [7, -4, 0, 100, -1] {
        print(sign(x));
    }
}
