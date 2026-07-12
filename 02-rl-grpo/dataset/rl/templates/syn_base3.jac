"""RL driver (synthetic): render a non-negative integer as a base-3 string."""

def base3(n: int) -> str {
    __HOLE__
}

with entry {
    for x in [0, 1, 3, 10, 26] {
        print(base3(x));
    }
}
