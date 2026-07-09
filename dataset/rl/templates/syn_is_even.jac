"""RL driver (synthetic): test whether an integer is even."""

def is_even(n: int) -> bool {
    __HOLE__
}

with entry {
    for x in [0, 1, 2, 7, -4] {
        print(is_even(x));
    }
}
