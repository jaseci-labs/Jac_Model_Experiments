"""RL driver (synthetic): sum of the decimal digits of an integer."""

def digit_sum(n: int) -> int {
    __HOLE__
}

with entry {
    for x in [0, 5, 42, 12345, 99999] {
        print(digit_sum(x));
    }
}
