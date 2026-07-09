"""RL driver (synthetic): reverse the decimal digits of an integer, keeping sign."""

def reverse_int(n: int) -> int {
    __HOLE__
}

with entry {
    for x in [123, 100, -456, 7, 0] {
        print(reverse_int(x));
    }
}
