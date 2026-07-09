"""RL driver (synthetic): factorial n! of a non-negative integer."""

def factorial(n: int) -> int {
    __HOLE__
}

with entry {
    for x in [0, 1, 5, 7, 10] {
        print(factorial(x));
    }
}
