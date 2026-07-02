"""RL driver (synthetic): test whether an integer is even."""

def is_even(n: int) -> bool {
    # >>>HOLE id="syn_is_even" instruction="Implement is_even(n): return True if n is evenly divisible by 2 (n % 2 == 0), otherwise False."
    return n % 2 == 0;
    # <<<HOLE
}

with entry {
    for x in [0, 1, 2, 7, -4] {
        print(is_even(x));
    }
}
