"""RL driver (synthetic): keep only the strictly positive integers in a list."""

def filter_positive(nums: list) -> list {
    # >>>HOLE id="syn_filter_positive" instruction="Implement filter_positive(nums): return a new list containing only the elements of nums that are strictly greater than 0, in their original order."
    return [n for n in nums if n > 0];
    # <<<HOLE
}

with entry {
    for x in [[-1, 2, -3, 4], [0, 1, 2], [-5, -6], [3, 0, -1, 9]] {
        print(filter_positive(x));
    }
}
