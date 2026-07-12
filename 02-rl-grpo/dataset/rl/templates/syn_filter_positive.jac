"""RL driver (synthetic): keep only the strictly positive integers in a list."""

def filter_positive(nums: list) -> list {
    __HOLE__
}

with entry {
    for x in [[-1, 2, -3, 4], [0, 1, 2], [-5, -6], [3, 0, -1, 9]] {
        print(filter_positive(x));
    }
}
