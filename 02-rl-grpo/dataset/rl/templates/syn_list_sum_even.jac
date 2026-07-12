"""RL driver (synthetic): sum the even integers in a list."""

def list_sum_even(nums: list) -> int {
    __HOLE__
}

with entry {
    for x in [[1, 2, 3, 4, 5, 6], [10, 15, 20, 25], [1, 3, 5], []] {
        print(list_sum_even(x));
    }
}
