"""RL driver (synthetic): sum the even integers in a list."""

def list_sum_even(nums: list) -> int {
    # >>>HOLE id="syn_list_sum_even" instruction="Implement list_sum_even(nums): return the sum of the even integers in nums (those where n % 2 == 0); return 0 if there are none."
    return sum([n for n in nums if n % 2 == 0]);
    # <<<HOLE
}

with entry {
    for x in [[1, 2, 3, 4, 5, 6], [10, 15, 20, 25], [1, 3, 5], []] {
        print(list_sum_even(x));
    }
}
