"""RL driver (synthetic): number of Collatz steps for n to reach 1."""

def collatz_steps(n: int) -> int {
    __HOLE__
}

with entry {
    for x in [1, 2, 6, 7, 27] {
        print(collatz_steps(x));
    }
}
