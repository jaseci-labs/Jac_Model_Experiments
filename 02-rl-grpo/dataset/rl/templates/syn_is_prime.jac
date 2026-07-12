"""RL driver (synthetic): primality test for an integer via trial division."""

def is_prime(n: int) -> bool {
    __HOLE__
}

with entry {
    for x in [1, 2, 15, 17, 97] {
        print(is_prime(x));
    }
}
