"""RL driver (synthetic): nth Fibonacci number (0-indexed, fib(0)=0, fib(1)=1)."""

def fibonacci(n: int) -> int {
    __HOLE__
}

with entry {
    for x in [0, 1, 2, 10, 20] {
        print(fibonacci(x));
    }
}
