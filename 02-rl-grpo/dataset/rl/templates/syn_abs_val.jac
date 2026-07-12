"""RL driver (synthetic): absolute value of an integer without the abs() builtin."""

def abs_val(n: int) -> int {
    __HOLE__
}

with entry {
    for x in [5, -5, 0, -123, 42] {
        print(abs_val(x));
    }
}
