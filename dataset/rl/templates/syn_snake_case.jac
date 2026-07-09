"""RL driver (synthetic): lowercase a phrase and turn spaces into underscores."""

def to_snake_case(s: str) -> str {
    __HOLE__
}

with entry {
    for x in ["Hello World", "Jac ML Studio", "The QUICK Brown Fox"] {
        print(to_snake_case(x));
    }
}
