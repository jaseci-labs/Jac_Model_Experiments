"""RL driver (synthetic): lowercase a phrase and turn spaces into underscores."""

def to_snake_case(s: str) -> str {
    # >>>HOLE id="syn_snake_case" instruction="Implement to_snake_case(s): return s lowercased with every space character replaced by a single underscore."
    return s.lower().replace(" ", "_");
    # <<<HOLE
}

with entry {
    for x in ["Hello World", "Jac ML Studio", "The QUICK Brown Fox"] {
        print(to_snake_case(x));
    }
}
