"""RL driver (pure-lib): app.displayUsername (strip the @domain off a profile handle).

Origin: this_is_jac/littlex/frontend.cl.jac (the displayUsername derivation inside the
app render, isolated from JSX and lifted into a standalone def). Deterministic: pure
string slicing on fixed inputs, no IO/time/random. The marked body is the unit the
policy must regenerate; the with entry block exercises it on fixed inputs and prints
a deterministic projection.
"""

def display_username(profileUsername: str) -> str {
    __HOLE__
}

with entry {
    samples = ["alice@example.com", "bob@littlex", "plainname", "", "x@y@z"];
    out = [];
    for s in samples {
        out.append("[" + s + "]->" + display_username(s));
    }
    print(" ".join(out));
}
