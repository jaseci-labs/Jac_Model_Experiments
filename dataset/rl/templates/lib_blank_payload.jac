"""RL driver (pure-lib): source_index._blank.

Origin: this_is_jac/source_index.jac. Deterministic: builds the empty
file-payload dict carrying an error message, with all other fields set to fixed
defaults. The marked body is the unit the policy must regenerate; the
`with entry` block exercises it on fixed inputs and prints a deterministic
projection.
"""

def _blank(rel: str, err: str) -> dict {
    __HOLE__
}

with entry {
    cases = [
        ("a/b.jac", "File not found."),
        ("", "Invalid path."),
        ("x.py", "Could not read file.")
    ];
    for (rel, err) in cases {
        d = _blank(rel, err);
        print(d["path"] + "|" + d["error"] + "|" + d["lang"] + "|" + str(d["lines"]) + "|" + str(d["truncated"]));
    }
}
