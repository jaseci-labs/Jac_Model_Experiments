"""RL driver (pure-lib): source_index._blank.

Origin: this_is_jac/source_index.jac. Deterministic: builds the empty
file-payload dict carrying an error message, with all other fields set to fixed
defaults. The marked body is the unit the policy must regenerate; the
`with entry` block exercises it on fixed inputs and prints a deterministic
projection.
"""

def _blank(rel: str, err: str) -> dict {
    # >>>HOLE id="lib_blank_payload" instruction="Implement _blank(rel, err): return a dict describing an empty/blank file payload. It must contain key 'path' set to rel, key 'error' set to err, and the fixed defaults 'lang' as empty string, 'html' as empty string, 'lines' as 0, 'raw' as empty string, and 'truncated' as False."
    return {
        "path": rel, "lang": "", "html": "", "lines": 0,
        "raw": "", "truncated": False, "error": err
    };
    # <<<HOLE
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
