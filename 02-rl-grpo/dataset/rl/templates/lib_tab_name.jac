"""RL driver (pure-lib): app.get_tab_name (map a tab key to its display label).

Origin: this_is_jac/littlex/frontend.cl.jac (the get_tab_name lambda, isolated from
the JSX render machinery and lifted into a standalone def). Deterministic: pure
string-to-string mapping with no IO/time/random. The marked body is the unit the
policy must regenerate; the with entry block exercises it on fixed inputs and prints
a deterministic projection.
"""

def get_tab_name(t: str) -> str {
    __HOLE__
}

with entry {
    keys = ["feed", "explore", "channels", "profile", "unknown"];
    out = [];
    for k in keys {
        out.append(k + "=" + get_tab_name(k));
    }
    print(" ".join(out));
}
