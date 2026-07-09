"""RL driver (pure-lib): social_graph feed search filter.

Origin: this_is_jac/littlex/social_graph.jac (the `load_feed` deliver search
filter: lowercase+strip the query, keep items whose lowercased content contains it,
isolated from the graph walk). Deterministic: pure case-insensitive substring
filter over a fixed list. The marked body is the unit the policy must regenerate;
the `with entry` block exercises it on fixed inputs and prints a deterministic
projection.
"""

def _search_filter(contents: list, query: str) -> list {
    __HOLE__
}

with entry {
    contents = ["Hello Jac", "Graph DB rocks", "jac is fun", "Python here"];
    for q in ["  JAC ", "graph", "xyz", ""] {
        print(str(_search_filter(contents, q)));
    }
}
