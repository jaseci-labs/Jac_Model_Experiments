"""RL driver (pure-lib): social_graph trending hashtag normalization.

Origin: this_is_jac/littlex/social_graph.jac (the per-word hashtag normalization
inside `get_trending`'s gather: `word.lower().rstrip(".,!?;:")`, isolated from the
graph counting). Deterministic: pure string lower + right-strip. The marked body is
the unit the policy must regenerate; the `with entry` block exercises it on fixed
hashtag tokens and prints a deterministic projection.
"""

def _norm_tag(word: str) -> str {
    __HOLE__
}

with entry {
    for w in ["#Jac!", "#GRAPH...", "#Hello?;", "#code", "#WOW!!!"] {
        print(_norm_tag(w));
    }
}
