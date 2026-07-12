"""RL driver (pure-lib | numeric | graph): guestbook.guestbook_stats (tally total signatures across all days).

Origin: this_is_jac/guestbook.jac (the guestbook_stats counting logic, isolated from io/time/graph/render
machinery). Deterministic: a fixed graph is built off a fresh, unpersisted hub node each run (3 Day nodes with
2/3/1 Visitors) so nothing accumulates on the persistent root; the body is a pure sum over edge counts that is
independent of traversal/iteration order. The marked body is the unit the policy must regenerate; the with entry
block exercises it on fixed inputs and prints a deterministic projection.
"""

edge OnDay {}
edge Signed {}

node Hub {}

node Day {
    has date: str = "";
}

node Visitor {
    has name: str = "";
}

def count_signatures(hub: Hub) -> int {
    __HOLE__
}

with entry {
    hub = Hub();

    d1 = hub +>: OnDay() :+> Day(date="2026-06-01");
    d1[0] +>: Signed() :+> Visitor(name="a");
    d1[0] +>: Signed() :+> Visitor(name="b");

    d2 = hub +>: OnDay() :+> Day(date="2026-06-02");
    d2[0] +>: Signed() :+> Visitor(name="c");
    d2[0] +>: Signed() :+> Visitor(name="d");
    d2[0] +>: Signed() :+> Visitor(name="e");

    d3 = hub +>: OnDay() :+> Day(date="2026-06-03");
    d3[0] +>: Signed() :+> Visitor(name="f");

    print(count_signatures(hub));
}
