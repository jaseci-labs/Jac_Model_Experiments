"""RL driver (pure-lib): analytics.analyze (compute an AnalyticsReport score).

Origin: this_is_jac/analytics.jac (the analyze logic, isolated from any io/time/graph
machinery). Deterministic: the score is float(signatures) * 1.5, an exact multiple of
1.5; only score and service are printed (never the non-deterministic pid), so stdout is
the same fixed lines on every run. The marked body is the unit the policy must
regenerate; the with entry block exercises it on fixed inputs and prints a deterministic
projection.
"""

import os;

obj AnalyticsReport {
    has pid: int,
        score: float,
        service: str = "analytics";
}

def analyze(signatures: int) -> AnalyticsReport {
    __HOLE__
}

with entry {
    for n in [0, 1, 2, 4, 10] {
        r = analyze(n);
        print(str(round(r.score, 6)) + " " + r.service);
    }
}
