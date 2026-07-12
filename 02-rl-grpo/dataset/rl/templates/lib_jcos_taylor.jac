"""RL driver (pure-lib): main.na.jcos (range-reduced Taylor cosine).

Origin: this_is_jac/main.jac (the `jcos` helper inside the `na {}` block).
Deterministic: reduces the angle into [-pi, pi] by adding/subtracting 2*pi, then
evaluates a 7-term Taylor series for cosine. The marked body is the unit the
policy must regenerate; the `with entry` block exercises it on fixed inputs and
prints a deterministic projection (rounded to avoid float noise).
"""

def jcos(a: float) -> float {
    __HOLE__
}

with entry {
    cases = [0.0, 1.5707963267948966, 3.141592653589793, -3.141592653589793, 6.283185307179586, 1.0, -2.0];
    for a in cases {
        print(str(round(jcos(a), 6)));
    }
}
