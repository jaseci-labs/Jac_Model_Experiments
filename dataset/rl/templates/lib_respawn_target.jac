"""RL driver (pure-lib): main.respawn ((re)place a target in the arena via seeded rng).

Origin: this_is_jac/raylib_shooter/web/main.jac (the respawn logic, isolated from any
io/time/graph machinery). Deterministic: the fixed rng seed makes the four LCG draws
identical every run, and all printed floats are rounded to 6 dp. The marked body is the
unit the policy must regenerate; the with entry block exercises it on fixed inputs and
prints a deterministic projection.
"""

glob rng: int = 987654321;

obj Target {
    has x: float,
        y: float,
        z: float,
        base_y: float,
        phase: float,
        alive: bool;
}

"""Uniform pseudo-random float in [0, 1) via a 31-bit LCG."""
def rand01 -> float {
    global rng;
    rng = (rng * 1103515245 + 12345) % 2147483648;
    return float(rng) / 2147483648.0;
}

"""Random float in [lo, hi)."""
def rand_range(lo: float, hi: float) -> float {
    return lo + rand01() * (hi - lo);
}

def respawn(t: Target) {
    __HOLE__
}

with entry {
    t = Target(x=0.0, y=0.0, z=0.0, base_y=0.0, phase=0.0, alive=False);
    respawn(t);
    print(
        str(round(t.x, 6)) + " " + str(round(t.base_y, 6)) + " "
        + str(round(t.z, 6)) + " " + str(round(t.phase, 6)) + " "
        + str(round(t.y, 6)) + " " + str(t.alive)
    );
}
