"""RL driver (numeric): raylib_shim._m_ortho (column-major orthographic matrix).

Origin: this_is_jac/raylib_shooter/web/raylib_shim.cl.jac (the _m_ortho logic,
isolated from io/time/graph/render machinery). Deterministic: pure +,-,*,/ arithmetic
on the six fixed ortho parameters, with no Math/GL/IO/time/random, so stdout is
byte-identical every run. The marked body is the unit the policy must regenerate; the
with entry block exercises it on fixed inputs and prints a deterministic projection
(floats rounded to 6 dp).
"""

def _m_ortho(l: float, r: float, b: float, t: float, n: float, f: float) -> list {
    __HOLE__
}

with entry {
    res = _m_ortho(0.0, 960.0, 600.0, 0.0, -1.0, 1.0);
    parts = [];
    for v in res {
        parts.append(str(round(v, 6)));
    }
    print(" ".join(parts));
}
