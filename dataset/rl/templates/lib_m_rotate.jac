"""RL driver (numeric): raylib_shim._m_rotate (axis-angle Rodrigues 4x4 rotation matrix).

Origin: this_is_jac/raylib_shim.cl.jac (the _m_rotate helper, isolated from io/time/graph/render
machinery). Deterministic: Math is aliased to Python's math module outside the hole, so cos/sin/hypot
are pure functions of fixed args; all printed floats are rounded. The marked body is the unit the
policy must regenerate; the with entry block exercises it on fixed inputs and prints a deterministic
projection.
"""

import math;
glob Math = math;

def _m_rotate(ang: float, x: float, y: float, z: float) -> list {
    __HOLE__
}

with entry {
    m = _m_rotate(90.0, 0.0, 0.0, 1.0);
    for v in m {
        print(str(round(v, 6)));
    }
}
