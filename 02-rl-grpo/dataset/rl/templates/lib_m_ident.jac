"""RL driver (pure-lib): raylib_shim._m_ident (column-major 4x4 identity matrix).

Origin: this_is_jac/raylib_shooter/web/raylib_shim.cl.jac (the _m_ident logic,
isolated from io/time/graph/render machinery). Deterministic: the body is a fixed
literal 16-float list with no inputs, Math/GL, or IO, so stdout is byte-identical
every run. The marked body is the unit the policy must regenerate; the with entry
block exercises it on fixed inputs and prints a deterministic projection.
"""

def _m_ident() -> list {
    __HOLE__
}

with entry {
    print(str(_m_ident()));
}
