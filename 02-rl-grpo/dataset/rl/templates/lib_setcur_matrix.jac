"""RL driver (pure-lib): raylib_shim._setcur (store the current matrix by rlgl mode).

Origin: this_is_jac/raylib_shim.cl.jac (the _setcur helper, isolated from io/time/graph/render
machinery). Deterministic: a pure mutation over a minimal inlined Shim object with fixed fields,
no Math/host/rng. The marked body is the unit the policy must regenerate; the with entry block
exercises both branches on fixed inputs and prints a deterministic projection.
"""

obj Shim {
    has proj: list = [],
        mv: list = [],
        mode: int = 0x1700;
}

def _setcur(sh: any, m: list) {
    __HOLE__
}

with entry {
    a = Shim(proj=[], mv=[], mode=0x1701);
    _setcur(a, [9.0, 8.0]);
    print(str(a.proj) + " " + str(a.mv));
    b = Shim(proj=[], mv=[], mode=0x1700);
    _setcur(b, [7.0, 6.0]);
    print(str(b.proj) + " " + str(b.mv));
}
