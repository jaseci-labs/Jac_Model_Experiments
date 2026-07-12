"""RL driver (pure-lib): raylib_shim._keymap (DOM key-code to raylib keycode lookup).

Origin: this_is_jac/raylib_shooter/web/raylib_shim.cl.jac (the _keymap logic, isolated
from io/time/graph/render machinery). Deterministic: a fixed dict lookup with a default of
0, no Math/GL/IO/time/random, so stdout is byte-identical every run. The marked body is the
unit the policy must regenerate; the with entry block exercises it on fixed inputs and
prints a deterministic projection.
"""

def _keymap(code: str) -> int {
    __HOLE__
}

with entry {
    codes = ["KeyW", "ArrowUp", "Space", "Nope"];
    parts = [];
    for code in codes {
        parts.append(str(_keymap(code)));
    }
    print(" ".join(parts));
}
