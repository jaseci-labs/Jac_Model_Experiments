"""RL driver (pure-lib): bench._is_lib (raylib shared-lib filename test).

Origin: this_is_jac/raylib_shooter/bench.jac (the `_is_lib` predicate — true for a
staged raylib shared-library filename on the given OS). Deterministic: pure string
prefix/suffix checks on fixed inputs. The marked body is the unit the policy must
regenerate; the `with entry` block exercises it on fixed (filename, os) pairs and
prints a deterministic projection.
"""

def _is_lib(fn: str, osname: str) -> bool {
    __HOLE__
}

with entry {
    cases = [
        ["libraylib.dylib", "Darwin"], ["libraylib.so", "Linux"],
        ["foo.dylib", "Darwin"], ["libraylib.so.5", "Linux"],
        ["libraylib.dylib", "Linux"]
    ];
    for c in cases {
        print(str(_is_lib(c[0], c[1])));
    }
}
