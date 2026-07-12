"""RL driver (pure-lib): capture.CaptureRunner.summary ok-count (count 'ok' results).

Origin: this_is_jac/capture.jac (the `oks = len([r for r in self.results if r["status"]
== "ok"])` expression from CaptureRunner.summary, isolated as a standalone def). The marked
body is the unit the policy must regenerate. Deterministic: count_ok runs on a FIXED list of
result dicts with mixed statuses, so it always prints '2'. The with entry block exercises it
on fixed inputs and prints a deterministic projection.
"""

def count_ok(results: list[dict]) -> int {
    __HOLE__
}

with entry {
    results = [
        {"step": "benchmark", "status": "ok", "detail": "a"},
        {"step": "shooter_jac.png", "status": "skip", "detail": "b"},
        {"step": "shooter_zig.png", "status": "fail", "detail": "c"},
        {"step": "manifest", "status": "ok", "detail": "d"}
    ];
    print(str(count_ok(results)));
}
