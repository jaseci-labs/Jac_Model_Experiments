"""RL driver (pure-lib): capture.CaptureRunner.summary (exit-code logic).

Origin: this_is_jac/capture.jac (the ok-count + exit-code computation inside
`summary`). Deterministic: counts results whose 'status' is 'ok' and returns 0
when at least one is ok, else 1. The marked body is the unit the policy must
regenerate; the `with entry` block exercises it on fixed inputs and prints a
deterministic projection.
"""

def _summary_code(results: list) -> int {
    __HOLE__
}

with entry {
    cases = [
        [],
        [{"status": "fail"}, {"status": "skip"}],
        [{"status": "ok"}, {"status": "fail"}],
        [{"status": "ok"}, {"status": "ok"}, {"status": "skip"}]
    ];
    for c in cases {
        print(str(_summary_code(c)));
    }
}
