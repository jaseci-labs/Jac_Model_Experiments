"""RL driver (pure-lib): capture.CaptureRunner.record (append result + print marked line).

Origin: this_is_jac/capture.jac (the CaptureRunner.record logic, isolated from any
io/time/subprocess machinery). Deterministic: record is called on three FIXED triples
covering all three status branches; the printed lines and the json projection of the
appended dicts are byte-identical every run. The marked body is the unit the policy must
regenerate; the with entry block exercises it on fixed inputs and prints a deterministic
projection.
"""

import json;

obj CaptureRunner {
    has results: list[dict] = [];

    def record(name: str, status: str, detail: str) -> None {
        __HOLE__
    }
}

with entry {
    runner = CaptureRunner();
    runner.record("benchmark", "ok", "Jac 120 / Zig 130");
    runner.record("shooter_jac.png", "skip", "no display");
    runner.record("shooter_zig.png", "fail", "crashed");
    print(json.dumps(runner.results, sort_keys=True));
}
