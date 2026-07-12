"""RL driver (pure-lib): capture.CaptureRunner.record (mark logic).

Origin: this_is_jac/capture.jac (the status->label mapping inside `record`).
Deterministic: maps a status string to a fixed four-character marker used in the
log line. The marked body is the unit the policy must regenerate; the
`with entry` block exercises it on fixed inputs and prints a deterministic
projection.
"""

def _status_mark(status: str) -> str {
    # >>>HOLE id="lib_capture_mark" instruction="Implement _status_mark(status): return a four-character marker string for a step status. If status equals 'ok' return 'ok  ' (ok followed by two spaces); else if status equals 'skip' return 'skip'; otherwise return 'FAIL'."
    return "ok  " if status == "ok" else ("skip" if status == "skip" else "FAIL");
    # <<<HOLE
}

with entry {
    for s in ["ok", "skip", "fail", "error", "OK", ""] {
        print("[" + _status_mark(s) + "] " + s);
    }
}
