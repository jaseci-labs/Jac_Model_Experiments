from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from data_generation.validation import CompilerResult


SubprocessRun = Callable[..., subprocess.CompletedProcess[str]]


class JacCliCompilerRunner:
    def __init__(
        self,
        *,
        command: str = "jac",
        timeout_seconds: int = 30,
        subprocess_run: SubprocessRun = subprocess.run,
    ) -> None:
        self.command = command
        self.timeout_seconds = timeout_seconds
        self.subprocess_run = subprocess_run

    def __call__(self, code: str) -> CompilerResult:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "snippet.jac"
            source_path.write_text(code)
            command: Sequence[str] = [self.command, "check", str(source_path)]
            try:
                completed = self.subprocess_run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return CompilerResult(passed=False, error_message=str(exc))

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        return CompilerResult(
            passed=completed.returncode == 0,
            stdout=stdout,
            stderr=stderr,
            error_message=(stderr or stdout).strip(),
        )
