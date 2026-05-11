import shutil
import subprocess

import pytest

from data_generation.jac_compiler import JacCliCompilerRunner


def test_jac_cli_compiler_runner_reports_success(monkeypatch):
    calls = []

    def fake_run(command, capture_output, text, timeout, check):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    runner = JacCliCompilerRunner(command="jac", subprocess_run=fake_run)

    result = runner("with entry { print(\"hi\"); }")

    assert result.passed is True
    assert result.stdout == "ok"
    assert calls[0][0:2] == ["jac", "check"]
    assert calls[0][2].endswith(".jac")


def test_jac_cli_compiler_runner_reports_failure(monkeypatch):
    def fake_run(command, capture_output, text, timeout, check):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="parse error")

    runner = JacCliCompilerRunner(command="jac", subprocess_run=fake_run)

    result = runner("BROKEN")

    assert result.passed is False
    assert result.stderr == "parse error"
    assert result.error_message == "parse error"


@pytest.mark.skipif(shutil.which("jac") is None, reason="jac CLI not installed")
def test_jac_cli_compiler_runner_smoke_valid_snippet():
    result = JacCliCompilerRunner()("with entry { print(\"hi\"); }")

    assert result.passed is True, result.error_message
