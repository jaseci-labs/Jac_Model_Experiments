import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
from run_leg_cf_check import run_leg_cf_check


def test_run_leg_cf_check_counts_passes():
    fake_results = [{"id": f"t{i}", "pass": i < 16} for i in range(16)]  # all pass
    with patch("run_leg_cf_check.run_model", return_value=fake_results):
        passed, total = run_leg_cf_check("03-new/adapters/cpt-v2/0000570_adapters.safetensors")
    assert (passed, total) == (16, 16)


def test_run_leg_cf_check_detects_regression():
    fake_results = [{"id": f"t{i}", "pass": i < 14} for i in range(16)]  # 14/16
    with patch("run_leg_cf_check.run_model", return_value=fake_results):
        passed, total = run_leg_cf_check("some/adapter.safetensors")
    assert (passed, total) == (14, 16)
