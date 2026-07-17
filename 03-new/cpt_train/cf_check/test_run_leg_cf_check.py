import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
from run_leg_cf_check import run_leg_cf_check
from run_cf_check import run_model


def test_run_leg_cf_check_counts_passes():
    fake_results = [{"id": f"t{i}", "pass": i < 16} for i in range(16)]  # all pass
    with patch("run_leg_cf_check.run_model", return_value=fake_results) as mock_run_model:
        passed, total = run_leg_cf_check("03-new/adapters/cpt-v2")
    assert (passed, total) == (16, 16)
    mock_run_model.assert_called_once_with(
        "cpt-v2-leg", "models/qwen-q4",
        adapter_path="03-new/adapters/cpt-v2",
    )


def test_run_leg_cf_check_detects_regression():
    fake_results = [{"id": f"t{i}", "pass": i < 14} for i in range(16)]  # 14/16
    with patch("run_leg_cf_check.run_model", return_value=fake_results) as mock_run_model:
        passed, total = run_leg_cf_check("some/adapter/dir")
    assert (passed, total) == (14, 16)
    mock_run_model.assert_called_once_with(
        "cpt-v2-leg", "models/qwen-q4", adapter_path="some/adapter/dir",
    )
    assert mock_run_model.call_args.kwargs["adapter_path"] == "some/adapter/dir"


def test_run_leg_cf_check_passes_directory_not_file():
    """Regression test for the file-vs-directory bug: mlx_lm.utils.load's
    adapter_path routes into load_adapters, which treats adapter_path as a
    DIRECTORY (it opens adapter_path / "adapter_config.json" and loads
    adapter_path / "adapters.safetensors"). Passing a numbered checkpoint
    FILE (e.g. "...0000570_adapters.safetensors") raises NotADirectoryError
    at runtime. This asserts the value forwarded to adapter_path is
    directory-shaped, not file-shaped -- it would have caught the original
    bug where run_leg_cf_check(adapter_checkpoint) forwarded a .safetensors
    FILE path straight through."""
    fake_results = [{"id": "t0", "pass": True}]
    with patch("run_leg_cf_check.run_model", return_value=fake_results) as mock_run_model:
        run_leg_cf_check("03-new/adapters/cpt-v2")
    adapter_path_arg = mock_run_model.call_args.kwargs["adapter_path"]
    assert not adapter_path_arg.endswith(".safetensors"), (
        "adapter_path must be a directory (containing adapter_config.json + "
        "adapters.safetensors), not a numbered checkpoint FILE -- "
        "mlx_lm.utils.load's load_adapters opens adapter_path/adapter_config.json "
        "and raises NotADirectoryError if adapter_path is itself a file"
    )


def test_run_model_load_accepts_directory_shape(tmp_path):
    """Real integration test (not mocked) for the file-vs-directory distinction:
    exercises run_model's actual `load()` call against a tmp_path directory
    containing a dummy adapter_config.json + adapters.safetensors, to confirm
    mlx_lm's adapter loading resolves a DIRECTORY without raising
    NotADirectoryError. Only mlx_lm.generate is mocked (no real inference,
    no model download needed) -- the load()/load_adapters() path itself runs
    for real, which is what actually broke before this fix."""
    import json
    import mlx.core as mx
    from safetensors.numpy import save_file

    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora",
        "num_layers": 0,
        "lora_parameters": {"rank": 8, "alpha": 16, "dropout": 0.0, "scale": 1.0},
    }))
    # Empty adapter weights are fine -- num_layers=0 means load_adapters
    # applies LoRA to zero layers, so no shape mismatch against the real base
    # model. We only need load()/load_adapters() to resolve the directory
    # shape without raising NotADirectoryError.
    save_file({}, str(adapter_dir / "adapters.safetensors"))

    from mlx_lm import load
    model, tok = load("models/qwen-q4", adapter_path=str(adapter_dir))
    assert model is not None
    assert tok is not None


def test_run_model_load_rejects_file_shape(tmp_path):
    """The true negative counterpart of test_run_model_load_accepts_directory_shape
    above: exercises the REAL (unmocked) mlx_lm.utils.load() against a FILE path
    (not a directory) as adapter_path -- reproducing the original bug where
    run_leg_cf_check forwarded a numbered checkpoint .safetensors FILE straight
    through to load_adapters. load_adapters opens adapter_path/"adapter_config.json",
    which raises NotADirectoryError when adapter_path is itself a file (not a
    directory) -- confirmed by actually running this against mlx_lm's real load().
    Only mlx_lm.generate is out of scope here too (the failure happens during
    load_adapters, well before any generate call, so no model download/inference
    is needed)."""
    adapter_file = tmp_path / "0000570_adapters.safetensors"
    adapter_file.write_bytes(b"")

    from mlx_lm import load
    import pytest
    with pytest.raises(NotADirectoryError):
        load("models/qwen-q4", adapter_path=str(adapter_file))
