from pathlib import Path
from unittest.mock import patch

import run_epoch_loop as rel


def test_parse_last_losses_takes_latest_occurrence():
    log = (
        "Iter 1: Train loss 1.060, ... Learning Rate 0.000e+00\n"
        "Iter 2: Train loss 2.967, ... Learning Rate 1.534e-08\n"
        "Iter 3: Val loss 1.556, ...\n"
        "Iter 4: Train loss 0.980, ... Learning Rate 1.074e-07\n"
    )
    train, val, lr = rel.parse_last_losses(log)
    assert train == 0.980
    assert val == 1.556
    assert lr == 1.074e-07


def test_parse_last_losses_handles_missing_fields():
    assert rel.parse_last_losses("no metrics here") == (None, None, None)


def test_resume_point_empty_dir_is_step_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(rel, "ADAPTER_DIR", tmp_path)
    assert rel.resume_point() == (0, None, None)


def test_resume_point_picks_highest_numbered_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(rel, "ADAPTER_DIR", tmp_path)
    (tmp_path / "0000544_adapters.safetensors").write_bytes(b"")
    (tmp_path / "0000544_optimizer.safetensors").write_bytes(b"")
    (tmp_path / "0001088_adapters.safetensors").write_bytes(b"")
    (tmp_path / "0001088_optimizer.safetensors").write_bytes(b"")
    steps, adapter, optimizer = rel.resume_point()
    assert steps == 1088
    assert adapter.endswith("0001088_adapters.safetensors")
    assert optimizer.endswith("0001088_optimizer.safetensors")


def test_resume_point_missing_optimizer_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(rel, "ADAPTER_DIR", tmp_path)
    (tmp_path / "0000544_adapters.safetensors").write_bytes(b"")
    steps, adapter, optimizer = rel.resume_point()
    assert steps == 544
    assert adapter is not None
    assert optimizer is None


def test_main_halts_and_reverts_on_regression_past_floor(tmp_path, monkeypatch):
    """A leg-7 CF regression must halt_keep_previous and copy leg 6's
    checkpoint back over the shared adapters.safetensors."""
    monkeypatch.setattr(rel, "ADAPTER_DIR", tmp_path)
    monkeypatch.setattr(rel, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(rel, "STATE_FILE", tmp_path / "results" / "training_state.json")
    monkeypatch.setattr(rel, "REVIEWS_FILE", tmp_path / "results" / "leg_reviews.md")
    monkeypatch.setattr(rel, "windows_per_epoch", lambda: 10)

    # pretend we're resuming right at the start of leg 7 (6 legs already done)
    (tmp_path / "0000060_adapters.safetensors").write_bytes(b"leg6-weights")
    (tmp_path / "0000060_optimizer.safetensors").write_bytes(b"leg6-opt")

    def fake_run_leg(leg, windows, done_steps, resume_adapter, resume_optimizer):
        # leg 7's own weights get written, as run_cpt_leg.py would do
        (tmp_path / f"{done_steps + windows:07d}_adapters.safetensors").write_bytes(b"leg7-weights")
        (tmp_path / f"{done_steps + windows:07d}_optimizer.safetensors").write_bytes(b"leg7-opt")
        (tmp_path / "adapters.safetensors").write_bytes(b"leg7-weights")  # rolling-latest
        return {"duration_s": 1.0, "train_loss": 0.5, "val_loss": 0.6, "final_lr": 1e-7}

    with patch.object(rel, "run_leg", side_effect=fake_run_leg), \
         patch.object(rel, "run_cf_check", return_value=(14, 16)):  # regression, past floor
        rel.main()

    assert (tmp_path / "adapters.safetensors").read_bytes() == b"leg6-weights"
    state = rel.load_state()
    assert state["status"] == "halt_keep_previous"
    assert state["legs"][-1]["leg"] == 7
    assert state["legs"][-1]["cf_passed"] is False
