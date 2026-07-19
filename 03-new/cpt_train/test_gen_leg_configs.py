import json

from gen_leg_configs import build_leg_configs, windows_per_epoch


def test_windows_per_epoch_reads_manifest(tmp_path):
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"packed": {"train": 570, "val": 100}}))
    assert windows_per_epoch(m) == 570


def test_all_legs_share_identical_schedule():
    configs, total_iters = build_leg_configs(windows=570, ceiling_epochs=12,
                                              data_dir="d", adapter_dir="a")
    assert len(configs) == 12
    assert total_iters == 570 * 12
    schedules = {json.dumps(c["lr_schedule"], sort_keys=True) for c in configs}
    assert len(schedules) == 1  # every leg's schedule block is byte-identical


def test_each_leg_iters_is_one_epoch():
    configs, _ = build_leg_configs(windows=570, ceiling_epochs=12, data_dir="d", adapter_dir="a")
    assert all(c["iters"] == 570 for c in configs)


def test_schedule_decays_to_floor_at_ceiling():
    configs, total_iters = build_leg_configs(windows=570, ceiling_epochs=12,
                                              data_dir="d", adapter_dir="a")
    args = configs[0]["lr_schedule"]["arguments"]
    assert args == [1.0e-5, total_iters, 1.0e-6]


def test_recipe_matches_cpt_v1_unchanged_fields():
    configs, _ = build_leg_configs(windows=570, ceiling_epochs=12, data_dir="d", adapter_dir="a")
    cfg = configs[0]
    assert cfg["lora_parameters"] == {"rank": 16, "scale": 2.0, "dropout": 0.05}
    assert cfg["num_layers"] == 16
    assert cfg["max_seq_length"] == 4096
    assert cfg["batch_size"] == 1
    assert cfg["learning_rate"] == 1.0e-5
