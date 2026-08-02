"""Unit tests for adapter_config_fix.py -- spectrum-plan.md §7's silent-drop trap."""

import json

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from adapter_config_fix import (
    adapter_keys,
    assert_adapter_keys_present,
    rewrite_adapter_config,
    spectrum_num_layers,
)
from spectrum_lora_layers import make_linear_to_lora_layers
from test_spectrum_lora_layers import LORA_CFG, tiny_model


def test_num_layers_covers_a_superset_of_the_picks():
    # min pick 7 -> 48 - 7 = 41 -> stock trailing slice covers blocks 7..47
    assert spectrum_num_layers([7, 20, 47]) == 41
    assert spectrum_num_layers(list(range(32, 48))) == 16   # matches the incumbent
    assert spectrum_num_layers([0, 47]) == 48               # worst case: all blocks


def test_num_layers_rejects_empty_picks():
    with pytest.raises(ValueError):
        spectrum_num_layers([])


def test_rewrite_adapter_config_sets_num_layers_and_provenance(tmp_path):
    cfg = tmp_path / "adapter_config.json"
    cfg.write_text(json.dumps({
        "fine_tune_type": "lora", "num_layers": 16,
        "lora_parameters": {"rank": 16, "scale": 2.0, "dropout": 0.05},
        "model": "models/qwen-q4",
    }))
    out = rewrite_adapter_config(tmp_path, [7, 11, 47])

    assert out["num_layers"] == 41
    assert out["spectrum_layers"] == [7, 11, 47]
    assert out["spectrum_num_layers_original"] == 16
    # everything load_adapters actually reads must survive untouched
    assert out["fine_tune_type"] == "lora"
    assert out["lora_parameters"] == {"rank": 16, "scale": 2.0, "dropout": 0.05}
    assert json.loads(cfg.read_text())["num_layers"] == 41


def test_rewrite_is_idempotent(tmp_path):
    cfg = tmp_path / "adapter_config.json"
    cfg.write_text(json.dumps({
        "fine_tune_type": "lora", "num_layers": 16,
        "lora_parameters": {"rank": 16, "scale": 2.0, "dropout": 0.05},
    }))
    rewrite_adapter_config(tmp_path, [7, 47])
    again = rewrite_adapter_config(tmp_path, [7, 47])
    assert again["num_layers"] == 41
    assert again["spectrum_num_layers_original"] == 16   # not overwritten with 41


def _save_adapter(model, path):
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(path), weights)
    return sorted(weights)


def test_assert_adapter_keys_present_passes_when_every_key_lands(tmp_path):
    trained = tiny_model(num_layers=4)
    trained.freeze()
    make_linear_to_lora_layers([0, 2])(trained, 2, LORA_CFG)
    f = tmp_path / "adapters.safetensors"
    saved = _save_adapter(trained, f)
    assert len(saved) == 32
    assert adapter_keys(f) == saved

    fresh = tiny_model(num_layers=4)
    fresh.freeze()
    make_linear_to_lora_layers([0, 2])(fresh, 2, LORA_CFG)
    assert assert_adapter_keys_present(fresh, f) == 32


def test_assert_adapter_keys_present_catches_the_silent_drop(tmp_path):
    """The exact failure spectrum-plan.md §7 describes: a spectrum adapter
    naming layer 0 loaded into a model whose LoRA was rebuilt on layers 2-3.
    `load_weights(strict=False)` would swallow it; this must not."""
    trained = tiny_model(num_layers=4)
    trained.freeze()
    make_linear_to_lora_layers([0, 2])(trained, 2, LORA_CFG)
    f = tmp_path / "adapters.safetensors"
    _save_adapter(trained, f)

    wrong = tiny_model(num_layers=4)
    wrong.freeze()
    make_linear_to_lora_layers([2, 3])(wrong, 2, LORA_CFG)   # the stock slice
    with pytest.raises(RuntimeError, match=r"layers\.0"):
        assert_adapter_keys_present(wrong, f)


def test_superset_conversion_makes_every_key_land(tmp_path):
    """Proves §7's primary fix works: converting blocks min(picks)..n-1 covers
    the picks, so nothing is dropped."""
    trained = tiny_model(num_layers=4)
    trained.freeze()
    make_linear_to_lora_layers([1, 3])(trained, 2, LORA_CFG)
    f = tmp_path / "adapters.safetensors"
    _save_adapter(trained, f)

    n_layers = spectrum_num_layers([1, 3], num_blocks=4)   # 4 - 1 = 3 -> blocks 1,2,3
    superset = list(range(4 - n_layers, 4))
    assert superset == [1, 2, 3]
    covered = tiny_model(num_layers=4)
    covered.freeze()
    make_linear_to_lora_layers(superset)(covered, len(superset), LORA_CFG)
    assert assert_adapter_keys_present(covered, f) == 32


def test_untrained_covered_layer_has_a_zero_lora_b(tmp_path):
    """§7's zero-delta claim: a covered-but-untrained layer contributes nothing,
    because lora_b initialises to zeros in both LoRALinear and LoRASwitchLinear."""
    m = tiny_model(num_layers=4)
    m.freeze()
    make_linear_to_lora_layers([1, 2, 3])(m, 3, LORA_CFG)
    zeros = [k for k, v in tree_flatten(m.trainable_parameters())
             if k.endswith("lora_b")]
    assert zeros
    for k, v in tree_flatten(m.trainable_parameters()):
        if k.endswith("lora_b"):
            assert float(mx.abs(v).max().item()) == 0.0, k
