"""Unit tests for spectrum_lora_layers.py.

The model fixture is a REAL mlx_lm Qwen3-MoE built from
`mlx_lm.models.qwen3_moe.ModelArgs`, just tiny (4 blocks, 4 experts, hidden 32).
That means the conversion under test walks the same classes the 30B model does --
`nn.Linear` for the attention projections and the router, `SwitchLinear` for the
three stacked expert projections -- so the LoRASwitchLinear branch is genuinely
exercised without loading 16GB.
"""

import json

import mlx.core as mx
import pytest

import mlx_lm.lora
import mlx_lm.tuner.utils as tuner_utils
from mlx_lm.models.qwen3_moe import Model as Qwen3MoeModel
from mlx_lm.models.qwen3_moe import ModelArgs

import spectrum_lora_layers as S

LORA_CFG = {"rank": 16, "scale": 2.0, "dropout": 0.05}


def tiny_model(num_layers=4, num_experts=4):
    args = ModelArgs(
        model_type="qwen3_moe",
        hidden_size=32,
        num_hidden_layers=num_layers,
        intermediate_size=64,
        num_attention_heads=4,
        num_experts=num_experts,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        moe_intermediate_size=16,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        head_dim=8,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        max_position_embeddings=128,
        norm_topk_prob=True,
    )
    m = Qwen3MoeModel(args)
    mx.eval(m.parameters())
    return m


@pytest.fixture(autouse=True)
def restore_upstream():
    """Every test gets the stock function back, whatever it did to the module."""
    stock = tuner_utils.linear_to_lora_layers
    yield
    tuner_utils.linear_to_lora_layers = stock
    mlx_lm.lora.linear_to_lora_layers = stock


# --- guard -----------------------------------------------------------------
def test_guard_passes_against_the_installed_mlx_lm():
    S._assert_upstream_still_matches()  # must not raise


def test_guard_pins_the_version_and_hash_it_was_written_against():
    assert S.UPSTREAM_MLX_LM_VERSION == "0.31.3"
    assert S.UPSTREAM_SRC_SHA256 == (
        "96aa5d61a790a24b228127f5b5eaf2205a833e636bfa1827ab8f65fbc6ddd111"
    )


def test_guard_fails_when_the_target_is_gone(monkeypatch):
    monkeypatch.delattr(tuner_utils, "linear_to_lora_layers")
    with pytest.raises(RuntimeError, match="linear_to_lora_layers"):
        S._assert_upstream_still_matches()


def test_guard_fails_when_the_signature_changed(monkeypatch):
    def impostor(model, n, config):  # missing use_dora
        pass
    monkeypatch.setattr(tuner_utils, "linear_to_lora_layers", impostor)
    monkeypatch.setattr(mlx_lm.lora, "linear_to_lora_layers", impostor)
    with pytest.raises(RuntimeError, match="signature"):
        S._assert_upstream_still_matches()


def test_guard_fails_when_the_trailing_slice_is_gone(monkeypatch):
    def impostor(model, num_layers, config, use_dora=False):
        for l in model.layers[:num_layers]:  # leading, not trailing
            pass
    monkeypatch.setattr(tuner_utils, "linear_to_lora_layers", impostor)
    monkeypatch.setattr(mlx_lm.lora, "linear_to_lora_layers", impostor)
    with pytest.raises(RuntimeError, match="trailing slice"):
        S._assert_upstream_still_matches()


def test_guard_fails_when_lora_py_no_longer_shares_the_object(monkeypatch):
    def other(model, num_layers, config, use_dora=False):
        pass
    monkeypatch.setattr(mlx_lm.lora, "linear_to_lora_layers", other)
    with pytest.raises(RuntimeError, match="import structure"):
        S._assert_upstream_still_matches()


# --- layer id loading ------------------------------------------------------
def test_load_layer_ids_sorts_dedupes_and_validates(tmp_path):
    p = tmp_path / "spectrum_layers.json"
    p.write_text(json.dumps({"layers": [7, 3, 3, 47], "count": 4}))
    assert S.load_layer_ids(p) == [3, 7, 47]


def test_load_layer_ids_rejects_out_of_range(tmp_path):
    p = tmp_path / "spectrum_layers.json"
    p.write_text(json.dumps({"layers": [0, 48]}))
    with pytest.raises(ValueError, match="out of range"):
        S.load_layer_ids(p, num_layers=48)


def test_load_layer_ids_fails_loudly_when_the_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        S.load_layer_ids(tmp_path / "nope.json")


# --- the replacement conversion -------------------------------------------
def test_replacement_lora_izes_exactly_the_requested_blocks():
    m = tiny_model(num_layers=4)
    m.freeze()
    S.make_linear_to_lora_layers([0, 2])(m, 2, LORA_CFG)
    rep = S.collect_lora_report(m)
    assert rep["layer_ids"] == [0, 2]


def test_replacement_hits_the_switchlinear_branch():
    m = tiny_model(num_layers=4)
    m.freeze()
    S.make_linear_to_lora_layers([1, 3])(m, 2, LORA_CFG)
    rep = S.collect_lora_report(m)
    assert rep["n_switch"] == 6  # 3 per selected layer
    assert rep["n_lora_tensors"] == 2 * 8 * 2  # layers x modules x {lora_a, lora_b}
    for L in (1, 3):
        assert rep["suffixes_by_layer"][L] == list(S.EXPECTED_SUFFIXES)


def test_replacement_matches_stock_when_asked_for_the_trailing_slice():
    picked = tiny_model(num_layers=4)
    picked.freeze()
    S.make_linear_to_lora_layers([2, 3])(picked, 2, LORA_CFG)
    a = S.collect_lora_report(picked)

    stock = tiny_model(num_layers=4)
    stock.freeze()
    tuner_utils.linear_to_lora_layers(stock, 2, LORA_CFG)
    b = S.collect_lora_report(stock)

    assert a["layer_ids"] == b["layer_ids"] == [2, 3]
    assert a["n_lora_tensors"] == b["n_lora_tensors"]
    assert a["n_switch"] == b["n_switch"]
    assert a["trainable_params"] == b["trainable_params"]


def test_capacity_is_placement_independent():
    """The whole comparison rests on this: any 2 of 4 blocks cost the same."""
    counts = set()
    for ids in ([0, 1], [0, 3], [1, 2], [2, 3]):
        m = tiny_model(num_layers=4)
        m.freeze()
        S.make_linear_to_lora_layers(ids)(m, 2, LORA_CFG)
        counts.add(S.collect_lora_report(m)["trainable_params"])
    assert len(counts) == 1, counts


def test_num_layers_mismatch_fails_loudly():
    m = tiny_model(num_layers=4)
    m.freeze()
    with pytest.raises(ValueError, match="num_layers"):
        S.make_linear_to_lora_layers([0, 2])(m, 16, LORA_CFG)


def test_replacement_rejects_an_index_past_the_end():
    m = tiny_model(num_layers=4)
    m.freeze()
    with pytest.raises(ValueError, match="out of range"):
        S.make_linear_to_lora_layers([0, 9])(m, 2, LORA_CFG)


# --- the rebind ------------------------------------------------------------
def test_apply_patch_rebinds_BOTH_module_attributes():
    stock = tuner_utils.linear_to_lora_layers
    S.apply_patch([0, 2])
    assert tuner_utils.linear_to_lora_layers is not stock
    assert mlx_lm.lora.linear_to_lora_layers is not stock
    assert mlx_lm.lora.linear_to_lora_layers is tuner_utils.linear_to_lora_layers


def test_rebind_actually_intercepts_the_training_path():
    """mlx_lm.lora's train_model resolves the name from ITS OWN globals.

    Rebinding only mlx_lm.tuner.utils would leave this call on the stock trailing
    slice while looking patched -- exactly the failure spectrum-plan.md §6.2
    calls out. Call through mlx_lm.lora's namespace to prove the intercept.
    """
    S.apply_patch([0, 2])
    m = tiny_model(num_layers=4)
    m.freeze()
    mlx_lm.lora.linear_to_lora_layers(m, 2, LORA_CFG)
    assert S.collect_lora_report(m)["layer_ids"] == [0, 2]  # NOT [2, 3]


def test_rebind_also_intercepts_load_adapters_path():
    """mlx_lm.tuner.utils.load_adapters calls the module-global name."""
    S.apply_patch([1, 3])
    m = tiny_model(num_layers=4)
    m.freeze()
    tuner_utils.linear_to_lora_layers(m, 2, LORA_CFG)
    assert S.collect_lora_report(m)["layer_ids"] == [1, 3]


def test_stock_control_reproduces_the_trailing_slice():
    """With no rebind, the same harness must measure {n-2, n-1} -- proof the
    assertions above are measuring placement and not a constant."""
    m = tiny_model(num_layers=4)
    m.freeze()
    tuner_utils.linear_to_lora_layers(m, 2, LORA_CFG)
    assert S.collect_lora_report(m)["layer_ids"] == [2, 3]


def test_apply_patch_refuses_when_mlx_lm_lora_the_dpo_package_is_loaded(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "mlx_lm_lora", object())
    with pytest.raises(RuntimeError, match="mlx_lm_lora"):
        S.apply_patch([0, 2])


# --- verify_layers ---------------------------------------------------------
def test_verify_layers_passes_on_a_correct_conversion():
    m = tiny_model(num_layers=4)
    m.freeze()
    S.make_linear_to_lora_layers([0, 2])(m, 2, LORA_CFG)
    ok, problems = S.verify_layers(m, [0, 2], expect_tensors=32)
    assert ok, problems


def test_verify_layers_fails_on_the_wrong_layer_set():
    m = tiny_model(num_layers=4)
    m.freeze()
    S.make_linear_to_lora_layers([0, 2])(m, 2, LORA_CFG)
    ok, problems = S.verify_layers(m, [1, 3], expect_tensors=32)
    assert not ok
    assert any("layer set" in p for p in problems)


def test_verify_layers_fails_on_a_wrong_trainable_param_count():
    m = tiny_model(num_layers=4)
    m.freeze()
    S.make_linear_to_lora_layers([0, 2])(m, 2, LORA_CFG)
    ok, problems = S.verify_layers(m, [0, 2], expect_tensors=32,
                                   expect_trainable=281_838_000)
    assert not ok
    assert any("trainable" in p for p in problems)


# --- the pinned capacity constant ------------------------------------------
def test_expected_trainable_params_matches_the_arithmetic_it_claims():
    """281_837_568, not the 281_838_080 that a naive read of mlx_lm's rounded
    "281.838M" banner produces. --verify-layers gates training on this exact
    integer, so an off-by-512 constant means the gate can never PASS.

    Derivation for one Qwen3-Coder-30B-A3B block at rank 16 (hidden 2048,
    kv 512, 128 experts, moe_intermediate 768).
    """
    r = 16
    dense = (
        2048 * r + r * 4096      # q_proj
        + 2048 * r + r * 512     # k_proj
        + 2048 * r + r * 512     # v_proj
        + 4096 * r + r * 2048    # o_proj
        + 2048 * r + r * 128     # mlp.gate (router)
    )
    expert = (
        128 * (2048 * r + r * 768)   # switch_mlp.gate_proj
        + 128 * (2048 * r + r * 768)  # switch_mlp.up_proj
        + 128 * (768 * r + r * 2048)  # switch_mlp.down_proj
    )
    assert dense + expert == S.PER_BLOCK_TRAINABLE_PARAMS
    assert S.PER_BLOCK_TRAINABLE_PARAMS * 16 == S.EXPECTED_TRAINABLE_PARAMS
    assert S.EXPECTED_TRAINABLE_PARAMS == 281_837_568
    # and it must round to the banner every previous run in this repo printed
    assert f"{S.EXPECTED_TRAINABLE_PARAMS / 1e6:.3f}M" == "281.838M"


def test_expected_lora_tensors_is_16_blocks_x_8_modules_x_2():
    assert S.EXPECTED_LORA_TENSORS == 16 * len(S.EXPECTED_SUFFIXES) * 2 == 256
