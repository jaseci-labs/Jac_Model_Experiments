"""Unit tests for the cptv2 arm's union+freeze driver.

Same fixture strategy as the fresh arm's suite: a REAL tiny mlx_lm Qwen3-MoE, so
the `SwitchLinear -> LoRASwitchLinear` branch and mlx's freeze/trainable
bookkeeping are genuinely exercised without loading 16GB.

The thing under test is spectrum-plan.md §8.3: convert the union
`picks | {32..47}`, load CPT-v2 into it, freeze `{32..47} \\ picks`, and keep the
trainable budget exactly equal to the fresh arm's. Every assertion below is about
one of those four properties or about the merge that undoes step 3 at save time.
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

import mlx_lm.lora
import mlx_lm.tuner.utils as tuner_utils
from mlx_lm.models.qwen3_moe import Model as Qwen3MoeModel
from mlx_lm.models.qwen3_moe import ModelArgs

import cptv2_spectrum_lora_layers as C
import merge_frozen_keys as M
import spectrum_lora_layers as S

LORA_CFG = {"rank": 16, "scale": 2.0, "dropout": 0.05}


def tiny_model(num_layers=8, num_experts=4):
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


def converted(picks, base, num_layers=8):
    m = tiny_model(num_layers=num_layers)
    m.freeze()
    C.make_union_linear_to_lora_layers(picks, base)(m, len(picks), LORA_CFG)
    return m


def fake_adapter(path, layers, value=1.0):
    """A safetensors file shaped like a real adapter over `layers`."""
    blob = {}
    for L in layers:
        for sfx in S.EXPECTED_SUFFIXES:
            for ab in ("lora_a", "lora_b"):
                blob[f"model.layers.{L}.{sfx}.{ab}"] = mx.full((2, 2), value + L)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), blob)
    return blob


@pytest.fixture(autouse=True)
def restore_upstream():
    stock = tuner_utils.linear_to_lora_layers
    yield
    tuner_utils.linear_to_lora_layers = stock
    mlx_lm.lora.linear_to_lora_layers = stock


@pytest.fixture(autouse=True)
def dpo_package_not_loaded(monkeypatch):
    """`C.apply_patch()` refuses to run with `mlx_lm_lora` imported -- correct for
    the SFT path, whose process never imports it. `test_cptv2_dpo_spectrum_train.py`
    DOES, and a directory-wide pytest run shares one process. Isolate here rather
    than weaken the guard; the guard itself is still covered by
    `test_apply_patch_refuses_when_the_dpo_package_is_loaded`.
    """
    monkeypatch.delitem(sys.modules, "mlx_lm_lora", raising=False)


# --- the real CPT-v2 checkpoint on disk ------------------------------------
def test_the_real_cpt_v2_checkpoint_is_the_artifact_8_3_describes():
    """Not a doc claim -- read the file. The union/freeze split is sized off it."""
    ids = C.assert_cpt_v2_shape()
    assert ids == list(range(32, 48))
    assert len(mx.load(str(C.CPT_V2_ADAPTER)).keys()) == 256


def test_cpt_v2_shape_check_rejects_a_wrong_layer_span(tmp_path):
    f = tmp_path / "adapters.safetensors"
    fake_adapter(f, range(16, 32))       # 256 keys, but the wrong blocks
    with pytest.raises(RuntimeError, match="covers layers"):
        C.assert_cpt_v2_shape(f)


def test_cpt_v2_shape_check_rejects_a_wrong_key_count(tmp_path):
    f = tmp_path / "adapters.safetensors"
    fake_adapter(f, range(32, 40))       # right span start, half the blocks
    with pytest.raises(RuntimeError, match="tensors"):
        C.assert_cpt_v2_shape(f)


def test_cpt_v2_shape_check_fails_loudly_when_the_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        C.assert_cpt_v2_shape(tmp_path / "nope.safetensors")


# --- set arithmetic --------------------------------------------------------
def test_union_and_frozen_split_the_real_selection_correctly():
    picks = json.loads(
        (Path(C.FRESH_SPECTRUM_DIR) / "configs" / "spectrum_layers.json").read_text()
    )["layers"]
    base = list(range(32, 48))
    assert C.union_layers(picks, base) == sorted(set(picks) | set(base))
    assert C.frozen_layers(picks, base) == sorted(set(base) - set(picks))
    # disjointness is what makes the merge safe
    assert not set(C.frozen_layers(picks, base)) & set(picks)
    assert set(C.union_layers(picks, base)) == set(picks) | set(C.frozen_layers(picks, base))


def test_frozen_set_is_empty_when_picks_already_cover_the_base():
    assert C.frozen_layers([4, 5, 6, 7], [4, 5]) == []
    assert C.union_layers([4, 5, 6, 7], [4, 5]) == [4, 5, 6, 7]


# --- conversion + freeze (§8.3 steps 1, 3, 4) ------------------------------
def test_union_is_converted_not_just_the_picks():
    m = converted(picks=[0, 1], base=[6, 7])
    assert S.collect_lora_report(m)["layer_ids"] == [0, 1, 6, 7]


def test_only_the_picks_are_trainable():
    m = converted(picks=[0, 1], base=[6, 7])
    assert C.trainable_layer_ids(m) == [0, 1]


def test_frozen_union_members_are_converted_but_contribute_no_gradients():
    """The property the whole design rests on: block 7 has LoRA modules (so
    CPT-v2's keys land) yet none of them appear in trainable_parameters()."""
    m = converted(picks=[0, 1], base=[6, 7])
    rep = S.collect_lora_report(m)
    assert 7 in rep["layer_ids"]
    assert rep["suffixes_by_layer"][7] == list(S.EXPECTED_SUFFIXES)
    trainable = {k for k, _ in tree_flatten(m.trainable_parameters())}
    assert not any(k.startswith("layers.7.") or k.startswith("model.layers.7.")
                   for k in trainable)


def test_capacity_equals_the_fresh_arm_at_the_same_picks():
    """§8.3 step 4. If the union counted as capacity this arm would train more
    parameters than sft-on-fresh-spectrum and the four-way table is meaningless."""
    fresh = tiny_model(num_layers=8)
    fresh.freeze()
    S.make_linear_to_lora_layers([0, 1])(fresh, 2, LORA_CFG)
    n_fresh = sum(v.size for _, v in tree_flatten(fresh.trainable_parameters()))

    cptv2 = converted(picks=[0, 1], base=[6, 7])
    n_cptv2 = sum(v.size for _, v in tree_flatten(cptv2.trainable_parameters()))
    assert n_cptv2 == n_fresh


def test_capacity_is_independent_of_how_much_the_picks_overlap_the_base():
    counts = set()
    for picks in ([0, 1], [0, 6], [6, 7], [3, 7]):
        m = converted(picks=picks, base=[6, 7])
        counts.add(sum(v.size for _, v in tree_flatten(m.trainable_parameters())))
    assert len(counts) == 1, counts


def test_picks_inside_the_base_stay_trainable():
    """A pick that happens to be a CPT-v2 block must NOT get frozen -- it is
    initialized from CPT-v2 and then trained, exactly as sft-on-cptv2's were."""
    m = converted(picks=[6, 7], base=[6, 7])
    assert C.trainable_layer_ids(m) == [6, 7]
    assert S.collect_lora_report(m)["layer_ids"] == [6, 7]


def test_num_layers_is_checked_against_the_PICKS_not_the_union():
    m = tiny_model(num_layers=8)
    m.freeze()
    fn = C.make_union_linear_to_lora_layers([0, 1], [6, 7])
    with pytest.raises(ValueError, match="num_layers"):
        fn(m, 4, LORA_CFG)          # 4 == len(union); must still be rejected
    fn(m, 2, LORA_CFG)              # 2 == len(picks); accepted


def test_a_resume_load_finds_a_home_for_every_base_key(tmp_path):
    """The reason the union exists. With picks-only conversion, strict=False
    would silently drop the base keys; this asserts it does not."""
    from adapter_config_fix import assert_adapter_keys_present
    f = tmp_path / "cpt.safetensors"
    fake_adapter(f, [6, 7])

    m = converted(picks=[0, 1], base=[6, 7])
    assert assert_adapter_keys_present(m, f) == 32

    picks_only = tiny_model(num_layers=8)
    picks_only.freeze()
    S.make_linear_to_lora_layers([0, 1])(picks_only, 2, LORA_CFG)
    with pytest.raises(RuntimeError, match="do not exist in the loaded model"):
        assert_adapter_keys_present(picks_only, f)


# --- verify_composition ----------------------------------------------------
def test_verify_composition_passes_on_a_correct_build():
    m = converted(picks=[0, 1], base=[6, 7])
    ok, problems = C.verify_composition(m, [0, 1], [6, 7], expect_tensors=32)
    assert ok, problems


def test_verify_composition_catches_a_picks_only_conversion():
    m = tiny_model(num_layers=8)
    m.freeze()
    S.make_linear_to_lora_layers([0, 1])(m, 2, LORA_CFG)      # no union
    ok, problems = C.verify_composition(m, [0, 1], [6, 7], expect_tensors=32)
    assert not ok
    assert any("union" in p for p in problems)


def test_verify_composition_catches_a_missing_freeze():
    """Union converted but nothing frozen -> the arm would train 4 blocks on a
    16-block budget. This is the failure that silently inflates capacity."""
    m = tiny_model(num_layers=8)
    m.freeze()
    S.make_linear_to_lora_layers([0, 1, 6, 7])(m, 4, LORA_CFG)   # union, no freeze
    ok, problems = C.verify_composition(m, [0, 1], [6, 7], expect_tensors=32)
    assert not ok
    assert any("TRAINABLE layer set" in p for p in problems)


def test_verify_composition_catches_a_wrong_capacity():
    m = converted(picks=[0, 1], base=[6, 7])
    ok, problems = C.verify_composition(m, [0, 1], [6, 7], expect_tensors=32,
                                        expect_trainable=1)
    assert not ok
    assert any("trainable params" in p for p in problems)


# --- the rebind ------------------------------------------------------------
def test_apply_patch_rebinds_both_attributes_with_the_union_version():
    stock = tuner_utils.linear_to_lora_layers
    C.apply_patch([0, 1], [6, 7])
    assert tuner_utils.linear_to_lora_layers is not stock
    assert mlx_lm.lora.linear_to_lora_layers is tuner_utils.linear_to_lora_layers

    m = tiny_model(num_layers=8)
    m.freeze()
    mlx_lm.lora.linear_to_lora_layers(m, 2, LORA_CFG)   # the training call site
    assert S.collect_lora_report(m)["layer_ids"] == [0, 1, 6, 7]
    assert C.trainable_layer_ids(m) == [0, 1]


def test_apply_patch_refuses_when_the_dpo_package_is_loaded(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "mlx_lm_lora", object())
    with pytest.raises(RuntimeError, match="mlx_lm_lora"):
        C.apply_patch([0, 1], [6, 7])


# --- config wiring ---------------------------------------------------------
CFG_DIR = Path(__file__).resolve().parent / "configs"


def _yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())


def test_sft_spectrum_yaml_resumes_from_the_real_cpt_v2_checkpoint():
    cfg = _yaml(CFG_DIR / "sft_spectrum.yaml")
    resume = C.REPO_ROOT / cfg["resume_adapter_file"]
    assert resume.resolve() == C.CPT_V2_ADAPTER.resolve()
    assert resume.exists()


def test_sft_spectrum_yaml_differs_from_the_incumbent_in_exactly_adapter_path():
    """The arm-vs-arm contrast must be placement only. Any second difference
    here is a confound the four-way table cannot separate."""
    spec = _yaml(CFG_DIR / "sft_spectrum.yaml")
    base = _yaml(CFG_DIR.parent.parent / "configs" / "sft.yaml")
    diff = {k for k in set(spec) | set(base) if spec.get(k) != base.get(k)}
    assert diff == {"adapter_path"}, diff
    assert spec["adapter_path"].endswith("adapters/sft-on-cptv2-spectrum")


def test_sft_spectrum_yaml_matches_the_fresh_spectrum_arm_on_every_recipe_key():
    """Both spectrum arms must share the recipe; only data/paths/resume differ."""
    cptv2 = _yaml(CFG_DIR / "sft_spectrum.yaml")
    fresh = _yaml(Path(C.FRESH_SPECTRUM_DIR) / "configs" / "sft_spectrum.yaml")
    arm_specific = {"data", "adapter_path", "resume_adapter_file"}
    for k in set(cptv2) | set(fresh):
        if k in arm_specific:
            continue
        assert cptv2.get(k) == fresh.get(k), f"{k}: {cptv2.get(k)} != {fresh.get(k)}"


def test_num_layers_in_the_yaml_equals_the_frozen_selection_size():
    cfg = _yaml(CFG_DIR / "sft_spectrum.yaml")
    picks = json.loads((CFG_DIR / "spectrum_layers.json").read_text())["layers"]
    assert cfg["num_layers"] == len(picks) == 16


def test_the_two_arms_share_one_selection_file_not_two_copies():
    """A copy could drift; a symlink cannot. The picks are model-derived, so the
    arms must be provably using the same ones."""
    link = CFG_DIR / "spectrum_layers.json"
    fresh = Path(C.FRESH_SPECTRUM_DIR) / "configs" / "spectrum_layers.json"
    assert link.is_symlink()
    assert link.resolve() == fresh.resolve()


# --- merge_frozen_keys (§8.3 step 5) ---------------------------------------
def test_merge_restores_the_frozen_cpt_v2_blocks(tmp_path):
    cpt = tmp_path / "cpt.safetensors"
    fake_adapter(cpt, range(32, 48), value=100.0)
    trained = tmp_path / "adapters.safetensors"
    picks = [0, 22, 34, 47]
    fake_adapter(trained, picks, value=1.0)

    out = tmp_path / "merged.safetensors"
    info = M.merge(trained, cpt, out, picks, base_layers=list(range(32, 48)))

    assert info["merged_frozen_layers"] == sorted(set(range(32, 48)) - set(picks))
    assert info["union"] == sorted(set(picks) | set(range(32, 48)))
    assert info["n_total_keys"] == 16 * len(info["union"])

    back = dict(mx.load(str(out)))
    # a trained pick keeps its TRAINED value, not the CPT-v2 one
    assert float(back["model.layers.34.self_attn.q_proj.lora_a"][0, 0]) == 1.0 + 34
    # a frozen block comes back bit-identical from CPT-v2
    assert float(back["model.layers.33.self_attn.q_proj.lora_a"][0, 0]) == 100.0 + 33


def test_merge_refuses_an_adapter_that_is_not_the_picks(tmp_path):
    cpt = tmp_path / "cpt.safetensors"
    fake_adapter(cpt, range(32, 48))
    trained = tmp_path / "adapters.safetensors"
    fake_adapter(trained, [1, 2, 3])          # not the picks
    with pytest.raises(RuntimeError, match="covers layers"):
        M.merge(trained, cpt, tmp_path / "m.safetensors", [0, 22, 34, 47],
                base_layers=list(range(32, 48)))


def test_merge_is_a_no_op_on_layer_count_when_picks_cover_the_base(tmp_path):
    cpt = tmp_path / "cpt.safetensors"
    fake_adapter(cpt, [6, 7])
    trained = tmp_path / "adapters.safetensors"
    fake_adapter(trained, [6, 7])
    info = M.merge(trained, cpt, tmp_path / "m.safetensors", [6, 7],
                   base_layers=[6, 7])
    assert info["merged_frozen_layers"] == []
    assert info["n_total_keys"] == 32


def test_merged_adapter_is_covered_by_the_eval_time_num_layers_rewrite(tmp_path):
    """§7 + §8.3 step 5 have to compose: the rewritten num_layers must cover the
    UNION, since that is what the merged artifact contains."""
    from adapter_config_fix import spectrum_num_layers
    picks = json.loads((CFG_DIR / "spectrum_layers.json").read_text())["layers"]
    union = C.union_layers(picks, range(32, 48))
    n = spectrum_num_layers(union)
    assert 48 - n == min(union)
    assert set(union) <= set(range(48 - n, 48))
