"""Unit tests for dpo_spectrum_train.py -- DPO on the Spectrum-picked layers.

Same fixture strategy as the SFT suites: a REAL tiny mlx_lm Qwen3-MoE, so the
`SwitchLinear -> LoRASwitchLinear` branch is genuinely exercised, and the REAL
installed `mlx_lm_lora` code -- `from_pretrained`, `create_dataset` -- is what
gets called, with only the 16GB model load stubbed out. No mocked
`linear_to_lora_layers`, no mocked dataset class: the things under test are the
actual upstream call sites.

The three questions this file exists to answer:

  1. Does the layer rebind reach the DPO training path (not just SFT)?
     `mlx_lm_lora/utils.py:15` is a `from ... import`, so it holds an independent
     binding -- test_rebinding_only_mlx_lm_does_not_reach_the_dpo_package proves
     that failure mode is real, and the rest prove the driver avoids it.
  2. Are BOTH monkey-patches live at once? Composing two patches is where one
     silently loses -- tested functionally (a real DPODataset build AND a real
     from_pretrained conversion in the same process), in both application orders.
  3. Does the `--resume-adapter-file` path still find a home for every key?
     `train.py:527` uses strict=False and will not tell you.
"""

import json
import types
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

import mlx_lm.lora
import mlx_lm.tuner.utils as tuner_utils
from mlx_lm.models.qwen3_moe import Model as Qwen3MoeModel
from mlx_lm.models.qwen3_moe import ModelArgs

import mlx_lm_lora
import mlx_lm_lora.train as _mt
import mlx_lm_lora.trainer.datasets as _ds
import mlx_lm_lora.utils as _mu

# dpo_spectrum_train FIRST: it is what puts sft_fresh_probe/ (where
# dpo_fixed_train.py lives) on sys.path, exactly as it does under the runner.
import dpo_spectrum_train as D          # noqa: I001
import dpo_fixed_train as F             # noqa: E402
import spectrum_lora_layers as S        # noqa: E402

LORA_PARAMS = {"rank": 16, "scale": 2.0, "dropout": 0.05}
QWEN_Q4 = D.REPO_ROOT / "models" / "qwen-q4"

# a real DPO row shape, matching sft_fresh_probe/dataset/dpo/train.jsonl's schema
ROWS = [
    {"prompt": "write a jac walker", "chosen": "walker w { can go with `root entry; }",
     "rejected": "function w() { return 1; }"},
    {"prompt": "declare a node", "chosen": "node n { has x: int = 0; }",
     "rejected": "class n: pass"},
]


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
def restore_everything():
    """Every test gets stock upstream back -- all four patchable attributes."""
    stock_fn = tuner_utils.linear_to_lora_layers
    stock_ds = _ds.DPODataset
    yield
    tuner_utils.linear_to_lora_layers = stock_fn
    mlx_lm.lora.linear_to_lora_layers = stock_fn
    _mu.linear_to_lora_layers = stock_fn
    _ds.DPODataset = stock_ds


@pytest.fixture
def stock_fn():
    """The genuine upstream `linear_to_lora_layers`, captured before any patch."""
    return tuner_utils.linear_to_lora_layers


@pytest.fixture
def stub_load(monkeypatch):
    """Replace only the 16GB weight load inside mlx_lm_lora.utils.from_pretrained.

    Everything else in that function -- freeze(), the linear_to_lora_layers call,
    the adapter_config.json write -- runs for real.
    """
    made = {}

    def fake_load(model, adapter_path=None, **kw):
        m = tiny_model(num_layers=made.get("num_layers", 4))
        made["model"] = m
        return m, object()

    monkeypatch.setattr(_mu, "load", fake_load)
    return made


def dpo_convert(stub_load, layer_count, num_layers=4, tmp_path=None):
    """Call the REAL from_pretrained with the REAL build_lora_config shape."""
    stub_load["num_layers"] = num_layers
    cfg = dict(LORA_PARAMS)
    cfg.update({"use_dora": False, "num_layers": layer_count})
    model, _tok, _af = _mu.from_pretrained(
        model="ignored-by-stub", new_adapter_path=str(tmp_path), lora_config=cfg
    )
    return model


# ==========================================================================
# 1. the import-graph question, answered against the installed package
# ==========================================================================
def test_mlx_lm_lora_reuses_mlx_lms_function_object_and_has_no_conversion_of_its_own():
    """It is the SAME object -- so spectrum_lora_layers.py's verbatim copy of the
    body is the correct body for the DPO path too; no second copy is needed."""
    assert _mu.linear_to_lora_layers is tuner_utils.linear_to_lora_layers
    assert mlx_lm.lora.linear_to_lora_layers is tuner_utils.linear_to_lora_layers


def test_rebinding_only_mlx_lm_does_not_reach_the_dpo_package():
    """...but it is bound by a `from ... import`, so `mlx_lm_lora.utils` holds an
    INDEPENDENT reference. This is the exact half-patch the third rebind site
    exists to prevent; without this test the third site looks like belt-and-braces.
    """
    sentinel = D.S.make_linear_to_lora_layers([0, 2])
    tuner_utils.linear_to_lora_layers = sentinel
    mlx_lm.lora.linear_to_lora_layers = sentinel
    assert _mu.linear_to_lora_layers is not sentinel
    assert not hasattr(_mu.linear_to_lora_layers, "spectrum_layer_ids")


def test_the_dpo_training_path_goes_through_from_pretrained():
    src = D._stripped_source(_mt.run)
    assert "from_pretrained(" in src
    assert D.CONVERT_CALL_MARKER in D._stripped_source(_mu.from_pretrained)


def test_upstream_resumes_AFTER_converting_and_with_strict_false():
    """The reason an unconverted block's resume keys vanish silently, and the
    reason the cptv2 arm must convert the union."""
    src = D._stripped_source(_mt.train_model)
    assert D.RESUME_MARKER in src
    run_src = D._stripped_source(_mt.run)
    assert run_src.find("from_pretrained(") < run_src.find("train_model(")


def test_the_dpo_trainer_saves_only_trainable_parameters():
    """Why the cptv2 arm needs merge_frozen_keys.py on the DPO artifacts too."""
    import inspect

    from mlx_lm_lora.trainer import dpo_trainer

    src = "".join(inspect.getsource(dpo_trainer.train_dpo).split())
    assert "adapter_weights=dict(tree_flatten(model.trainable_parameters()))" in src


# ==========================================================================
# 2. guards
# ==========================================================================
def test_guard_passes_against_the_installed_mlx_lm_lora():
    D.assert_dpo_upstream_still_matches()   # must not raise


def test_guard_pins_the_version_and_hashes_it_was_written_against():
    assert D.UPSTREAM_MLX_LM_LORA_VERSION == "2.1.0"
    assert mlx_lm_lora.__version__ == "2.1.0"
    import hashlib
    assert hashlib.sha256(
        D._stripped_source(_mu.from_pretrained).encode()
    ).hexdigest() == D.FROM_PRETRAINED_SHA256
    assert hashlib.sha256(
        D._stripped_source(_mt.train_model).encode()
    ).hexdigest() == D.TRAIN_MODEL_SHA256


def test_guard_fails_when_from_pretrained_stops_converting(monkeypatch):
    def impostor(model, adapter_path=None, new_adapter_path=None,
                 lora_config=None, quantized_load=None):
        return None, None, None
    monkeypatch.setattr(_mu, "from_pretrained", impostor)
    with pytest.raises(RuntimeError, match="no longer calls"):
        D.assert_dpo_upstream_still_matches()


def test_guard_fails_when_the_version_moves(monkeypatch):
    monkeypatch.setattr(mlx_lm_lora, "__version__", "9.9.9")
    with pytest.raises(RuntimeError, match="pinned"):
        D.assert_dpo_upstream_still_matches()


def test_guard_fails_when_the_dpo_package_grows_its_own_conversion(monkeypatch):
    def other(model, num_layers, config, use_dora=False):
        pass
    monkeypatch.setattr(_mu, "linear_to_lora_layers", other)
    with pytest.raises(RuntimeError, match="not the same object"):
        D.assert_dpo_upstream_still_matches()


# ==========================================================================
# 3. the rebind reaches the DPO path
# ==========================================================================
def test_apply_patches_rebinds_all_three_sites_to_one_object():
    fn = D.apply_patches([0, 2])
    assert tuner_utils.linear_to_lora_layers is fn
    assert mlx_lm.lora.linear_to_lora_layers is fn
    assert _mu.linear_to_lora_layers is fn
    assert fn.spectrum_layer_ids == [0, 2]


def test_layer_rebind_applies_during_a_real_dpo_shaped_conversion(stub_load, tmp_path):
    """The headline claim: `mlx_lm_lora.utils.from_pretrained` -- the function
    `mlx_lm_lora.train.run()` actually calls for DPO -- converts the PICKS."""
    D.apply_patches([0, 2])
    m = dpo_convert(stub_load, layer_count=2, tmp_path=tmp_path)
    assert S.collect_lora_report(m)["layer_ids"] == [0, 2]      # NOT [2, 3]


def test_stock_dpo_path_converts_the_trailing_slice(stub_load, tmp_path):
    """Control: with no rebind the same harness measures {n-2, n-1}. Proof the
    assertion above is measuring placement and not a constant."""
    m = dpo_convert(stub_load, layer_count=2, tmp_path=tmp_path)
    assert S.collect_lora_report(m)["layer_ids"] == [2, 3]


def test_the_dpo_conversion_hits_the_switchlinear_branch(stub_load, tmp_path):
    D.apply_patches([1, 3])
    m = dpo_convert(stub_load, layer_count=2, tmp_path=tmp_path)
    rep = S.collect_lora_report(m)
    assert rep["n_switch"] == 6                      # 3 stacked expert projections/layer
    assert rep["n_lora_tensors"] == 2 * 8 * 2
    for L in (1, 3):
        assert rep["suffixes_by_layer"][L] == list(S.EXPECTED_SUFFIXES)


def test_dpo_capacity_is_placement_independent(stub_load, tmp_path):
    counts = set()
    for ids in ([0, 1], [0, 3], [2, 3]):
        for mod, attr in D.patch_sites():
            setattr(mod, attr, tuner_utils.linear_to_lora_layers)
        fn = S.make_linear_to_lora_layers(ids)
        for mod, attr in D.patch_sites():
            setattr(mod, attr, fn)
        m = dpo_convert(stub_load, layer_count=2, tmp_path=tmp_path)
        counts.add(S.collect_lora_report(m)["trainable_params"])
    assert len(counts) == 1, counts


def test_num_layers_mismatch_still_fails_loudly_on_the_dpo_path(stub_load, tmp_path):
    """`--num-layers 16` on the CLI vs a 15-pick selection must not silently win."""
    D.apply_patches([0, 2])
    with pytest.raises(ValueError, match="num_layers"):
        dpo_convert(stub_load, layer_count=3, tmp_path=tmp_path)


def test_from_pretrained_writes_a_num_layers_the_eval_must_rewrite(stub_load, tmp_path):
    """The §7 trap survives into DPO: the artifact claims num_layers=len(picks),
    so load_adapters would rebuild the TRAILING slice at eval time."""
    D.apply_patches([0, 2])
    dpo_convert(stub_load, layer_count=2, tmp_path=tmp_path)
    cfg = json.loads((tmp_path / "adapter_config.json").read_text())
    assert cfg["num_layers"] == 2                    # NOT the picks
    from adapter_config_fix import spectrum_num_layers
    assert spectrum_num_layers([0, 2], num_blocks=4) == 4    # what the eval needs


# ==========================================================================
# 4. BOTH patches, simultaneously -- the composition risk
# ==========================================================================
def _real_tokenizer():
    if not QWEN_Q4.exists():
        pytest.skip(f"{QWEN_Q4} not present")
    from mlx_lm.utils import load_tokenizer
    return load_tokenizer(QWEN_Q4)


def _dpo_dataset_via_upstream(tok):
    """Build through `create_dataset`, which resolves DPODataset as a module
    global at call time -- the real code path `mlx_lm_lora.train.run()` takes."""
    cfg = types.SimpleNamespace(train_mode="dpo")
    return _ds.create_dataset(ROWS, tok, cfg)


def test_chat_template_fix_is_live_through_the_real_create_dataset():
    tok = _real_tokenizer()
    D.apply_patches([0, 2])
    dset = _dpo_dataset_via_upstream(tok)
    assert isinstance(dset, F.FixedDPODataset)
    for i in range(len(ROWS)):
        for side in ("chosen", "rejected"):
            txt = tok.decode(dset[i][side])
            assert not txt.endswith("<|im_start|>assistant\n")
            assert txt.endswith("<|im_end|>\n")


def test_stock_create_dataset_still_shows_the_bug():
    """Control for the test above -- if upstream were already fixed, the patch
    would be redundant and the composition question moot."""
    tok = _real_tokenizer()
    dset = _dpo_dataset_via_upstream(tok)
    assert not isinstance(dset, F.FixedDPODataset)
    assert tok.decode(dset[0]["chosen"]).endswith("<|im_start|>assistant\n")


def test_BOTH_patches_are_active_in_the_same_process(stub_load, tmp_path):
    """The composition test. One apply_patches(), then exercise BOTH real upstream
    paths back to back: the dataset builder AND the model converter. A carelessly
    composed pair passes one of these and fails the other.
    """
    tok = _real_tokenizer()
    D.apply_patches([0, 2])

    dset = _dpo_dataset_via_upstream(tok)
    assert isinstance(dset, F.FixedDPODataset)
    assert not tok.decode(dset[0]["chosen"]).endswith("<|im_start|>assistant\n")

    m = dpo_convert(stub_load, layer_count=2, tmp_path=tmp_path)
    assert S.collect_lora_report(m)["layer_ids"] == [0, 2]


def test_layer_patch_does_not_undo_the_chat_template_patch(stub_load, tmp_path):
    """Order A: dataset patch first, then layers. Asserts the second does not
    reset the first (e.g. by re-importing or reloading the datasets module)."""
    tok = _real_tokenizer()
    F.apply_patch()
    fn = S.make_linear_to_lora_layers([0, 2])
    for mod, attr in D.patch_sites():
        setattr(mod, attr, fn)
    D.assert_patches_active([0, 2])
    assert isinstance(_dpo_dataset_via_upstream(tok), F.FixedDPODataset)
    assert S.collect_lora_report(
        dpo_convert(stub_load, 2, tmp_path=tmp_path))["layer_ids"] == [0, 2]


def test_chat_template_patch_does_not_undo_the_layer_patch(stub_load, tmp_path):
    """Order B: layers first, then dataset."""
    tok = _real_tokenizer()
    fn = S.make_linear_to_lora_layers([0, 2])
    for mod, attr in D.patch_sites():
        setattr(mod, attr, fn)
    F.apply_patch()
    D.assert_patches_active([0, 2])
    assert isinstance(_dpo_dataset_via_upstream(tok), F.FixedDPODataset)
    assert S.collect_lora_report(
        dpo_convert(stub_load, 2, tmp_path=tmp_path))["layer_ids"] == [0, 2]


def test_assert_patches_active_catches_a_missing_chat_template_fix():
    fn = S.make_linear_to_lora_layers([0, 2])
    for mod, attr in D.patch_sites():
        setattr(mod, attr, fn)                        # layers only
    with pytest.raises(RuntimeError, match="chat-template fix is not live"):
        D.assert_patches_active([0, 2])


def test_assert_patches_active_catches_a_missing_layer_site(stock_fn, stub_load,
                                                            tmp_path):
    """The half-patch that would otherwise look fine: two of three sites rebound,
    and the DPO conversion silently stays on the trailing slice. This is exactly
    what "rebind mlx_lm only, let the from-import pick it up" produces."""
    D.apply_patches([0, 2])
    _mu.linear_to_lora_layers = stock_fn                 # undo just the DPO site
    with pytest.raises(RuntimeError, match=r"mlx_lm_lora\.utils"):
        D.assert_patches_active([0, 2])
    # and the failure it would have caused, demonstrated
    assert S.collect_lora_report(
        dpo_convert(stub_load, 2, tmp_path=tmp_path))["layer_ids"] == [2, 3]


def test_assert_patches_active_catches_three_different_objects():
    D.apply_patches([0, 2])
    _mu.linear_to_lora_layers = S.make_linear_to_lora_layers([1, 3])   # a DIFFERENT
    with pytest.raises(RuntimeError, match="SAME object"):
        D.assert_patches_active()


def test_assert_patches_active_catches_the_wrong_layer_ids():
    D.apply_patches([0, 2])
    with pytest.raises(RuntimeError, match="layer ids"):
        D.assert_patches_active([1, 3])


def test_double_apply_is_refused_with_an_honest_message():
    """Re-applying would make the guards -- which read the STOCK source -- raise
    'upstream changed', which is false. Fail with the true reason instead."""
    D.apply_patches([0, 2])
    with pytest.raises(RuntimeError, match="already run in this process"):
        D.apply_patches([0, 2])


def test_apply_patches_requires_exactly_one_of_ids_or_replacement():
    with pytest.raises(ValueError):
        D.apply_patches()
    with pytest.raises(ValueError):
        D.apply_patches([0, 2], replacement=S.make_linear_to_lora_layers([0, 2]))


def test_apply_patches_accepts_an_injected_replacement(stub_load, tmp_path):
    """The extension point the cptv2 arm uses -- guards, three sites, dataset fix
    and post-condition are shared, only the conversion differs."""
    fn = S.make_linear_to_lora_layers([1, 3])
    got = D.apply_patches(replacement=fn)
    assert got is fn
    assert _mu.linear_to_lora_layers is fn
    assert _ds.DPODataset is F.FixedDPODataset
    m = dpo_convert(stub_load, layer_count=2, tmp_path=tmp_path)
    assert S.collect_lora_report(m)["layer_ids"] == [1, 3]


# ==========================================================================
# 5. the resume path (strict=False will not tell you)
# ==========================================================================
def fake_adapter(path, layers, value=1.0):
    blob = {}
    for L in layers:
        for sfx in S.EXPECTED_SUFFIXES:
            for ab in ("lora_a", "lora_b"):
                blob[f"model.layers.{L}.{sfx}.{ab}"] = mx.full((2, 2), value + L)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), blob)
    return blob


def test_resume_from_the_sft_spectrum_adapter_finds_a_home_for_every_key(
        stub_load, tmp_path):
    """The fresh arm is safe by construction -- DPO converts the same picks SFT
    trained. Asserted, not assumed."""
    from adapter_config_fix import assert_adapter_keys_present
    sft = tmp_path / "sft.safetensors"
    fake_adapter(sft, [0, 2])

    D.apply_patches([0, 2])
    m = dpo_convert(stub_load, layer_count=2, tmp_path=tmp_path)
    assert assert_adapter_keys_present(m, sft) == 32


def test_resuming_a_spectrum_adapter_on_the_stock_trailing_slice_would_drop_keys(
        stub_load, tmp_path):
    """What running the STOCK DPO recipe on a spectrum SFT adapter would do:
    block 0's 16 tensors silently vanish under strict=False."""
    from adapter_config_fix import assert_adapter_keys_present
    sft = tmp_path / "sft.safetensors"
    fake_adapter(sft, [0, 2])

    m = dpo_convert(stub_load, layer_count=2, tmp_path=tmp_path)   # unpatched
    with pytest.raises(RuntimeError, match="do not exist in the loaded model"):
        assert_adapter_keys_present(m, sft)
    # and prove strict=False is indeed silent about it
    m.load_weights(str(sft), strict=False)


# ==========================================================================
# 6. wiring: paths, layers file, recipe constants
# ==========================================================================
def test_default_layers_path_is_the_frozen_selection():
    assert Path(D.DEFAULT_LAYERS_PATH) == D.SELF_DIR / "configs" / "spectrum_layers.json"
    picks = S.load_layer_ids(D.DEFAULT_LAYERS_PATH)
    assert len(picks) == 16
    assert picks == [0, 22, 23, 27, 30, 34, 36, 37, 38, 39, 41, 42, 43, 44, 45, 47]


def test_repo_root_resolves_to_the_real_repo():
    assert (D.REPO_ROOT / "model-experiments" / "04-cpt-sft").is_dir()


def test_dpo_lora_yaml_is_the_corrected_recipe():
    """§3 of the comparison report: fuse:false, rank 16 / scale 2.0 / dropout .05.
    This DPO-spectrum run must match the stock recipe exactly except for layers."""
    import yaml
    cfg = yaml.safe_load((D.PROBE_DIR / "configs" / "dpo_lora.yaml").read_text())
    assert cfg["fuse"] is False
    assert cfg["lora_parameters"] == {"rank": 16, "scale": 2.0, "dropout": 0.05}


def test_dpo_dataset_is_the_654_pair_set():
    p = D.PROBE_DIR / "dataset" / "dpo" / "train.jsonl"
    assert sum(1 for _ in p.open()) == 654
