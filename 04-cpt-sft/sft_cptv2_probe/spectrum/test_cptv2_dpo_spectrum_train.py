"""Unit tests for the cptv2 arm's DPO driver -- union + freeze across the resume.

Same fixture strategy as every other suite here: a REAL tiny mlx_lm Qwen3-MoE and
the REAL installed `mlx_lm_lora` call sites (`from_pretrained`, `create_dataset`),
with only the 16GB weight load stubbed.

The property under test is the one the fresh arm gets for free and this arm does
not: DPO resumes from an adapter covering the UNION `picks | {32..47}`, and
`mlx_lm_lora/train.py:527` applies it with `strict=False` AFTER conversion. If
DPO converted only the picks, the 5 CPT-v2-only blocks (80 tensors) would be
dropped without a word -- the same bug the SFT side already fixed, one stage
later. Several tests below deliberately build that broken configuration and
assert it is caught.
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

import mlx_lm_lora.trainer.datasets as _ds
import mlx_lm_lora.utils as _mu

# cptv2_dpo_spectrum_train wires every sys.path entry the rest of these imports
# need (fresh spectrum dir, fresh probe dir), so it must come first.
import cptv2_dpo_spectrum_train as CD      # noqa: I001
import cptv2_spectrum_lora_layers as C     # noqa: E402
import dpo_fixed_train as F                # noqa: E402
import dpo_spectrum_train as D             # noqa: E402
import merge_frozen_keys as M              # noqa: E402
import spectrum_lora_layers as S           # noqa: E402

LORA_PARAMS = {"rank": 16, "scale": 2.0, "dropout": 0.05}
QWEN_Q4 = CD.REPO_ROOT / "models" / "qwen-q4"
PICKS = [0, 1]
BASE = [6, 7]
UNION = [0, 1, 6, 7]
FROZEN = [6, 7]

ROWS = [
    {"prompt": "write a jac walker", "chosen": "walker w { can go with `root entry; }",
     "rejected": "function w() { return 1; }"},
]


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


def fake_adapter(path, layers, value=1.0):
    blob = {}
    for L in layers:
        for sfx in S.EXPECTED_SUFFIXES:
            for ab in ("lora_a", "lora_b"):
                blob[f"model.layers.{L}.{sfx}.{ab}"] = mx.full((2, 2), value + L)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), blob)
    return blob


@pytest.fixture(autouse=True)
def restore_everything():
    stock_fn = tuner_utils.linear_to_lora_layers
    stock_ds = _ds.DPODataset
    yield
    tuner_utils.linear_to_lora_layers = stock_fn
    mlx_lm.lora.linear_to_lora_layers = stock_fn
    _mu.linear_to_lora_layers = stock_fn
    _ds.DPODataset = stock_ds


@pytest.fixture
def stub_load(monkeypatch):
    made = {}

    def fake_load(model, adapter_path=None, **kw):
        m = tiny_model(num_layers=made.get("num_layers", 8))
        made["model"] = m
        return m, object()

    monkeypatch.setattr(_mu, "load", fake_load)
    return made


def dpo_convert(stub_load, layer_count, tmp_path, num_layers=8):
    """The REAL mlx_lm_lora.utils.from_pretrained, real build_lora_config shape."""
    stub_load["num_layers"] = num_layers
    cfg = dict(LORA_PARAMS)
    cfg.update({"use_dora": False, "num_layers": layer_count})
    model, _tok, _af = _mu.from_pretrained(
        model="ignored-by-stub", new_adapter_path=str(tmp_path), lora_config=cfg
    )
    return model


# ==========================================================================
# 1. the union survives into the DPO conversion path
# ==========================================================================
def test_union_is_converted_on_the_real_dpo_path(stub_load, tmp_path):
    CD.apply_patches(PICKS, BASE)
    m = dpo_convert(stub_load, len(PICKS), tmp_path)
    assert S.collect_lora_report(m)["layer_ids"] == UNION


def test_only_the_picks_are_trainable_on_the_dpo_path(stub_load, tmp_path):
    CD.apply_patches(PICKS, BASE)
    m = dpo_convert(stub_load, len(PICKS), tmp_path)
    assert C.trainable_layer_ids(m) == PICKS


def test_dpo_capacity_equals_the_fresh_dpo_arm(stub_load, tmp_path):
    """The union is placement plumbing, never extra budget."""
    D.apply_patches([0, 1])                              # fresh arm, same picks
    fresh = dpo_convert(stub_load, 2, tmp_path)
    n_fresh = sum(v.size for _, v in tree_flatten(fresh.trainable_parameters()))

    for mod, attr in D.patch_sites():                    # re-arm for the cptv2 fn
        setattr(mod, attr, C.make_union_linear_to_lora_layers(PICKS, BASE))
    cptv2 = dpo_convert(stub_load, 2, tmp_path)
    n_cptv2 = sum(v.size for _, v in tree_flatten(cptv2.trainable_parameters()))
    assert n_cptv2 == n_fresh


def test_num_layers_is_checked_against_the_picks_not_the_union(stub_load, tmp_path):
    CD.apply_patches(PICKS, BASE)
    with pytest.raises(ValueError, match="num_layers"):
        dpo_convert(stub_load, len(UNION), tmp_path)     # 4 == len(union)


def test_verify_composition_passes_on_the_dpo_converted_model(stub_load, tmp_path):
    CD.apply_patches(PICKS, BASE)
    m = dpo_convert(stub_load, len(PICKS), tmp_path)
    ok, problems = C.verify_composition(m, PICKS, BASE, expect_tensors=32)
    assert ok, problems


# ==========================================================================
# 2. the resume path -- the bug this arm exists to avoid
# ==========================================================================
def test_a_union_shaped_resume_finds_a_home_for_every_key(stub_load, tmp_path):
    from adapter_config_fix import assert_adapter_keys_present
    sft = tmp_path / "sft.safetensors"
    fake_adapter(sft, UNION)

    CD.apply_patches(PICKS, BASE)
    m = dpo_convert(stub_load, len(PICKS), tmp_path)
    assert assert_adapter_keys_present(m, sft) == 16 * len(UNION)


def test_a_picks_only_dpo_conversion_would_silently_drop_the_frozen_blocks(
        stub_load, tmp_path):
    """The exact regression: convert only the picks, resume the union-shaped SFT
    adapter, and 32 tensors (blocks 6-7) vanish under strict=False without error.
    """
    from adapter_config_fix import assert_adapter_keys_present
    sft = tmp_path / "sft.safetensors"
    fake_adapter(sft, UNION)

    D.apply_patches(PICKS)                        # fresh-arm driver: picks only
    m = dpo_convert(stub_load, len(PICKS), tmp_path)
    with pytest.raises(RuntimeError, match="do not exist in the loaded model"):
        assert_adapter_keys_present(m, sft)
    m.load_weights(str(sft), strict=False)        # and it does NOT complain
    assert C.trainable_layer_ids(m) == PICKS


def test_the_frozen_blocks_load_and_stay_out_of_the_gradients(stub_load, tmp_path):
    """End to end across the resume: block 6/7 weights come IN from the adapter,
    and are absent from trainable_parameters() so DPO cannot move them."""
    sft = tmp_path / "sft.safetensors"
    blob = fake_adapter(sft, UNION, value=5.0)

    CD.apply_patches(PICKS, BASE)
    m = dpo_convert(stub_load, len(PICKS), tmp_path)
    m.load_weights(str(sft), strict=False)
    mx.eval(m.parameters())

    loaded = dict(tree_flatten(m.parameters()))
    for L in FROZEN:
        k = f"model.layers.{L}.self_attn.q_proj.lora_a"
        assert mx.array_equal(loaded[k], blob[k]), k
    trainable = {k for k, _ in tree_flatten(m.trainable_parameters())}
    assert not any(k.startswith(("layers.6.", "layers.7.",
                                 "model.layers.6.", "model.layers.7."))
                   for k in trainable)


def test_the_dpo_save_drops_the_frozen_blocks_and_the_merge_puts_them_back(
        stub_load, tmp_path):
    """Simulates dpo_trainer.py:672-685 (`tree_flatten(trainable_parameters())`)
    and then run_dpo_spectrum.sh's merge. Without the merge the DPO artifact is
    picks-only and the CPT-v2 delta is gone at eval -- §8.3 step 5, one stage on.
    """
    cpt = tmp_path / "cpt.safetensors"
    cpt_blob = fake_adapter(cpt, BASE, value=100.0)
    sft = tmp_path / "sft.safetensors"
    fake_adapter(sft, PICKS, value=1.0)
    # the SFT artifact as the merge leaves it: picks + frozen CPT-v2 blocks
    merged_sft = tmp_path / "sft_merged.safetensors"
    M.merge(sft, cpt, merged_sft, PICKS, base_layers=BASE)

    CD.apply_patches(PICKS, BASE)
    m = dpo_convert(stub_load, len(PICKS), tmp_path)
    m.load_weights(str(merged_sft), strict=False)

    # what the DPO trainer actually writes
    dpo_out = tmp_path / "dpo.safetensors"
    mx.save_safetensors(str(dpo_out), dict(tree_flatten(m.trainable_parameters())))
    assert C.adapter_layer_ids(dpo_out) == PICKS                    # frozen: GONE

    dpo_merged = tmp_path / "dpo_merged.safetensors"
    info = M.merge(dpo_out, cpt, dpo_merged, PICKS, base_layers=BASE)
    assert info["union"] == UNION
    assert info["n_total_keys"] == 16 * len(UNION)
    back = dict(mx.load(str(dpo_merged)))
    for k, v in cpt_blob.items():
        assert mx.array_equal(back[k], v), k                        # bit-identical


# ==========================================================================
# 3. assert_frozen_blocks_match_cpt_v2 -- the pre-flight on the SFT artifact
# ==========================================================================
def test_frozen_check_passes_on_a_correctly_merged_sft_adapter(tmp_path):
    cpt = tmp_path / "cpt.safetensors"
    fake_adapter(cpt, BASE, value=100.0)
    sft = tmp_path / "sft.safetensors"
    fake_adapter(sft, PICKS, value=1.0)
    merged = tmp_path / "merged.safetensors"
    M.merge(sft, cpt, merged, PICKS, base_layers=BASE)

    info = CD.assert_frozen_blocks_match_cpt_v2(merged, PICKS, BASE, cpt)
    assert info["union"] == UNION
    assert info["frozen_layers"] == FROZEN
    assert info["n_frozen_verified"] == 16 * len(FROZEN)


def test_frozen_check_rejects_an_unmerged_picks_only_adapter(tmp_path):
    """The likeliest real failure: run_sft_spectrum.sh's step-5 merge was skipped."""
    cpt = tmp_path / "cpt.safetensors"
    fake_adapter(cpt, BASE)
    sft = tmp_path / "sft.safetensors"
    fake_adapter(sft, PICKS)
    with pytest.raises(RuntimeError, match="expected the union"):
        CD.assert_frozen_blocks_match_cpt_v2(sft, PICKS, BASE, cpt)


def test_frozen_check_rejects_an_adapter_whose_frozen_blocks_moved(tmp_path):
    """Frozen means frozen: if a block in {32..47}\\picks changed value, it was
    trained, so this arm's capacity exceeded the fresh arm's."""
    cpt = tmp_path / "cpt.safetensors"
    fake_adapter(cpt, BASE, value=100.0)
    sft = tmp_path / "sft.safetensors"
    fake_adapter(sft, PICKS, value=1.0)
    merged = tmp_path / "merged.safetensors"
    M.merge(sft, cpt, merged, PICKS, base_layers=BASE)

    blob = dict(mx.load(str(merged)))
    k = f"model.layers.{FROZEN[0]}.self_attn.q_proj.lora_a"
    blob[k] = blob[k] + 1.0
    mx.save_safetensors(str(merged), blob)

    with pytest.raises(RuntimeError, match="differ from"):
        CD.assert_frozen_blocks_match_cpt_v2(merged, PICKS, BASE, cpt)


def test_frozen_check_fails_loudly_when_the_adapter_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        CD.assert_frozen_blocks_match_cpt_v2(tmp_path / "nope.safetensors",
                                             PICKS, BASE, tmp_path / "also-nope")


def test_frozen_check_against_the_real_cpt_v2_checkpoint_and_real_picks(tmp_path):
    """Not a synthetic layout -- the REAL 256-key CPT-v2 file and the REAL 16
    frozen picks, merged the way run_sft_spectrum.sh does it."""
    picks = S.load_layer_ids(CD.DEFAULT_LAYERS_PATH)
    base = C.assert_cpt_v2_shape()
    trained = tmp_path / "sft.safetensors"
    fake_adapter(trained, picks, value=1.0)
    merged = tmp_path / "merged.safetensors"
    M.merge(trained, C.CPT_V2_ADAPTER, merged, picks, base_layers=base)

    info = CD.assert_frozen_blocks_match_cpt_v2(merged, picks)
    assert info["union"] == sorted(set(picks) | set(range(32, 48)))
    assert info["n_keys"] == 16 * 21 == 336
    assert info["n_frozen_verified"] == 16 * 5 == 80     # the 80 tensors at risk


def test_the_real_sft_spectrum_adapter_if_it_exists_is_union_shaped():
    """Runs for real once the cptv2 SFT-spectrum arm has trained + merged; skips
    (rather than lying) before that. This is the exact preflight
    run_dpo_spectrum.sh performs before spending DPO compute."""
    if not CD.DEFAULT_SFT_ADAPTER.exists():
        pytest.skip(f"{CD.DEFAULT_SFT_ADAPTER} not built yet (SFT-spectrum arm "
                    f"has not run) -- run_dpo_spectrum.sh gates on this too")
    picks = S.load_layer_ids(CD.DEFAULT_LAYERS_PATH)
    info = CD.assert_frozen_blocks_match_cpt_v2(CD.DEFAULT_SFT_ADAPTER, picks)
    assert info["n_keys"] == 336


# ==========================================================================
# 4. both patches, on this arm too
# ==========================================================================
def _real_tokenizer():
    if not QWEN_Q4.exists():
        pytest.skip(f"{QWEN_Q4} not present")
    from mlx_lm.utils import load_tokenizer
    return load_tokenizer(QWEN_Q4)


def test_BOTH_patches_active_on_the_cptv2_arm(stub_load, tmp_path):
    tok = _real_tokenizer()
    CD.apply_patches(PICKS, BASE)

    dset = _ds.create_dataset(ROWS, tok, types.SimpleNamespace(train_mode="dpo"))
    assert isinstance(dset, F.FixedDPODataset)
    assert not tok.decode(dset[0]["chosen"]).endswith("<|im_start|>assistant\n")

    m = dpo_convert(stub_load, len(PICKS), tmp_path)
    assert S.collect_lora_report(m)["layer_ids"] == UNION
    assert C.trainable_layer_ids(m) == PICKS


def test_apply_patches_rebinds_all_three_sites_with_the_union_version():
    fn = CD.apply_patches(PICKS, BASE)
    for mod, attr in D.patch_sites():
        assert getattr(mod, attr) is fn
    assert fn.spectrum_layer_ids == PICKS
    assert fn.union_layer_ids == UNION
    assert fn.frozen_layer_ids == FROZEN


def test_patch_report_exposes_the_union_and_frozen_sets():
    CD.apply_patches(PICKS, BASE)
    rep = D.assert_patches_active(PICKS)
    assert rep["union_layer_ids"] == UNION
    assert rep["frozen_layer_ids"] == FROZEN


def test_double_apply_is_refused_on_this_arm_too():
    CD.apply_patches(PICKS, BASE)
    with pytest.raises(RuntimeError, match="already run in this process"):
        CD.apply_patches(PICKS, BASE)


# ==========================================================================
# 5. eval-time composition (§7 over the UNION)
# ==========================================================================
def test_eval_time_num_layers_must_cover_the_union_not_the_picks():
    from adapter_config_fix import spectrum_num_layers
    picks = S.load_layer_ids(CD.DEFAULT_LAYERS_PATH)
    union = C.union_layers(picks, range(32, 48))
    n = spectrum_num_layers(union)
    assert 48 - n == min(union)
    assert set(union) <= set(range(48 - n, 48))


def test_from_pretrained_writes_a_num_layers_the_eval_must_rewrite(stub_load, tmp_path):
    CD.apply_patches(PICKS, BASE)
    dpo_convert(stub_load, len(PICKS), tmp_path)
    cfg = json.loads((tmp_path / "adapter_config.json").read_text())
    assert cfg["num_layers"] == len(PICKS)        # NOT the union -> §7 rewrite needed


# ==========================================================================
# 6. wiring
# ==========================================================================
def test_default_paths_point_at_this_arm():
    assert Path(CD.DEFAULT_LAYERS_PATH).resolve() == (
        Path(C.FRESH_SPECTRUM_DIR) / "configs" / "spectrum_layers.json").resolve()
    assert CD.DEFAULT_SFT_ADAPTER.parent.name == "sft-on-cptv2-spectrum"


def test_dpo_lora_yaml_matches_the_fresh_arms_recipe():
    import yaml
    here = yaml.safe_load(
        (CD.SELF_DIR.parent / "configs" / "dpo_lora.yaml").read_text())
    fresh = yaml.safe_load(
        (Path(C.FRESH_SPECTRUM_DIR).parent / "configs" / "dpo_lora.yaml").read_text())
    assert here["fuse"] is False
    assert here["lora_parameters"] == fresh["lora_parameters"] == {
        "rank": 16, "scale": 2.0, "dropout": 0.05}


def test_dpo_dataset_is_the_654_pair_set():
    p = CD.SELF_DIR.parent / "dataset" / "dpo" / "train.jsonl"
    assert sum(1 for _ in p.open()) == 654
