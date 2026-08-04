import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_cpt_leg_spectrum as m


def test_block_indices_parses_layer_number():
    keys = [
        "model.layers.0.self_attn.q_proj.lora_a",
        "model.layers.22.mlp.gate.lora_b",
        "model.layers.47.mlp.switch_mlp.down_proj.lora_a",
    ]
    assert m._block_indices(keys) == [0, 22, 47]


def test_block_indices_ignores_non_layer_keys():
    keys = ["model.embed_tokens.weight", "model.layers.5.self_attn.o_proj.lora_a"]
    assert m._block_indices(keys) == [5]


def test_block_indices_dedupes_and_sorts():
    keys = [
        "model.layers.30.self_attn.q_proj.lora_a",
        "model.layers.30.self_attn.q_proj.lora_b",
        "model.layers.0.mlp.gate.lora_a",
    ]
    assert m._block_indices(keys) == [0, 30]


def test_repo_root_resolves_to_real_repo():
    # This is the exact assertion run_cpt_leg_spectrum.py makes at import time --
    # re-asserted here so a future refactor of the parents[] index breaks a fast
    # test, not a 26-hour run.
    assert (m.REPO_ROOT / "models" / "qwen-q4").exists()


def test_fresh_spec_on_sys_path():
    assert str(m.FRESH_SPEC) in sys.path
    assert (m.FRESH_SPEC / "spectrum_lora_layers.py").exists()
