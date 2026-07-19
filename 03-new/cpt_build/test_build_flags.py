from pathlib import Path

from build_cpt import rehearsal_target, resolve_out_dir, ROOT


def test_rehearsal_target_v1_default_matches_old_behavior():
    # old hardcoded behavior: jac_tokens // 4
    jac_tokens = 3_050_000
    assert rehearsal_target(jac_tokens, 0.25) == jac_tokens // 4


def test_rehearsal_target_v2_fraction():
    jac_tokens = 2_059_000
    target = rehearsal_target(jac_tokens, 0.111)
    # ~10% of (jac_tokens + target), within rounding
    total = jac_tokens + target
    assert abs(target / total - 0.10) < 0.01


def test_resolve_out_dir_default():
    assert resolve_out_dir(None) == ROOT / "03-new" / "dataset" / "cpt"


def test_resolve_out_dir_override():
    assert resolve_out_dir("03-new/dataset/cpt-v2") == ROOT / "03-new" / "dataset" / "cpt-v2"
