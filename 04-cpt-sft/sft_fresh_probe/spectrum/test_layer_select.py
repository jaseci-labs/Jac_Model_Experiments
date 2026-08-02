"""Unit tests for layer_select.py -- spectrum-plan.md §5's aggregation rule."""

import json

import pytest

from layer_select import (
    TRAILING16,
    boundary_tie,
    build_layer_scores_artifact,
    build_selection_artifact,
    dense_layer_scores,
    expert_layer_stats,
    overlap_with_trailing16,
    rank_layers,
    zscore,
)
from snr_scan import DENSE_MODULE_TYPES


def _scan(dense_snr, experts=None):
    """dense_snr: {layer: {module_type: snr}} -> a scan_snapshot-shaped dict."""
    return {
        "snapshot": "/fake/snap",
        "snapshot_revision": "deadbeef",
        "generated": "2026-08-02T00:00:00+00:00",
        "num_layers": len(dense_snr),
        "dense": {
            str(L): {mt: {"snr": v, "shape": [4, 2], "n_signal": 1}
                     for mt, v in mods.items()}
            for L, mods in dense_snr.items()
        },
        "experts": experts or {},
        "expert_sample": None,
        "include_experts": True,
        "wall_clock_s": 1.0,
        "peak_rss_mb": 10.0,
    }


def test_zscore_centers_and_scales():
    z = zscore([1.0, 2.0, 3.0])
    assert z == pytest.approx([-1.2247448, 0.0, 1.2247448])
    assert sum(z) == pytest.approx(0.0)


def test_zscore_of_constant_is_all_zero():
    assert zscore([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]


def test_dense_layer_scores_zscores_each_module_type_independently():
    # q_proj is on a 1000x scale, mlp.gate on a 1x scale. If the aggregation
    # failed to z-score per module type, q_proj would decide the ranking alone.
    scan = _scan({
        0: {"self_attn.q_proj": 1000.0, "self_attn.k_proj": 1.0,
            "self_attn.v_proj": 1.0, "self_attn.o_proj": 1.0, "mlp.gate": 1.0},
        1: {"self_attn.q_proj": 2000.0, "self_attn.k_proj": 2.0,
            "self_attn.v_proj": 2.0, "self_attn.o_proj": 2.0, "mlp.gate": 2.0},
        2: {"self_attn.q_proj": 3000.0, "self_attn.k_proj": 3.0,
            "self_attn.v_proj": 3.0, "self_attn.o_proj": 3.0, "mlp.gate": 3.0},
    })
    s = dense_layer_scores(scan)
    assert set(s) == {0, 1, 2}
    # every module type has the same within-type ordering, so the mean z-score
    # is exactly the shared z-score of [1,2,3]
    assert s[0] == pytest.approx(-1.2247448)
    assert s[1] == pytest.approx(0.0)
    assert s[2] == pytest.approx(1.2247448)


def test_dense_layer_scores_rejects_a_layer_missing_a_module_type():
    scan = _scan({
        0: {"self_attn.q_proj": 1.0, "self_attn.k_proj": 1.0,
            "self_attn.v_proj": 1.0, "self_attn.o_proj": 1.0, "mlp.gate": 1.0},
        1: {"self_attn.q_proj": 2.0},
    })
    with pytest.raises(ValueError, match=r"self_attn\.k_proj"):
        dense_layer_scores(scan)


def test_qproj_only_fallback_rule_is_supported():
    scan = _scan({
        0: {"self_attn.q_proj": 5.0, "self_attn.k_proj": 0.0, "self_attn.v_proj": 0.0,
            "self_attn.o_proj": 0.0, "mlp.gate": 0.0},
        1: {"self_attn.q_proj": 1.0, "self_attn.k_proj": 9.0, "self_attn.v_proj": 9.0,
            "self_attn.o_proj": 9.0, "mlp.gate": 9.0},
    })
    s = dense_layer_scores(scan, module_types=("self_attn.q_proj",))
    assert s[0] > s[1]


def test_rank_layers_takes_top_k_descending():
    scores = {0: 0.1, 1: 5.0, 2: 3.0, 3: -1.0, 4: 4.0}
    assert rank_layers(scores, k=3) == [1, 4, 2]


def test_rank_layers_breaks_exact_ties_by_lower_layer_index():
    scores = {9: 1.0, 3: 1.0, 7: 1.0, 1: 0.0}
    assert rank_layers(scores, k=3) == [3, 7, 9]
    assert boundary_tie(scores, k=3) is False
    # a tie that straddles the k-th position IS a boundary tie
    assert boundary_tie({0: 2.0, 1: 1.0, 2: 1.0}, k=2) is True


def test_rank_layers_rejects_k_larger_than_the_pool():
    with pytest.raises(ValueError):
        rank_layers({0: 1.0}, k=16)


def test_overlap_with_trailing16():
    assert TRAILING16 == list(range(32, 48))
    assert overlap_with_trailing16(list(range(32, 48))) == 16
    assert overlap_with_trailing16(list(range(0, 16))) == 0
    assert overlap_with_trailing16([31, 32, 33, 47]) == 3


def test_expert_layer_stats_records_mean_and_median():
    scan = _scan(
        {0: {mt: 1.0 for mt in DENSE_MODULE_TYPES}},
        experts={"0": {"mlp.experts.gate_proj": [1.0, 2.0, 6.0],
                       "mlp.experts.up_proj": [4.0],
                       "mlp.experts.down_proj": [0.0, 2.0]}},
    )
    st = expert_layer_stats(scan)
    assert st[0]["n"] == 6
    assert st[0]["mean"] == pytest.approx(15.0 / 6)
    assert st[0]["median"] == pytest.approx(2.0)


def test_selection_artifact_matches_the_spec_shape(tmp_path):
    dense = {}
    for L in range(48):
        # layers 0..15 score highest -> picks are maximally disjoint from 32..47
        v = float(48 - L)
        dense[L] = {mt: v for mt in DENSE_MODULE_TYPES}
    scan = _scan(dense)
    scores = dense_layer_scores(scan)
    picks = rank_layers(scores, k=16)
    art = build_selection_artifact(scan, picks, rule="dense-z-mean",
                                   spectrum_rev="self-implemented:snr_scan.py")

    assert art["count"] == 16
    assert art["layers"] == sorted(range(16))
    assert art["layers_by_rank"] == list(range(16))
    assert art["rule"] == "dense-z-mean"
    assert art["source_snapshot"] == "deadbeef"
    assert art["overlap_with_trailing16"] == 0
    assert art["tie_at_boundary"] is False
    assert set(art) >= {"layers", "layers_by_rank", "count", "rule",
                        "source_snapshot", "spectrum_rev", "generated",
                        "overlap_with_trailing16", "tie_at_boundary"}
    json.dumps(art)  # must be serialisable exactly as written to disk


def test_layer_scores_artifact_carries_dense_and_expert_side_by_side():
    scan = _scan(
        {L: {mt: float(L) for mt in DENSE_MODULE_TYPES} for L in range(3)},
        experts={str(L): {"mlp.experts.gate_proj": [float(L)]} for L in range(3)},
    )
    art = build_layer_scores_artifact(scan)
    assert art["rule"] == "dense-z-mean"
    assert set(art["layers"]["1"]) >= {"dense_z_mean", "dense_snr", "expert"}
    assert art["layers"]["1"]["expert"]["mean"] == pytest.approx(1.0)
    assert art["layers"]["2"]["dense_snr"]["self_attn.q_proj"] == pytest.approx(2.0)
