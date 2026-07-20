from acceptance_readout import decide_acceptance


def test_accepted_when_both_bars_clear():
    track_a = {"base_mean": 0.40, "cpt_v1_mean": 0.41, "cpt_v2_mean": 0.55}
    track_b = {"cpt_v2_win_or_tie_rate": 0.58}
    verdict = decide_acceptance(track_a, track_b)
    assert verdict["accepted"] is True


def test_rejected_when_track_a_margin_too_small():
    track_a = {"base_mean": 0.40, "cpt_v1_mean": 0.41, "cpt_v2_mean": 0.415}
    track_b = {"cpt_v2_win_or_tie_rate": 0.60}
    verdict = decide_acceptance(track_a, track_b)
    assert verdict["accepted"] is False
    assert "track_a" in verdict["reason"]


def test_rejected_when_track_b_below_half():
    track_a = {"base_mean": 0.40, "cpt_v1_mean": 0.41, "cpt_v2_mean": 0.55}
    track_b = {"cpt_v2_win_or_tie_rate": 0.45}
    verdict = decide_acceptance(track_a, track_b)
    assert verdict["accepted"] is False
    assert "track_b" in verdict["reason"]


def test_null_when_both_fail():
    track_a = {"base_mean": 0.40, "cpt_v1_mean": 0.41, "cpt_v2_mean": 0.41}
    track_b = {"cpt_v2_win_or_tie_rate": 0.30}
    verdict = decide_acceptance(track_a, track_b)
    assert verdict["accepted"] is False
    assert "track_a" in verdict["reason"] and "track_b" in verdict["reason"]
