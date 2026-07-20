from score_track_b import resolve_winner, aggregate


def test_resolve_winner_cpt_v2_first_a_wins():
    assert resolve_winner(order="cpt_v2_first", judge_winner="A") == "cpt_v2"


def test_resolve_winner_cpt_v2_first_b_wins():
    assert resolve_winner(order="cpt_v2_first", judge_winner="B") == "oracle"


def test_resolve_winner_oracle_first_a_wins():
    assert resolve_winner(order="oracle_first", judge_winner="A") == "oracle"


def test_resolve_winner_tie():
    assert resolve_winner(order="cpt_v2_first", judge_winner="tie") == "tie"


def test_aggregate_counts_and_win_rate():
    results = [{"winner": "cpt_v2"}, {"winner": "cpt_v2"}, {"winner": "oracle"}, {"winner": "tie"}]
    agg = aggregate(results)
    assert agg == {"cpt_v2_wins": 2, "oracle_wins": 1, "ties": 1, "total": 4,
                    "cpt_v2_win_or_tie_rate": 0.75}
