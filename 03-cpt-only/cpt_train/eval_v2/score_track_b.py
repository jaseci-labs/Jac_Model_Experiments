"""Resolves blinded judge verdicts back to real labels (using the order
recorded by track_b_judge_prep.py) and aggregates win/loss/tie -- design.md
section 6.4's acceptance bar checks cpt_v2_win_or_tie_rate >= 0.5 here."""


def resolve_winner(order: str, judge_winner: str) -> str:
    if judge_winner == "tie":
        return "tie"
    is_a = judge_winner == "A"
    cpt_v2_was_a = order == "cpt_v2_first"
    cpt_v2_won = is_a == cpt_v2_was_a
    return "cpt_v2" if cpt_v2_won else "oracle"


def aggregate(results: list) -> dict:
    total = len(results)
    cpt_v2_wins = sum(1 for r in results if r["winner"] == "cpt_v2")
    oracle_wins = sum(1 for r in results if r["winner"] == "oracle")
    ties = sum(1 for r in results if r["winner"] == "tie")
    return {
        "cpt_v2_wins": cpt_v2_wins, "oracle_wins": oracle_wins, "ties": ties, "total": total,
        "cpt_v2_win_or_tie_rate": round((cpt_v2_wins + ties) / total, 4) if total else 0.0,
    }
