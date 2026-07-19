"""Final accept/reject verdict, design.md section 6.4's bar: Track A beats
BOTH base and cpt-v1 by a real (non-noise) margin AND Track B win-or-tie
rate >= 0.5. 'Real margin' threshold set at 0.03 cosine-similarity points --
CPT-v1's null was byte-identical (delta ~0), so any margin above measurement
noise is meaningfully different from that null; 0.03 is a conservative floor,
not a statistically derived value -- note this plainly in the readout, don't
imply more rigor than a single-run threshold actually has."""
MARGIN_THRESHOLD = 0.03
WIN_RATE_THRESHOLD = 0.5


def decide_acceptance(track_a: dict, track_b: dict) -> dict:
    margin_vs_base = track_a["cpt_v2_mean"] - track_a["base_mean"]
    margin_vs_v1 = track_a["cpt_v2_mean"] - track_a["cpt_v1_mean"]
    track_a_ok = margin_vs_base >= MARGIN_THRESHOLD and margin_vs_v1 >= MARGIN_THRESHOLD
    track_b_ok = track_b["cpt_v2_win_or_tie_rate"] >= WIN_RATE_THRESHOLD

    reasons = []
    if not track_a_ok:
        reasons.append(f"track_a margin too small (vs base {margin_vs_base:+.3f}, "
                        f"vs cpt-v1 {margin_vs_v1:+.3f}, need >= {MARGIN_THRESHOLD})")
    if not track_b_ok:
        reasons.append(f"track_b win-or-tie rate {track_b['cpt_v2_win_or_tie_rate']:.2f} "
                        f"below {WIN_RATE_THRESHOLD}")

    return {
        "accepted": track_a_ok and track_b_ok,
        "margin_vs_base": round(margin_vs_base, 4),
        "margin_vs_v1": round(margin_vs_v1, 4),
        "win_or_tie_rate": track_b["cpt_v2_win_or_tie_rate"],
        "reason": "; ".join(reasons) if reasons else "both tracks cleared their bar",
    }


def main():
    import json
    from pathlib import Path

    track_a_raw = json.loads(Path("03-new/results/cpt-v2/track_a.json").read_text())
    n = len(track_a_raw)
    track_a = {
        "base_mean": sum(r["base"] for r in track_a_raw.values()) / n,
        "cpt_v1_mean": sum(r["cpt_v1"] for r in track_a_raw.values()) / n,
        "cpt_v2_mean": sum(r["cpt_v2"] for r in track_a_raw.values()) / n,
    }
    track_b = json.loads(Path("03-new/results/cpt-v2/track_b.json").read_text())
    verdict = decide_acceptance(track_a, track_b)
    print(json.dumps(verdict, indent=2))
    Path("03-new/results/cpt-v2/acceptance.json").write_text(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
