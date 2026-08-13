#!/usr/bin/env python3
"""Arm 5 three-way comparison statistics (cpt-spectrum-plan.md §7.2/§7.3).

Two-proportion z-test, this project's standing convention, identical to the one
used for `2026-08-spectrum-vs-stock-comparison.md` / `spectrum-results-summary.md`:

    z = (p1 - p2) / sqrt( p_pool * (1 - p_pool) * (1/n1 + 1/n2) )
    p_pool = (x1 + x2) / (n1 + n2)

Two-sided p from the normal CDF.  All n = 855 (the frozen code-graded holdout).

Arms A (cptv2 stock, incumbent) and B (cptv2 spectrum union/freeze) and the
fresh-spectrum reference are QUOTED from their own committed metrics files, not
retyped -- so the report cannot drift from the artifacts.  Arm C (cptspectrum,
the clean chain) is read from this arm's own metrics files.

Writes JSON to stdout and to --out.
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
N = 855


def overall_rows(path):
    p = ROOT / path
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        if r.get("category") == "__overall__" and r.get("total") == N:
            out.append(r)
    return out


def last_full(path):
    """Highest-step full-holdout row (the SFT final / DPO final)."""
    rows = overall_rows(path)
    if not rows:
        return None
    return max(rows, key=lambda r: r.get("step", -1))


def dpo_rows(path):
    """DPO eval writes two full rows with distinguishing offsets:
    final = step + 1_000_000, best = step + 500_000 (eval_dpo_spectrum.sh)."""
    rows = overall_rows(path)
    final = best = None
    for r in rows:
        s = r.get("step", 0)
        if s >= 1_000_000:
            if final is None or s > final.get("step", 0):
                final = r
        elif s >= 500_000:
            if best is None or s > best.get("step", 0):
                best = r
    if best is None:
        best = final  # eval script skips the duplicate when best == last
    return best, final


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def ztest(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    pool = (x1 + x2) / (n1 + n2)
    denom = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if denom == 0:
        return {"delta_pp": 0.0, "z": 0.0, "p": 1.0, "significant": "No (exact tie)"}
    z = (p1 - p2) / denom
    p = 2 * (1 - norm_cdf(abs(z)))
    if p < 0.05:
        sig = "Yes"
    elif p < 0.10:
        sig = "Marginal"
    else:
        sig = "No"
    if x1 == x2:
        sig = "No (exact tie)"
    return {
        "delta_pp": round((p1 - p2) * 100, 2),
        "z": round(z, 3),
        "p": round(p, 4) if p >= 0.0001 else "<0.0001",
        "significant": sig,
    }


# ---------------------------------------------------------------- arm sources
P = "model-experiments/04-cpt-sft"
SOURCES = {
    "A_cptv2_stock": {
        "sft": f"{P}/sft_cptv2_probe/results/sft/metrics_functional.jsonl",
        "dpo": f"{P}/sft_cptv2_probe/results/dpo-v4-nofuse/metrics_functional.jsonl",
    },
    "B_cptv2_spectrum": {
        "sft": f"{P}/sft_cptv2_probe/results/sft-spectrum/metrics_functional.jsonl",
        "dpo": f"{P}/sft_cptv2_probe/results/dpo-spectrum/metrics_functional.jsonl",
    },
    "C_cptspectrum": {
        "sft": f"{P}/sft_cptspectrum_probe/results/sft-spectrum/metrics_functional.jsonl",
        "dpo": f"{P}/sft_cptspectrum_probe/results/dpo-spectrum/metrics_functional.jsonl",
    },
    "ref_fresh_spectrum": {
        "sft": f"{P}/sft_fresh_probe/results/sft-spectrum/metrics_functional.jsonl",
        "dpo": f"{P}/sft_fresh_probe/results/dpo-spectrum/metrics_functional.jsonl",
    },
    "ref_fresh_stock": {
        "sft": f"{P}/sft_fresh_probe/results/sft/metrics_functional.jsonl",
        "dpo": f"{P}/sft_fresh_probe/results/dpo-nofuse/metrics_functional.jsonl",
    },
}

# Published numbers from five-arms-overview.md §2, used to cross-check that the
# metrics files we read still say what the committed reports say. A mismatch is
# reported, never silently preferred either way.
PUBLISHED = {
    "A_cptv2_stock": {"sft": 621, "dpo_best": 613, "dpo_final": 555},
    "B_cptv2_spectrum": {"sft": 603, "dpo_best": 613, "dpo_final": 590},
    "ref_fresh_spectrum": {"sft": 639, "dpo_best": 634, "dpo_final": 622},
    "ref_fresh_stock": {"sft": 597, "dpo_best": 597, "dpo_final": 531},
}


def collect():
    arms = {}
    for name, src in SOURCES.items():
        sft = last_full(src["sft"])
        best, final = dpo_rows(src["dpo"])
        arms[name] = {
            "sft": sft.get("runs") if sft else None,
            "sft_step": sft.get("step") if sft else None,
            "dpo_best": best.get("runs") if best else None,
            "dpo_best_step": (best.get("step") - 500_000) if best and best.get("step", 0) >= 500_000 and best.get("step", 0) < 1_000_000 else None,
            "dpo_final": final.get("runs") if final else None,
            "n": N,
            "sources": src,
        }
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "arm5_stats.json"))
    args = ap.parse_args()

    arms = collect()

    # cross-check against published numbers
    warnings = []
    for name, pub in PUBLISHED.items():
        got = arms.get(name, {})
        for stage, expect in pub.items():
            actual = got.get(stage)
            if actual is None:
                warnings.append(f"{name}.{stage}: no full-holdout row found in {SOURCES[name]}")
            elif actual != expect:
                warnings.append(
                    f"{name}.{stage}: metrics file says {actual}/855 but the committed "
                    f"reports say {expect}/855 -- using the METRICS FILE value, flag in the report"
                )

    tests = {}
    C = arms["C_cptspectrum"]
    for stage in ("sft", "dpo_best", "dpo_final"):
        for opp in ("A_cptv2_stock", "B_cptv2_spectrum", "ref_fresh_spectrum"):
            c, o = C.get(stage), arms[opp].get(stage)
            if c is None or o is None:
                tests[f"C_vs_{opp}__{stage}"] = {"error": "missing data"}
                continue
            tests[f"C_vs_{opp}__{stage}"] = ztest(c, N, o, N)

    result = {
        "n": N,
        "method": "two-proportion z-test, two-sided, alpha=0.05; z=(p1-p2)/sqrt(p_pool(1-p_pool)(1/n1+1/n2))",
        "arms": arms,
        "tests": tests,
        "warnings": warnings,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
