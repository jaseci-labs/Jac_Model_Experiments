"""Per-leg CF-check for the CPT-v2 epoch loop (Task 13's operational driver).
Reuses run_cf_check.py's run_model/grade against models/qwen-q4 + the leg's
adapter checkpoint applied on-the-fly (mlx_lm.utils.load's adapter_path param
-- no fuse needed per leg, only the final accepted checkpoint gets fused)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_cf_check import run_model


def run_leg_cf_check(adapter_checkpoint: str) -> tuple:
    results = run_model("cpt-v2-leg", "models/qwen-q4", adapter_path=adapter_checkpoint)
    passed = sum(r["pass"] for r in results)
    return passed, len(results)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter_checkpoint")
    args = ap.parse_args()
    passed, total = run_leg_cf_check(args.adapter_checkpoint)
    print(f"CF-check: {passed}/{total}")
