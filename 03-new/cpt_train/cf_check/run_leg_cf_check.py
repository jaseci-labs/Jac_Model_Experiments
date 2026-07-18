"""Per-leg CF-check for the CPT-v2 epoch loop (Task 13's operational driver).
Reuses run_cf_check.py's run_model/grade against models/qwen-q4 + the leg's
adapter directory applied on-the-fly (mlx_lm.utils.load's adapter_path param
-- no fuse needed per leg, only the final accepted checkpoint gets fused).

IMPORTANT CONTRACT NOTE: mlx_lm.utils.load(..., adapter_path=...) routes into
load_adapters, which treats adapter_path as a DIRECTORY -- it opens
adapter_path / "adapter_config.json" and loads adapter_path / "adapters.safetensors".
Passing a numbered checkpoint FILE (e.g. 0000570_adapters.safetensors) raises
NotADirectoryError. run_cpt_leg.py (Task 8) writes to a SHARED adapter directory
across all legs (e.g. 03-new/adapters/cpt-v2/) -- both a numbered file (for
resume-point tracking) AND the unnumbered rolling-latest adapters.safetensors +
adapter_config.json that mlx_lm.tuner.trainer.train()'s own final-save step
always writes. Since the directory is shared and overwritten each leg, its
CURRENT adapters.safetensors always reflects the MOST RECENTLY COMPLETED leg's
weights -- exactly what a CF-check running right after that leg wants.

CALLER CAVEAT (authoritative for Task 13's operational loop): this function
must be called right after a leg's training completes, before the next leg's
training starts -- because the directory is shared and gets overwritten by
the next leg's run_cpt_leg.py call. Calling it later (e.g. after leg N+1 has
already started/finished) will silently CF-check the wrong leg's weights.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_cf_check import run_model


def run_leg_cf_check(adapter_dir: str) -> tuple:
    results = run_model("cpt-v2-leg", "models/qwen-q4", adapter_path=adapter_dir)
    passed = sum(r["pass"] for r in results)
    return passed, len(results)


def _save_full_results(adapter_dir: str, results: list):
    """Best-effort persistence of full per-task CF-check output (generated
    code + text, not just pass/fail) so a leg review can actually read what
    the model produced, not just a count. Keyed by the just-completed leg's
    global step (read back from the same numbered checkpoint the CF-check
    itself just evaluated) so results accumulate across legs without
    collision. Never raises -- must not break run_epoch_loop.py's stdout
    parsing of the "CF-check: N/M" line if persistence fails for any reason
    (e.g. results dir not writable)."""
    try:
        ckpts = sorted(Path(adapter_dir).glob("*_adapters.safetensors"))
        step = ckpts[-1].name.split("_")[0] if ckpts else "unknown"
        out_dir = Path(__file__).resolve().parents[2] / "results" / "cpt-v2"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"cf_check_{step}.json").write_text(json.dumps(results, indent=2))
    except Exception as e:
        print(f"  (non-fatal: failed to persist full CF-check results: {e})")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "adapter_dir",
        help="the shared adapter directory (e.g. 03-new/adapters/cpt-v2), whose "
             "adapters.safetensors reflects the most recently completed leg. Must "
             "be called right after that leg's training completes, before the "
             "next leg's training starts.",
    )
    args = ap.parse_args()
    results = run_model("cpt-v2-leg", "models/qwen-q4", adapter_path=args.adapter_dir)
    passed, total = sum(r["pass"] for r in results), len(results)
    _save_full_results(args.adapter_dir, results)
    print(f"CF-check: {passed}/{total}")
