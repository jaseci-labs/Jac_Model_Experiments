#!/usr/bin/env python3
"""Turn per-matrix SNR into the 16 decoder-block indices that get LoRA.

Implements spectrum-plan.md §5.1's PRIMARY rule verbatim:

  For each decoder block L, take the SNR of its five DENSE matrices only --
  self_attn.{q,k,v,o}_proj and mlp.gate. Z-score each module type's 48 values
  across layers independently (so q_proj is only ever compared against other
  layers' q_proj, never against o_proj at a different aspect ratio), average the
  five z-scores into one per-layer score, rank descending, take the top 16.

The 384 expert matrices per layer are EXCLUDED from the ranking and RECORDED
alongside it (§5.1 "recorded but not used for selection"), because §4.3's MoE
question is unresolved and 384 uninterpretable numbers per layer would otherwise
decide the ranking outright. LoRA is still applied to all 8 module types in each
selected layer -- exclusion is from the ranking, not from training.

The q_proj-only FALLBACK (§5.1) is available as
`dense_layer_scores(scan, module_types=("self_attn.q_proj",))` and must be
recorded in SELECTION.md with its reason if used.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from snr_scan import DENSE_MODULE_TYPES, EXPERT_MODULE_TYPES

TRAILING16: List[int] = list(range(32, 48))


def zscore(values: Sequence[float]) -> List[float]:
    """Population z-score (ddof=0). A constant vector maps to all zeros."""
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return []
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    if var <= 0.0:
        return [0.0] * n
    sd = var ** 0.5
    return [(v - mean) / sd for v in vals]


def dense_layer_scores(scan: dict,
                       module_types: Sequence[str] = DENSE_MODULE_TYPES) -> Dict[int, float]:
    """Per-layer mean of per-module-type z-scores (spectrum-plan.md §5.1)."""
    dense = scan["dense"]
    layers = sorted(int(k) for k in dense)
    if not layers:
        raise ValueError("scan['dense'] is empty -- nothing to rank")

    per_type_z: Dict[str, List[float]] = {}
    for mt in module_types:
        raw = []
        for L in layers:
            entry = dense[str(L)].get(mt)
            if entry is None:
                raise ValueError(
                    f"layer {L} has no SNR for module type {mt!r}; the scan is "
                    f"incomplete and the ranking would silently compare unequal sets"
                )
            raw.append(float(entry["snr"]))
        per_type_z[mt] = zscore(raw)

    return {
        L: sum(per_type_z[mt][i] for mt in module_types) / len(module_types)
        for i, L in enumerate(layers)
    }


def expert_layer_stats(scan: dict) -> Dict[int, Dict[str, float]]:
    """Per-layer mean/median over every recorded expert-matrix SNR."""
    out: Dict[int, Dict[str, float]] = {}
    for key, mods in scan.get("experts", {}).items():
        vals: List[float] = []
        for mt in EXPERT_MODULE_TYPES:
            vals.extend(float(v) for v in mods.get(mt, []))
        if not vals:
            continue
        out[int(key)] = {
            "mean": sum(vals) / len(vals),
            "median": float(statistics.median(vals)),
            "n": len(vals),
        }
    return out


def rank_layers(layer_scores: Dict[int, float], k: int = 16) -> List[int]:
    """Top-k layer indices, descending by score, ties broken by LOWER index."""
    if k > len(layer_scores):
        raise ValueError(f"asked for {k} layers but only {len(layer_scores)} scored")
    ordered = sorted(layer_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [L for L, _ in ordered[:k]]


def boundary_tie(layer_scores: Dict[int, float], k: int = 16) -> bool:
    """True if the k-th and (k+1)-th scores are exactly equal (a real tie-break)."""
    if k >= len(layer_scores):
        return False
    ordered = sorted(layer_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered[k - 1][1] == ordered[k][1]


def overlap_with_trailing16(picks: Sequence[int]) -> int:
    return len(set(picks) & set(TRAILING16))


def build_layer_scores_artifact(scan: dict,
                                module_types: Sequence[str] = DENSE_MODULE_TYPES) -> dict:
    """snr/layer_scores.json -- dense scores AND expert stats, side by side."""
    scores = dense_layer_scores(scan, module_types)
    experts = expert_layer_stats(scan)
    dense = scan["dense"]
    return {
        "rule": "dense-z-mean" if tuple(module_types) == DENSE_MODULE_TYPES else "q_proj-only",
        "module_types": list(module_types),
        "source_snapshot": scan.get("snapshot_revision"),
        "generated": datetime.now(timezone.utc).isoformat(),
        "expert_sample": scan.get("expert_sample"),
        "layers": {
            str(L): {
                "dense_z_mean": scores[L],
                "dense_snr": {mt: float(dense[str(L)][mt]["snr"])
                              for mt in dense[str(L)] if mt in DENSE_MODULE_TYPES},
                "expert": experts.get(L),
            }
            for L in sorted(scores)
        },
    }


def build_selection_artifact(scan: dict, picks_by_rank: Sequence[int], rule: str,
                             spectrum_rev: str) -> dict:
    """configs/spectrum_layers.json -- spectrum-plan.md §5.3, frozen once training starts."""
    picks = [int(i) for i in picks_by_rank]
    if len(set(picks)) != len(picks):
        raise ValueError(f"duplicate layer index in picks: {picks}")
    return {
        "layers": sorted(picks),
        "layers_by_rank": picks,
        "count": len(picks),
        "rule": rule,
        "source_snapshot": scan.get("snapshot_revision"),
        "spectrum_rev": spectrum_rev,
        "generated": datetime.now(timezone.utc).isoformat(),
        "overlap_with_trailing16": overlap_with_trailing16(picks),
        "tie_at_boundary": False,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snr", required=True, help="snr/snr_raw.json from snr_scan.py")
    ap.add_argument("--layer-scores-out", required=True, help="snr/layer_scores.json")
    ap.add_argument("--selection-out", required=True, help="configs/spectrum_layers.json")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--rule", choices=["dense-z-mean", "q_proj-only"], default="dense-z-mean")
    ap.add_argument("--spectrum-rev", default="self-implemented:snr_scan.py",
                    help="provenance string for the ranking implementation")
    args = ap.parse_args(argv)

    scan = json.loads(Path(args.snr).read_text())
    module_types = (DENSE_MODULE_TYPES if args.rule == "dense-z-mean"
                    else ("self_attn.q_proj",))
    scores = dense_layer_scores(scan, module_types)
    picks = rank_layers(scores, k=args.k)

    ls = build_layer_scores_artifact(scan, module_types)
    sel = build_selection_artifact(scan, picks, args.rule, args.spectrum_rev)
    sel["tie_at_boundary"] = boundary_tie(scores, k=args.k)

    for path, blob in ((args.layer_scores_out, ls), (args.selection_out, sel)):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(blob, indent=2))
        print(f"wrote {p}")

    print(f"picks (rank order): {sel['layers_by_rank']}")
    print(f"picks (ascending):  {sel['layers']}")
    print(f"overlap_with_trailing16: {sel['overlap_with_trailing16']}/{args.k}")
    if sel["tie_at_boundary"]:
        print("!!! exact tie at the k-th position -- broken by lower layer index (§5.2)")
    if sel["layers"] == TRAILING16:
        print("!!! picks == {32..47}: Spectrum endorses the existing default. "
              "Per spectrum-workflow.md Phase 2, SKIP training and write it up "
              "as a null-by-construction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
