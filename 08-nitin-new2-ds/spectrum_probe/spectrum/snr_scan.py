#!/usr/bin/env python3
"""Marchenko-Pastur bulk-edge SNR over a bf16 HF safetensors snapshot.

WHY THIS EXISTS INSTEAD OF `cognitivecomputations/spectrum`
-----------------------------------------------------------
`spectrum-plan.md` §4 specifies the computation completely (bulk edge from
random-matrix theory, above-edge energy over in-bulk energy, per 2-D weight
matrix), and §4.3 already decided the MoE handling this repo will use (rank the
5 dense projection types only, z-scored per module type; expert scores recorded
but not used for selection). Depending on an unvendored, unpinned external
script for a number that decides which layers get trained would break the same
rule `dpo_fixed_train.py` and `run_cpt_leg.py` follow -- compose your own driver,
pin what you depend on. The formula below is ~40 lines and unit-tested; the
external repo is not reachable from this environment and would be unpinned if it
were.

THE FORMULA (spectrum-plan.md §4, steps 1-4)
--------------------------------------------
For a real 2-D weight matrix W of shape (m, n):

  gamma      = m / n                                  (aspect ratio)
  sigma^2    = Var(W entries)                         (noise scale)
  S          = (1/n) W W^T                            (m x m)
  lambda_i   = s_i^2 / n                              (s_i = singular values of W)
  lambda_+   = sigma^2 * (1 + sqrt(gamma))^2          (upper bulk edge)

Everything at or below lambda_+ is what a structureless matrix of the same shape
and scale produces on its own; everything above it is learned structure.

  SNR = sum(lambda_i : lambda_i > lambda_+) / sum(lambda_i : lambda_i <= lambda_+)

The (m - n) exactly-zero eigenvalues that exist when m > n contribute 0 to the
denominator, so they are harmless to include.

WHY A HAND-ROLLED SAFETENSORS READER
------------------------------------
The snapshot is BF16 and 57GB on a 48GB machine, so it must be read one tensor
at a time and never materialised whole (spectrum-plan.md §11 risk 4). Neither
available streaming backend works here: `safetensors.safe_open(framework="np")`
and `framework="mlx"` both raise `TypeError: data type 'bfloat16' not understood`
on this file (verified 2026-08-02, safetensors 0.7.0), and the torch backend
would drag a second tensor framework into an mlx-only recipe. Parsing the
8-byte length + JSON header and mmap-ing a single tensor's byte range is ~30
lines of stdlib, decodes BF16 by the exact bit relation (bf16 is the top 16 bits
of an IEEE-754 float32), and holds at most one matrix at a time -- the largest
being `model.embed_tokens.weight` at 151936x2048 (~1.2GB once widened to F32),
and every matrix §5.1 actually needs is <=33MB as F32.
"""

from __future__ import annotations

import argparse
import json
import mmap
import re
import resource
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DENSE_MODULE_TYPES: Tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate",
)

EXPERT_MODULE_TYPES: Tuple[str, ...] = (
    "mlp.experts.gate_proj",
    "mlp.experts.up_proj",
    "mlp.experts.down_proj",
)

# numpy-native safetensors dtypes. BF16 is handled separately (no numpy dtype).
_NP_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
}

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)\.weight$")
_EXPERT_RE = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$")


# --------------------------------------------------------------------------
# streaming safetensors reader
# --------------------------------------------------------------------------
def read_safetensors_header(path) -> Tuple[dict, int]:
    """Return (tensor header dict, byte offset where the data section starts).

    Reads only the header -- a few hundred KB -- regardless of file size.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        (hdr_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hdr_len))
    header.pop("__metadata__", None)
    return header, 8 + hdr_len


def load_tensor_f32(path, name: str, header: Optional[dict] = None,
                    data_start: Optional[int] = None) -> np.ndarray:
    """Load exactly one tensor as float32, mmap-ing only its own byte range.

    The returned array owns its memory (the mmap is closed before returning),
    so nothing keeps the shard mapped after the call.
    """
    path = Path(path)
    if header is None or data_start is None:
        header, data_start = read_safetensors_header(path)
    if name not in header:
        raise KeyError(f"{name!r} not in {path.name}")
    meta = header[name]
    dtype = meta["dtype"]
    start, end = meta["data_offsets"]
    shape = tuple(meta["shape"])

    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if dtype == "BF16":
                raw = np.frombuffer(mm, dtype=np.uint16,
                                    count=(end - start) // 2, offset=data_start + start)
                arr = (raw.astype(np.uint32) << 16).view(np.float32)
            elif dtype in _NP_DTYPES:
                np_dt = _NP_DTYPES[dtype]
                raw = np.frombuffer(mm, dtype=np_dt,
                                    count=(end - start) // np.dtype(np_dt).itemsize,
                                    offset=data_start + start)
                arr = raw.astype(np.float32)
            else:
                raise ValueError(f"unsupported safetensors dtype {dtype!r} for {name!r}")
            del raw  # drop the mmap-backed view before unmapping
        finally:
            mm.close()
    return arr.reshape(shape)


# --------------------------------------------------------------------------
# Marchenko-Pastur SNR
# --------------------------------------------------------------------------
def mp_upper_edge(m: int, n: int, sigma_sq: float) -> float:
    """Upper edge of the MP bulk for an (m, n) matrix with entry variance sigma^2."""
    gamma = m / n
    return float(sigma_sq * (1.0 + np.sqrt(gamma)) ** 2)


def matrix_snr(w: np.ndarray) -> Dict[str, object]:
    """Above-bulk-edge energy over in-bulk energy for one 2-D weight matrix."""
    w = np.asarray(w)
    if w.ndim != 2:
        raise ValueError(f"matrix_snr needs a 2-D matrix, got shape {w.shape}")
    m, n = w.shape
    gamma = m / n
    sigma_sq = float(np.var(w, dtype=np.float64))
    if sigma_sq <= 0.0:
        return {
            "snr": 0.0, "signal_energy": 0.0, "noise_energy": 0.0,
            "n_signal": 0, "n_eigenvalues": int(min(m, n)),
            "upper_edge": 0.0, "gamma": gamma, "sigma_sq": 0.0,
            "shape": [int(m), int(n)],
        }

    s = np.linalg.svd(w.astype(np.float32), compute_uv=False).astype(np.float64)
    eig = (s * s) / n                      # eigenvalues of (1/n) W W^T
    edge = mp_upper_edge(m, n, sigma_sq)
    above = eig > edge
    signal = float(eig[above].sum())
    noise = float(eig[~above].sum())
    if noise > 0.0:
        snr = signal / noise
    else:
        snr = float("inf") if signal > 0.0 else 0.0
    return {
        "snr": snr,
        "signal_energy": signal,
        "noise_energy": noise,
        "n_signal": int(above.sum()),
        "n_eigenvalues": int(eig.size),
        "upper_edge": edge,
        "gamma": gamma,
        "sigma_sq": sigma_sq,
        "shape": [int(m), int(n)],
    }


# --------------------------------------------------------------------------
# whole-snapshot scan
# --------------------------------------------------------------------------
def _classify(name: str) -> Optional[Tuple[int, str, Optional[int]]]:
    """('model.layers.7.self_attn.q_proj.weight') -> (7, 'self_attn.q_proj', None)

    Expert matrices collapse onto their MLX stacked name (spectrum-plan.md §4.3):
    all 128 `mlp.experts.<E>.down_proj` in layer L are one MLX
    `layers.L.mlp.switch_mlp.down_proj`, so they share a module type here and the
    expert index is returned separately.
    """
    hit = _LAYER_RE.match(name)
    if not hit:
        return None
    layer, suffix = int(hit.group(1)), hit.group(2)
    if suffix in DENSE_MODULE_TYPES:
        return layer, suffix, None
    ex = _EXPERT_RE.match(suffix)
    if ex:
        return layer, f"mlp.experts.{ex.group(2)}", int(ex.group(1))
    return None


def _peak_rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if raw > 1e8 else raw / 1e3  # bytes on macOS, KB on Linux


def scan_snapshot(snapshot_dir, layers: Optional[Sequence[int]] = None,
                  include_experts: bool = True, expert_sample: Optional[int] = None,
                  log=print) -> dict:
    """Scan a sharded HF snapshot, one tensor at a time, and return per-matrix SNR.

    `expert_sample=K` scores only experts 0..K-1 per layer instead of all 128 --
    an escape hatch for wall-clock, recorded in the output so provenance is never
    ambiguous. `expert_sample=None` scores all of them.
    """
    snapshot_dir = Path(snapshot_dir)
    index = json.loads((snapshot_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    wanted: Dict[str, List[Tuple[str, int, str, Optional[int]]]] = {}
    for name, shard in weight_map.items():
        hit = _classify(name)
        if hit is None:
            continue
        layer, mtype, expert = hit
        if layers is not None and layer not in layers:
            continue
        if expert is not None:
            if not include_experts:
                continue
            if expert_sample is not None and expert >= expert_sample:
                continue
        wanted.setdefault(shard, []).append((name, layer, mtype, expert))

    dense: Dict[str, Dict[str, dict]] = {}
    experts: Dict[str, Dict[str, List[float]]] = {}
    t0 = time.time()
    total = sum(len(v) for v in wanted.values())
    done = 0
    for shard in sorted(wanted):
        path = snapshot_dir / shard
        header, data_start = read_safetensors_header(path)
        for name, layer, mtype, expert in sorted(wanted[shard]):
            w = load_tensor_f32(path, name, header, data_start)
            r = matrix_snr(w)
            del w
            if expert is None:
                dense.setdefault(str(layer), {})[mtype] = r
            else:
                experts.setdefault(str(layer), {}).setdefault(mtype, []).append(r["snr"])
            done += 1
            if done % 200 == 0 or done == total:
                log(f"  [{done}/{total}] {shard} {name} snr={r['snr']:.4f} "
                    f"({time.time() - t0:.0f}s, peak RSS {_peak_rss_mb():.0f}MB)")

    return {
        "snapshot": str(snapshot_dir),
        "snapshot_revision": snapshot_dir.name,
        "generated": datetime.now(timezone.utc).isoformat(),
        "num_layers": len({int(k) for k in dense}) if dense else 0,
        "dense": dense,
        "experts": experts,
        "expert_sample": expert_sample,
        "include_experts": include_experts,
        "wall_clock_s": time.time() - t0,
        "peak_rss_mb": _peak_rss_mb(),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", required=True,
                    help="path to the bf16 HF snapshot dir (NOT models/qwen-q4)")
    ap.add_argument("--out", required=True, help="output JSON path (snr/snr_raw.json)")
    ap.add_argument("--no-experts", action="store_true",
                    help="skip the 18432 expert matrices entirely")
    ap.add_argument("--expert-sample", type=int, default=None,
                    help="score only the first N experts per layer (default: all 128)")
    ap.add_argument("--layers", default=None,
                    help="comma-separated layer subset, e.g. 0,1,47 (default: all)")
    args = ap.parse_args(argv)

    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = scan_snapshot(args.snapshot, layers=layers,
                           include_experts=not args.no_experts,
                           expert_sample=args.expert_sample)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}  wall_clock={result['wall_clock_s']:.0f}s "
          f"peak_rss={result['peak_rss_mb']:.0f}MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
