# Spectrum SNR Layer-Selection SFT Probe — Scaffolding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and self-test every non-GPU-bound piece of the Spectrum layer-selection probe specified in `model-experiments/04-cpt-sft/docs/spectrum-plan.md` — a self-contained Marchenko-Pastur SNR scanner, the dense-z-mean layer-selection rule, the `linear_to_lora_layers` rebind driver, the Phase-1 config, and the gated run/eval scripts — so the only remaining work is the user's own long-running scan and training launch.

**Architecture:** Four small Python modules plus two bash runners under `model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/`. `snr_scan.py` streams one tensor at a time out of the 57GB bf16 HF snapshot via a hand-rolled safetensors header parse + `mmap` (numpy has no bf16 dtype, so the reader decodes BF16→F32 by bit-shifting) and emits per-matrix SNR; `layer_select.py` turns those per-matrix numbers into the 16 frozen layer picks; `spectrum_lora_layers.py` rebinds `linear_to_lora_layers` on **both** `mlx_lm.tuner.utils` and `mlx_lm.lora` behind a five-check upstream guard and then delegates to the stock `mlx_lm.lora.main()`; `adapter_config_fix.py` implements spectrum-plan.md §7's `num_layers := 48 - min(picks)` rewrite and the mandatory loaded-key assertion that `strict=False` would otherwise swallow. Nothing edits `.venv/`.

**Tech Stack:** Python 3.14 in `/Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv`, numpy (SVD + bf16 decode), stdlib `mmap`/`struct`/`json` (safetensors streaming — no torch, no external `safetensors` backend, both of which fail on BF16 via numpy), mlx 0.31.2 / mlx-lm 0.31.3 (`nn.Module`, `LoRALinear`, `LoRASwitchLinear`, `mlx_lm.models.qwen3_moe` for a tiny real-class test fixture), pytest 9.1.0, bash runners following `run_sft.sh`'s dry-run-first + `CONFIRM_FULL_RUN=1` discipline.

## Global Constraints

- **Do NOT run the full SNR scan against the real 57GB bf16 model in this task.** Loading a 57GB checkpoint on a 48GB-unified-memory machine is a real OOM risk, and there may be other resource-heavy work running in the background right now (a large git push). Build the SNR-scan script to work correctly (unit-test it against small synthetic weight matrices with known SNR properties — e.g. a low-rank-plus-noise matrix should score differently than pure noise), but do not invoke it against the real model weights. Build the streaming path (`safetensors` header + `mmap`, one tensor at a time, never the whole model) and unit-test it against **one** real shard file only (`~/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/*/model-00001-of-00016.safetensors`), reading it in a memory-safe streaming way to confirm the streaming approach actually works end-to-end on real data, without processing all 16 shards.
- **Do NOT launch any real training run.** Build `run_sft_spectrum.sh` with the same dry-run/`CONFIRM_FULL_RUN` gate as `run_sft.sh`, but do not set that env var or invoke a real multi-hour training job.
- **Phase 1 only.** Write only the fresh-arm config (`sft_spectrum.yaml`). Do not build Phase-2/cptv2 scaffolding (`sft_cptv2_probe/spectrum/`, union-convert, freeze, merge-back) — spectrum-plan.md §8.2 gates it on a Phase-1 result that does not exist.
- **Never edit `.venv/`.** Every behaviour change is composed from the installed package's public API in our own driver, exactly as `sft_fresh_probe/dpo_fixed_train.py` and `03-cpt-only/cpt_train/run_cpt_leg.py` do.
- **Fail loudly, never silently.** Every guard raises and exits non-zero. No warn-and-continue anywhere in this scaffolding.
- Pinned upstream facts, verified against the installed venv on 2026-08-02 and hard-coded into the guard: mlx-lm `0.31.3`, mlx `0.31.2`; `inspect.signature(linear_to_lora_layers).parameters == ['model','num_layers','config','use_dora']`; whitespace-stripped `inspect.getsource` sha256 = `96aa5d61a790a24b228127f5b5eaf2205a833e636bfa1827ab8f65fbc6ddd111`; whitespace-stripped source contains `forlinmodel.layers[-max(num_layers,0):]`; `mlx_lm.lora.linear_to_lora_layers is mlx_lm.tuner.utils.linear_to_lora_layers` → `True` (confirming `mlx_lm/lora.py:20` imports it **by name**, so §6.2's two-attribute rebind is mandatory).
- Model facts, verified from `model.safetensors.index.json` (18,867 tensors) on 2026-08-02: 48 decoder blocks; per layer exactly 5 dense LoRA-relevant matrices (`self_attn.{q,k,v,o}_proj`, `mlp.gate`) and 384 expert matrices (`mlp.experts.<0..127>.{gate,up,down}_proj`, 6,144 of each type across the model); snapshot revision `b2cff646eb4bb1d68355c01b18ae02e7cf42d120`; `model.layers.0.self_attn.q_proj.weight` is BF16, shape `[4096, 2048]`.
- Everything except the LoRA'd layer index set stays byte-identical to `sft_fresh_probe/configs/sft.yaml` (`num_layers: 16`, rank 16, scale 2.0, dropout 0.05, iters 8200, seed 42, cosine_decay warmup 820, max_seq_length 3072, mask_prompt true, grad_checkpoint true). `sft_spectrum.yaml` differs from `sft.yaml` in exactly one key: `adapter_path`.
- **Do not commit.** Leave all changes unstaged for the user to review.
- Run tests with the repo venv: `.venv/bin/python -m pytest model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/ -v` from the repo root.

---

## File Structure

| File | Responsibility |
|---|---|
| `model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/snr_scan.py` | streaming safetensors reader (BF16→F32) + Marchenko-Pastur bulk-edge SNR per matrix + whole-snapshot scan CLI |
| `.../spectrum/test_snr_scan.py` | synthetic-matrix SNR properties, synthetic BF16 file round-trip, one-real-shard streaming test |
| `.../spectrum/layer_select.py` | z-score-per-module-type → mean → top-16 aggregation, tie-break, overlap, `layer_scores.json` + `spectrum_layers.json` writers |
| `.../spectrum/test_layer_select.py` | z-score maths, ranking, tie-break, overlap, artifact shape |
| `.../spectrum/spectrum_lora_layers.py` | the `linear_to_lora_layers` replacement + 5-check upstream guard + two-attribute rebind + `--verify-layers` + delegation to `mlx_lm.lora.main()` |
| `.../spectrum/test_spectrum_lora_layers.py` | guard checks, rebind identity on both modules, index selection / tensor count / switch-count on a tiny real-class Qwen3-MoE model, stock control |
| `.../spectrum/adapter_config_fix.py` | spectrum-plan.md §7: `num_layers := 48 - min(picks)` rewrite + mandatory all-adapter-keys-present assertion |
| `.../spectrum/test_adapter_config_fix.py` | rewrite maths, provenance key, key-assertion catches a dropped layer |
| `.../spectrum/configs/sft_spectrum.yaml` | Phase-1 fresh-arm training config |
| `.../spectrum/README.md` | what has been built vs. what the user still has to run, with exact commands |
| `model-experiments/04-cpt-sft/sft_fresh_probe/run_sft_spectrum.sh` | `run_sft.sh` retargeted to the driver, same watchdog + `CONFIRM_FULL_RUN` gate, plus a `--verify-layers` preflight |
| `model-experiments/04-cpt-sft/sft_fresh_probe/eval_sft_spectrum.sh` | `eval_sft_sweep.sh` retargeted, with the §7 config rewrite + key assertion run before any scoring |

`configs/spectrum_layers.json` is deliberately **not** created by this plan — it is the scan's output (spectrum-plan.md §5.3, frozen once training starts). Every consumer must fail loudly when it is absent.

---

### Task 1: Streaming safetensors reader + Marchenko-Pastur SNR

**Files:**
- Create: `model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/snr_scan.py`
- Test: `model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/test_snr_scan.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `DENSE_MODULE_TYPES: tuple[str, ...]` = `("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate")`
  - `EXPERT_MODULE_TYPES: tuple[str, ...]` = `("mlp.experts.gate_proj", "mlp.experts.up_proj", "mlp.experts.down_proj")`
  - `read_safetensors_header(path: str | Path) -> tuple[dict, int]` → `(header, data_start_byte_offset)`
  - `load_tensor_f32(path: str | Path, name: str, header: dict | None = None, data_start: int | None = None) -> numpy.ndarray` (float32, owns its memory, mmap released before return)
  - `mp_upper_edge(m: int, n: int, sigma_sq: float) -> float`
  - `matrix_snr(w: numpy.ndarray) -> dict` → keys `snr, signal_energy, noise_energy, n_signal, n_eigenvalues, upper_edge, gamma, sigma_sq, shape`
  - `scan_snapshot(snapshot_dir, layers=None, include_experts=True, expert_sample=None, log=print) -> dict` → keys `snapshot, snapshot_revision, generated, num_layers, dense, experts, expert_sample, wall_clock_s, peak_rss_mb`; `dense[str(layer)][module_type] -> matrix_snr dict`; `experts[str(layer)][expert_module_type] -> list[float]`

- [x] **Step 1: Write the failing tests**

Create `model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/test_snr_scan.py`:

```python
"""Unit tests for snr_scan.py.

The synthetic cases pin the Marchenko-Pastur bulk-edge SNR down to properties
that are true by construction (a structureless Gaussian matrix has essentially
no above-edge energy; a low-rank-plus-noise matrix has exactly its planted rank
above the edge), so a regression in the formula shows up as a failed assertion
rather than as a plausible-looking ranking.
"""

import json
import os
import resource
import struct
from pathlib import Path

import numpy as np
import pytest

from snr_scan import (
    DENSE_MODULE_TYPES,
    load_tensor_f32,
    matrix_snr,
    mp_upper_edge,
    read_safetensors_header,
)

SNAPSHOT = Path(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/"
        "snapshots/b2cff646eb4bb1d68355c01b18ae02e7cf42d120"
    )
)
SHARD1 = SNAPSHOT / "model-00001-of-00016.safetensors"


def test_mp_upper_edge_matches_closed_form():
    # gamma = m/n = 0.25 -> edge = sigma^2 * (1 + 0.5)^2 = 2.25 * sigma^2
    assert mp_upper_edge(100, 400, 1.0) == pytest.approx(2.25)
    assert mp_upper_edge(100, 400, 3.0) == pytest.approx(6.75)
    # square matrix: gamma = 1 -> edge = 4 * sigma^2
    assert mp_upper_edge(256, 256, 1.0) == pytest.approx(4.0)


def test_pure_noise_has_almost_no_above_edge_energy():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((200, 400)).astype(np.float32)
    r = matrix_snr(w)
    assert r["n_eigenvalues"] == 200
    assert r["gamma"] == pytest.approx(0.5)
    # A structureless matrix is exactly what the bulk describes: at most a
    # handful of finite-size fluctuations poke past the edge.
    assert r["n_signal"] <= 3, r
    assert r["snr"] < 0.05, r


def test_low_rank_plus_noise_scores_far_above_pure_noise():
    rng = np.random.default_rng(1)
    noise = rng.standard_normal((200, 400))
    spike = rng.standard_normal((200, 5)) @ rng.standard_normal((5, 400)) * 3.0
    structured = matrix_snr((noise + spike).astype(np.float32))
    pure = matrix_snr(noise.astype(np.float32))
    assert structured["n_signal"] >= 5, structured
    assert structured["snr"] > 0.5, structured
    assert structured["snr"] > 10 * max(pure["snr"], 1e-6), (structured, pure)


def test_snr_increases_with_planted_rank():
    rng = np.random.default_rng(2)
    noise = rng.standard_normal((200, 400))

    def planted(rank):
        spike = rng.standard_normal((200, rank)) @ rng.standard_normal((rank, 400)) * 3.0
        return matrix_snr((noise + spike).astype(np.float32))

    r5 = planted(5)
    r20 = planted(20)
    assert r20["n_signal"] > r5["n_signal"], (r5, r20)
    assert r20["snr"] > r5["snr"], (r5, r20)


def test_snr_is_scale_invariant():
    rng = np.random.default_rng(3)
    noise = rng.standard_normal((120, 240))
    spike = rng.standard_normal((120, 4)) @ rng.standard_normal((4, 240)) * 2.0
    w = (noise + spike).astype(np.float32)
    a = matrix_snr(w)
    b = matrix_snr(w * 37.0)
    assert b["snr"] == pytest.approx(a["snr"], rel=1e-4)
    assert b["n_signal"] == a["n_signal"]


def test_zero_matrix_is_degenerate_not_a_crash():
    r = matrix_snr(np.zeros((16, 32), dtype=np.float32))
    assert r["snr"] == 0.0
    assert r["n_signal"] == 0


def test_matrix_snr_rejects_non_2d():
    with pytest.raises(ValueError):
        matrix_snr(np.zeros((4, 4, 4), dtype=np.float32))


def _write_bf16_safetensors(path: Path, arrays: dict) -> None:
    """Write a minimal BF16 safetensors file (numpy has no bf16 dtype)."""
    header, blobs, offset = {}, [], 0
    for name, arr in arrays.items():
        bits = arr.astype(np.float32).view(np.uint32)
        bf16 = (bits >> 16).astype(np.uint16)  # truncate-toward-zero bf16
        raw = bf16.tobytes()
        header[name] = {
            "dtype": "BF16",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    hdr = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hdr)))
        fh.write(hdr)
        for raw in blobs:
            fh.write(raw)


def test_bf16_roundtrip_through_the_streaming_reader(tmp_path):
    rng = np.random.default_rng(4)
    a = rng.standard_normal((8, 16)).astype(np.float32)
    b = rng.standard_normal((4, 4)).astype(np.float32)
    p = tmp_path / "tiny.safetensors"
    _write_bf16_safetensors(p, {"a": a, "b": b})

    header, data_start = read_safetensors_header(p)
    assert set(header) == {"a", "b"}
    assert header["a"]["dtype"] == "BF16"
    assert data_start == 8 + len(json.dumps(header).encode())

    got = load_tensor_f32(p, "a", header, data_start)
    assert got.shape == (8, 16)
    assert got.dtype == np.float32
    # bf16 keeps 8 mantissa bits -> ~3 decimal digits
    assert np.allclose(got, a, rtol=1e-2, atol=1e-2)
    assert load_tensor_f32(p, "b", header, data_start).shape == (4, 4)


def test_load_tensor_f32_rejects_unknown_name(tmp_path):
    p = tmp_path / "tiny.safetensors"
    _write_bf16_safetensors(p, {"a": np.zeros((2, 2), dtype=np.float32)})
    with pytest.raises(KeyError):
        load_tensor_f32(p, "nope")


@pytest.mark.skipif(not SHARD1.exists(), reason="bf16 HF snapshot not present")
def test_streams_one_real_shard_without_loading_it_whole():
    """Read exactly two matrices out of a 3.6GB shard and score one of them.

    Guards the memory claim in spectrum-plan.md §11 risk 4: the reader must
    never hold more than one matrix, so RSS growth stays in the tens of MB even
    though the file on disk is gigabytes.
    """
    shard_bytes = SHARD1.stat().st_size
    assert shard_bytes > 1_000_000_000, shard_bytes

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    header, data_start = read_safetensors_header(SHARD1)
    assert "model.layers.0.self_attn.q_proj.weight" in header
    assert header["model.layers.0.self_attn.q_proj.weight"]["dtype"] == "BF16"

    q = load_tensor_f32(SHARD1, "model.layers.0.self_attn.q_proj.weight", header, data_start)
    assert q.shape == (4096, 2048)
    assert np.isfinite(q).all()

    g = load_tensor_f32(SHARD1, "model.layers.0.mlp.gate.weight", header, data_start)
    assert g.shape == (128, 2048)

    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, kilobytes on Linux; normalise to MB.
    scale = 1e6 if after > 1e8 else 1e3
    grew_mb = (after - before) / scale
    assert grew_mb < 600, f"reader grew RSS by {grew_mb:.1f}MB — not streaming"

    r = matrix_snr(q)
    assert r["shape"] == [4096, 2048]
    assert r["gamma"] == pytest.approx(2.0)
    assert r["snr"] > 0.0
    print(f"\nreal q_proj L0 SNR={r['snr']:.4f} n_signal={r['n_signal']} "
          f"edge={r['upper_edge']:.3e} sigma2={r['sigma_sq']:.3e} "
          f"RSS+{grew_mb:.1f}MB shard={shard_bytes/1e9:.2f}GB")


def test_dense_module_types_match_the_spec():
    assert DENSE_MODULE_TYPES == (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate",
    )
```

- [x] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/test_snr_scan.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'snr_scan'`.

- [x] **Step 3: Write `snr_scan.py`**

```python
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
rule `dpo_fixed_train.py` and `run_cpt_leg.py` follow — compose your own driver,
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
of an IEEE-754 float32), and holds at most one matrix at a time — the largest
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

    Reads only the header — a few hundred KB — regardless of file size.
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

    `expert_sample=K` scores only experts 0..K-1 per layer instead of all 128 —
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
```

- [x] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/test_snr_scan.py -v -s`
Expected: all PASS, including `test_streams_one_real_shard_without_loading_it_whole`, whose `-s` output prints the real layer-0 `q_proj` SNR and the reader's RSS growth.

- [x] **Step 5: Checkpoint (do not commit — leave unstaged per Global Constraints)**

Confirm `git status --short model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/` lists both new files as untracked.

---

### Task 2: Layer selection (dense-z-mean aggregation)

**Files:**
- Create: `model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/layer_select.py`
- Test: `model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/test_layer_select.py`

**Interfaces:**
- Consumes: `snr_scan.DENSE_MODULE_TYPES`, `snr_scan.EXPERT_MODULE_TYPES`, and the dict shape returned by `snr_scan.scan_snapshot` (`scan["dense"][str(layer)][module_type]["snr"]`, `scan["experts"][str(layer)][module_type] -> list[float]`).
- Produces:
  - `TRAILING16: list[int]` = `list(range(32, 48))`
  - `zscore(values: Sequence[float]) -> list[float]`
  - `dense_layer_scores(scan: dict, module_types=DENSE_MODULE_TYPES) -> dict[int, float]`
  - `expert_layer_stats(scan: dict) -> dict[int, dict[str, float]]` (keys `mean`, `median`, `n`)
  - `rank_layers(layer_scores: dict[int, float], k: int = 16) -> list[int]` (rank order, tie-break by lower layer index)
  - `boundary_tie(layer_scores: dict[int, float], k: int = 16) -> bool`
  - `overlap_with_trailing16(picks: Sequence[int]) -> int`
  - `build_layer_scores_artifact(scan: dict) -> dict`
  - `build_selection_artifact(scan, picks_by_rank, rule, spectrum_rev) -> dict` (spectrum-plan.md §5.3 shape, `layers` ascending + `layers_by_rank`)

- [x] **Step 1: Write the failing tests**

Create `model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/test_layer_select.py`:

```python
"""Unit tests for layer_select.py — spectrum-plan.md §5's aggregation rule."""

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
    with pytest.raises(ValueError, match="self_attn.k_proj"):
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
```

- [x] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/test_layer_select.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'layer_select'`.

- [x] **Step 3: Write `layer_select.py`**

```python
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
    picks = list(picks_by_rank)
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
```

- [x] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/test_layer_select.py -v`
Expected: 11 PASS.

- [x] **Step 5: Checkpoint** — re-run both test files together: `.venv/bin/python -m pytest model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/ -q`. Leave unstaged.

---

### Task 3: The `linear_to_lora_layers` driver

**Files:**
- Create: `model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/spectrum_lora_layers.py`
- Test: `model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/test_spectrum_lora_layers.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-2 at import time; at runtime it reads the JSON that `layer_select.build_selection_artifact` produces (key `layers`, a list of ints).
- Produces:
  - `UPSTREAM_MLX_LM_VERSION: str` = `"0.31.3"`
  - `UPSTREAM_SRC_SHA256: str` = `"96aa5d61a790a24b228127f5b5eaf2205a833e636bfa1827ab8f65fbc6ddd111"`
  - `_assert_upstream_still_matches() -> None`
  - `load_layer_ids(path, num_layers: int = 48) -> list[int]` (sorted, deduped, validated `0 <= i < num_layers`)
  - `make_linear_to_lora_layers(layer_ids: Sequence[int]) -> Callable[[nn.Module, int, dict, bool], None]`
  - `apply_patch(layer_ids: Sequence[int]) -> None` (guard → rebind both `mlx_lm.tuner.utils` and `mlx_lm.lora`)
  - `collect_lora_report(model) -> dict` → keys `layer_ids, n_lora_tensors, n_switch, trainable_params, total_params, suffixes_by_layer`
  - `verify_layers(model, layer_ids, expect_tensors=None, expect_trainable=None) -> tuple[bool, list[str]]`
  - `EXPECTED_SUFFIXES: tuple[str, ...]` = the 8 module suffixes from the trained adapters

- [x] **Step 1: Write the failing tests**

Create `model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/test_spectrum_lora_layers.py`:

```python
"""Unit tests for spectrum_lora_layers.py.

The model fixture is a REAL mlx_lm Qwen3-MoE built from
`mlx_lm.models.qwen3_moe.ModelArgs`, just tiny (4 blocks, 4 experts, hidden 32).
That means the conversion under test walks the same classes the 30B model does --
`nn.Linear` for the attention projections and the router, `SwitchLinear` for the
three stacked expert projections -- so the LoRASwitchLinear branch is genuinely
exercised without loading 16GB.
"""

import importlib
import json

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

import mlx_lm.lora
import mlx_lm.tuner.utils as tuner_utils
from mlx_lm.models.qwen3_moe import Model as Qwen3MoeModel
from mlx_lm.models.qwen3_moe import ModelArgs
from mlx_lm.tuner.lora import LoRALinear, LoRASwitchLinear

import spectrum_lora_layers as S

LORA_CFG = {"rank": 16, "scale": 2.0, "dropout": 0.05}


def tiny_model(num_layers=4, num_experts=4):
    args = ModelArgs(
        model_type="qwen3_moe",
        hidden_size=32,
        num_hidden_layers=num_layers,
        intermediate_size=64,
        num_attention_heads=4,
        num_experts=num_experts,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        moe_intermediate_size=16,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        head_dim=8,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        max_position_embeddings=128,
        norm_topk_prob=True,
    )
    m = Qwen3MoeModel(args)
    mx.eval(m.parameters())
    return m


@pytest.fixture(autouse=True)
def restore_upstream():
    """Every test gets the stock function back, whatever it did to the module."""
    stock = tuner_utils.linear_to_lora_layers
    yield
    tuner_utils.linear_to_lora_layers = stock
    mlx_lm.lora.linear_to_lora_layers = stock


# --- guard -----------------------------------------------------------------
def test_guard_passes_against_the_installed_mlx_lm():
    S._assert_upstream_still_matches()  # must not raise


def test_guard_pins_the_version_and_hash_it_was_written_against():
    assert S.UPSTREAM_MLX_LM_VERSION == "0.31.3"
    assert S.UPSTREAM_SRC_SHA256 == (
        "96aa5d61a790a24b228127f5b5eaf2205a833e636bfa1827ab8f65fbc6ddd111"
    )


def test_guard_fails_when_the_target_is_gone(monkeypatch):
    monkeypatch.delattr(tuner_utils, "linear_to_lora_layers")
    with pytest.raises(RuntimeError, match="linear_to_lora_layers"):
        S._assert_upstream_still_matches()


def test_guard_fails_when_the_signature_changed(monkeypatch):
    def impostor(model, n, config):  # missing use_dora
        pass
    monkeypatch.setattr(tuner_utils, "linear_to_lora_layers", impostor)
    monkeypatch.setattr(mlx_lm.lora, "linear_to_lora_layers", impostor)
    with pytest.raises(RuntimeError, match="signature"):
        S._assert_upstream_still_matches()


def test_guard_fails_when_the_trailing_slice_is_gone(monkeypatch):
    def impostor(model, num_layers, config, use_dora=False):
        for l in model.layers[:num_layers]:  # leading, not trailing
            pass
    monkeypatch.setattr(tuner_utils, "linear_to_lora_layers", impostor)
    monkeypatch.setattr(mlx_lm.lora, "linear_to_lora_layers", impostor)
    with pytest.raises(RuntimeError, match="trailing slice"):
        S._assert_upstream_still_matches()


def test_guard_fails_when_lora_py_no_longer_shares_the_object(monkeypatch):
    def other(model, num_layers, config, use_dora=False):
        pass
    monkeypatch.setattr(mlx_lm.lora, "linear_to_lora_layers", other)
    with pytest.raises(RuntimeError, match="import structure"):
        S._assert_upstream_still_matches()


# --- layer id loading ------------------------------------------------------
def test_load_layer_ids_sorts_dedupes_and_validates(tmp_path):
    p = tmp_path / "spectrum_layers.json"
    p.write_text(json.dumps({"layers": [7, 3, 3, 47], "count": 4}))
    assert S.load_layer_ids(p) == [3, 7, 47]


def test_load_layer_ids_rejects_out_of_range(tmp_path):
    p = tmp_path / "spectrum_layers.json"
    p.write_text(json.dumps({"layers": [0, 48]}))
    with pytest.raises(ValueError, match="out of range"):
        S.load_layer_ids(p, num_layers=48)


def test_load_layer_ids_fails_loudly_when_the_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        S.load_layer_ids(tmp_path / "nope.json")


# --- the replacement conversion -------------------------------------------
def test_replacement_lora_izes_exactly_the_requested_blocks():
    m = tiny_model(num_layers=4)
    m.freeze()
    S.make_linear_to_lora_layers([0, 2])(m, 2, LORA_CFG)
    rep = S.collect_lora_report(m)
    assert rep["layer_ids"] == [0, 2]


def test_replacement_hits_the_switchlinear_branch():
    m = tiny_model(num_layers=4)
    m.freeze()
    S.make_linear_to_lora_layers([1, 3])(m, 2, LORA_CFG)
    rep = S.collect_lora_report(m)
    assert rep["n_switch"] == 6  # 3 per selected layer
    assert rep["n_lora_tensors"] == 2 * 8 * 2  # layers x modules x {lora_a, lora_b}
    for L in (1, 3):
        assert rep["suffixes_by_layer"][L] == list(S.EXPECTED_SUFFIXES)


def test_replacement_matches_stock_when_asked_for_the_trailing_slice():
    picked = tiny_model(num_layers=4)
    picked.freeze()
    S.make_linear_to_lora_layers([2, 3])(picked, 2, LORA_CFG)
    a = S.collect_lora_report(picked)

    stock = tiny_model(num_layers=4)
    stock.freeze()
    tuner_utils.linear_to_lora_layers(stock, 2, LORA_CFG)
    b = S.collect_lora_report(stock)

    assert a["layer_ids"] == b["layer_ids"] == [2, 3]
    assert a["n_lora_tensors"] == b["n_lora_tensors"]
    assert a["n_switch"] == b["n_switch"]
    assert a["trainable_params"] == b["trainable_params"]


def test_capacity_is_placement_independent():
    """The whole comparison rests on this: any 2 of 4 blocks cost the same."""
    counts = set()
    for ids in ([0, 1], [0, 3], [1, 2], [2, 3]):
        m = tiny_model(num_layers=4)
        m.freeze()
        S.make_linear_to_lora_layers(ids)(m, 2, LORA_CFG)
        counts.add(S.collect_lora_report(m)["trainable_params"])
    assert len(counts) == 1, counts


def test_num_layers_mismatch_fails_loudly():
    m = tiny_model(num_layers=4)
    m.freeze()
    with pytest.raises(ValueError, match="num_layers"):
        S.make_linear_to_lora_layers([0, 2])(m, 16, LORA_CFG)


def test_replacement_rejects_an_index_past_the_end():
    m = tiny_model(num_layers=4)
    m.freeze()
    with pytest.raises(ValueError, match="out of range"):
        S.make_linear_to_lora_layers([0, 9])(m, 2, LORA_CFG)


# --- the rebind ------------------------------------------------------------
def test_apply_patch_rebinds_BOTH_module_attributes():
    stock = tuner_utils.linear_to_lora_layers
    S.apply_patch([0, 2])
    assert tuner_utils.linear_to_lora_layers is not stock
    assert mlx_lm.lora.linear_to_lora_layers is not stock
    assert mlx_lm.lora.linear_to_lora_layers is tuner_utils.linear_to_lora_layers


def test_rebind_actually_intercepts_the_training_path():
    """mlx_lm.lora's train_model resolves the name from ITS OWN globals.

    Rebinding only mlx_lm.tuner.utils would leave this call on the stock trailing
    slice while looking patched -- exactly the failure spectrum-plan.md §6.2
    calls out. Call through mlx_lm.lora's namespace to prove the intercept.
    """
    S.apply_patch([0, 2])
    m = tiny_model(num_layers=4)
    m.freeze()
    mlx_lm.lora.linear_to_lora_layers(m, 2, LORA_CFG)
    assert S.collect_lora_report(m)["layer_ids"] == [0, 2]  # NOT [2, 3]


def test_rebind_also_intercepts_load_adapters_path():
    """mlx_lm.tuner.utils.load_adapters calls the module-global name."""
    S.apply_patch([1, 3])
    m = tiny_model(num_layers=4)
    m.freeze()
    tuner_utils.linear_to_lora_layers(m, 2, LORA_CFG)
    assert S.collect_lora_report(m)["layer_ids"] == [1, 3]


def test_stock_control_reproduces_the_trailing_slice():
    """With no rebind, the same harness must measure {n-2, n-1} -- proof the
    assertions above are measuring placement and not a constant."""
    m = tiny_model(num_layers=4)
    m.freeze()
    tuner_utils.linear_to_lora_layers(m, 2, LORA_CFG)
    assert S.collect_lora_report(m)["layer_ids"] == [2, 3]


def test_apply_patch_refuses_when_mlx_lm_lora_the_dpo_package_is_loaded(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "mlx_lm_lora", object())
    with pytest.raises(RuntimeError, match="mlx_lm_lora"):
        S.apply_patch([0, 2])


# --- verify_layers ---------------------------------------------------------
def test_verify_layers_passes_on_a_correct_conversion():
    m = tiny_model(num_layers=4)
    m.freeze()
    S.make_linear_to_lora_layers([0, 2])(m, 2, LORA_CFG)
    ok, problems = S.verify_layers(m, [0, 2], expect_tensors=32)
    assert ok, problems


def test_verify_layers_fails_on_the_wrong_layer_set():
    m = tiny_model(num_layers=4)
    m.freeze()
    S.make_linear_to_lora_layers([0, 2])(m, 2, LORA_CFG)
    ok, problems = S.verify_layers(m, [1, 3], expect_tensors=32)
    assert not ok
    assert any("layer set" in p for p in problems)


def test_verify_layers_fails_on_a_wrong_trainable_param_count():
    m = tiny_model(num_layers=4)
    m.freeze()
    S.make_linear_to_lora_layers([0, 2])(m, 2, LORA_CFG)
    ok, problems = S.verify_layers(m, [0, 2], expect_tensors=32,
                                   expect_trainable=281_838_000)
    assert not ok
    assert any("trainable" in p for p in problems)
```

- [x] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/test_spectrum_lora_layers.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'spectrum_lora_layers'`.

- [x] **Step 3: Write `spectrum_lora_layers.py`**

```python
#!/usr/bin/env python3
"""Drop-in replacement for `mlx_lm.lora` that LoRA-izes an EXPLICIT layer list.

WHAT IT CHANGES
---------------
`mlx_lm/tuner/utils.py:103` is the whole target:

    for l in model.layers[-max(num_layers, 0):]:     # trailing 16 -> blocks 32-47

That trailing slice is `mlx_lm`'s default, never a measured choice for this
project (spectrum-plan.md §1). This driver swaps it for an explicit, possibly
non-contiguous index list read from `configs/spectrum_layers.json`, and reuses
every other line of the upstream function VERBATIM -- the `to_lora` closure with
its full type dispatch (including the `SwitchLinear`/`QuantizedSwitchLinear` ->
`LoRASwitchLinear` branch, which is load-bearing here: 6 of this model's 8 LoRA
tensors per block are stacked expert projections), the `config.get("keys")` /
`get_keys_for_lora` walker, and the trailing `model.named_modules()` pass -- so a
future upstream change surfaces as a diff here rather than being silently
overridden.

`num_layers` is NOT ignored: it is asserted equal to `len(layer_ids)`, so a
config/CLI mismatch fails loudly instead of silently training a different
capacity than the YAML claims.

WHY A DRIVER AND NOT A PATCH
----------------------------
Same call `sft_fresh_probe/dpo_fixed_train.py` and
`03-cpt-only/cpt_train/run_cpt_leg.py` already made: our own script composing the
installed package's public API, never an edit to `.venv/`. It survives
`pip install --upgrade` and venv rebuilds by construction and leaves every other
recipe in this repo on the stock code path.

WHERE THE REBIND GOES -- TWO ATTRIBUTES, NOT ONE
------------------------------------------------
`mlx_lm/lora.py:20` does `from .tuner.utils import linear_to_lora_layers`, so
`train_model` (line 238) resolves the name from `mlx_lm.lora`'s OWN globals.
Rebinding only `mlx_lm.tuner.utils` would leave the training path on the stock
slice while LOOKING patched. Both are rebound:

  mlx_lm.tuner.utils.linear_to_lora_layers  -- used by load_adapters (line 131)
  mlx_lm.lora.linear_to_lora_layers         -- used by train_model  (line 238)

`mlx_lm_lora.utils.linear_to_lora_layers` (the DPO package) is deliberately NOT
rebound -- DPO is out of scope (spectrum-plan.md §12) -- so `apply_patch` asserts
that package is not even imported rather than silently half-patching it.

USAGE
-----
    # self-test before spending compute (spectrum-plan.md §6.4)
    python spectrum_lora_layers.py --verify-layers --model models/qwen-q4 \\
        --spectrum-layers configs/spectrum_layers.json

    # training: every other flag is stock `mlx_lm.lora`
    python spectrum_lora_layers.py --spectrum-layers configs/spectrum_layers.json \\
        --config configs/sft_spectrum.yaml --adapter-path adapters/sft-on-fresh-spectrum
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

import mlx_lm
import mlx_lm.lora
import mlx_lm.tuner.utils as _tu
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear
from mlx_lm.tuner.dora import DoRAEmbedding, DoRALinear
from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear

# --- pinned upstream facts (verified against the installed venv 2026-08-02) ---
UPSTREAM_MLX_LM_VERSION = "0.31.3"
UPSTREAM_SRC_SHA256 = "96aa5d61a790a24b228127f5b5eaf2205a833e636bfa1827ab8f65fbc6ddd111"
UPSTREAM_PARAMS = {"model", "num_layers", "config", "use_dora"}
TRAILING_SLICE_MARKER = "forlinmodel.layers[-max(num_layers,0):]"

# the 8 LoRA-eligible module suffixes present in every one of the 48 blocks,
# read off the trained `sft-on-fresh/adapters.safetensors` key list
EXPECTED_SUFFIXES = (
    "mlp.gate",
    "mlp.switch_mlp.down_proj",
    "mlp.switch_mlp.gate_proj",
    "mlp.switch_mlp.up_proj",
    "self_attn.k_proj",
    "self_attn.o_proj",
    "self_attn.q_proj",
    "self_attn.v_proj",
)

# spectrum-plan.md §6.4 / §3: the invariant that makes the comparison fair
EXPECTED_LORA_TENSORS = 256
EXPECTED_TRAINABLE_PARAMS = 281_838_080   # 281.838M, 0.923% of 30532.123M
DEFAULT_NUM_BLOCKS = 48

_HELP = (
    f"Installed mlx-lm: {getattr(mlx_lm, '__version__', '?')} "
    f"(driver written against {UPSTREAM_MLX_LM_VERSION}). Re-read "
    f"`.venv/lib/python3.14/site-packages/mlx_lm/tuner/utils.py:linear_to_lora_layers` "
    f"and `mlx_lm/lora.py`'s imports, update this driver's verbatim copy and its "
    f"pinned hash, and re-run the --verify-layers self-test before training."
)


def _assert_upstream_still_matches() -> None:
    """Fail at import time, never silently, if the composed-against code moved.

    Inverted twin of dpo_fixed_train.py's `_assert_upstream_still_buggy()`: that
    guard checks a bug is STILL present; this one checks the code we copied
    verbatim is STILL shaped the way we copied it (spectrum-plan.md §6.3).
    """
    fn = getattr(_tu, "linear_to_lora_layers", None)
    if fn is None:
        raise RuntimeError(
            "mlx_lm.tuner.utils.linear_to_lora_layers is gone -- this driver's "
            "composition target moved. " + _HELP
        )

    params = set(inspect.signature(fn).parameters)
    if params != UPSTREAM_PARAMS:
        raise RuntimeError(
            f"linear_to_lora_layers signature changed: upstream={sorted(params)} "
            f"expected={sorted(UPSTREAM_PARAMS)}. The call contract this driver "
            f"is a drop-in for no longer holds. " + _HELP
        )

    src = "".join(inspect.getsource(fn).split())
    if TRAILING_SLICE_MARKER not in src:
        raise RuntimeError(
            "linear_to_lora_layers no longer takes a trailing slice "
            f"({TRAILING_SLICE_MARKER!r} absent). The premise of this driver -- and "
            "of the probe's framing, which calls layers 32-47 an unexamined "
            "DEFAULT -- has changed. " + _HELP
        )

    got = hashlib.sha256(src.encode()).hexdigest()
    if got != UPSTREAM_SRC_SHA256:
        raise RuntimeError(
            f"linear_to_lora_layers source hash changed: got {got}, pinned "
            f"{UPSTREAM_SRC_SHA256}. Something in the function body was edited "
            f"(to_lora's type dispatch? get_keys_for_lora's type tuple?), and this "
            f"driver's verbatim copy is now silently stale. " + _HELP
        )

    if mlx_lm.lora.linear_to_lora_layers is not fn:
        raise RuntimeError(
            "mlx_lm.lora.linear_to_lora_layers is not the same object as "
            "mlx_lm.tuner.utils.linear_to_lora_layers -- the import structure "
            "changed, so this driver's two-attribute rebind list is no longer the "
            "complete set of call sites. " + _HELP
        )


def load_layer_ids(path, num_layers: int = DEFAULT_NUM_BLOCKS) -> List[int]:
    """Read, sort, dedupe and range-validate the frozen picks (§5.3)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"spectrum layer selection not found: {p}. It is produced by the SNR "
            f"scan (spectrum-workflow.md Phases 1-2) and frozen before training. "
            f"Run snr_scan.py then layer_select.py first."
        )
    blob = json.loads(p.read_text())
    raw = blob["layers"] if isinstance(blob, dict) else blob
    ids = sorted({int(i) for i in raw})
    bad = [i for i in ids if not (0 <= i < num_layers)]
    if bad:
        raise ValueError(f"layer index out of range for a {num_layers}-block model: {bad}")
    if not ids:
        raise ValueError(f"{p} contains no layer indices")
    return ids


def make_linear_to_lora_layers(layer_ids: Sequence[int]) -> Callable:
    """Build the drop-in replacement bound to an explicit index list.

    Body copied VERBATIM from mlx_lm 0.31.3
    `mlx_lm/tuner/utils.py:linear_to_lora_layers` (Apple Inc., MIT) except for
    the single loop that selects which blocks get converted.
    """
    ids = sorted({int(i) for i in layer_ids})

    def spectrum_linear_to_lora_layers(model: nn.Module, num_layers: int,
                                       config: Dict, use_dora: bool = False):
        if num_layers != len(ids):
            raise ValueError(
                f"num_layers={num_layers} but the spectrum selection has "
                f"{len(ids)} layers {ids}. Capacity and placement must not "
                f"disagree -- fix the config or the selection, do not proceed."
            )
        bad = [i for i in ids if i >= len(model.layers)]
        if bad:
            raise ValueError(
                f"layer index out of range: {bad} (model has {len(model.layers)} blocks)"
            )

        # --- verbatim from upstream ------------------------------------------
        def to_lora(layer):
            if not use_dora and hasattr(layer, "to_lora"):
                return layer.to_lora(
                    r=config["rank"],
                    scale=config["scale"],
                    dropout=config["dropout"],
                )

            if isinstance(layer, (nn.Linear, nn.QuantizedLinear)):
                LoRALayer = DoRALinear if use_dora else LoRALinear
            elif isinstance(layer, (SwitchLinear, QuantizedSwitchLinear)):
                if use_dora:
                    raise ValueError(f"{type(layer).__name__} doesn't support DoRA yet.")
                LoRALayer = LoRASwitchLinear
            elif isinstance(layer, (nn.Embedding, nn.QuantizedEmbedding)):
                LoRALayer = DoRAEmbedding if use_dora else LoRAEmbedding
            else:
                raise ValueError(
                    f"Can't convert layer of type {type(layer).__name__} to LoRA"
                )

            return LoRALayer.from_base(
                layer,
                r=config["rank"],
                scale=config["scale"],
                dropout=config["dropout"],
            )

        if (keys := config.get("keys", None)) is None:
            keys = set()

            def get_keys_for_lora(p, m):
                types = (
                    nn.Linear,
                    nn.QuantizedLinear,
                    SwitchLinear,
                    QuantizedSwitchLinear,
                    nn.Embedding,
                    nn.QuantizedEmbedding,
                )
                if hasattr(m, "to_lora") or isinstance(m, types):
                    keys.add(p)

            for l in model.layers:
                l.apply_to_modules(get_keys_for_lora)
        # --- end verbatim -----------------------------------------------------

        # THE ONE CHANGED STATEMENT: explicit indices, not model.layers[-n:]
        for i in ids:
            l = model.layers[i]
            lora_layers = [(k, to_lora(m)) for k, m in l.named_modules() if k in keys]
            if lora_layers:
                l.update_modules(tree_unflatten(lora_layers))

        # --- verbatim from upstream ------------------------------------------
        lora_modules = [(k, to_lora(m)) for k, m in model.named_modules() if k in keys]
        if lora_modules:
            model.update_modules(tree_unflatten(lora_modules))
        # --- end verbatim -----------------------------------------------------

    spectrum_linear_to_lora_layers.spectrum_layer_ids = ids
    return spectrum_linear_to_lora_layers


def apply_patch(layer_ids: Sequence[int]) -> None:
    """Guard, then rebind BOTH call sites (spectrum-plan.md §6.2)."""
    if "mlx_lm_lora" in sys.modules:
        raise RuntimeError(
            "mlx_lm_lora (the DPO package) is imported. It has its own "
            "linear_to_lora_layers call path (mlx_lm_lora/utils.py:197) which this "
            "driver deliberately does NOT rebind -- DPO is out of scope "
            "(spectrum-plan.md §12). Running it here would half-patch the "
            "conversion. Use the stock DPO driver instead."
        )
    _assert_upstream_still_matches()
    replacement = make_linear_to_lora_layers(layer_ids)
    _tu.linear_to_lora_layers = replacement
    mlx_lm.lora.linear_to_lora_layers = replacement


# --------------------------------------------------------------------------
# self-test (spectrum-plan.md §6.4)
# --------------------------------------------------------------------------
def collect_lora_report(model) -> dict:
    """Measure what actually got converted, from the model object itself."""
    layer_ids, suffixes, n_switch = set(), {}, 0
    for i, layer in enumerate(model.layers):
        found = []
        for name, mod in layer.named_modules():
            if isinstance(mod, (LoRALinear, LoRASwitchLinear, DoRALinear,
                                DoRAEmbedding, LoRAEmbedding)):
                found.append(name)
                if isinstance(mod, LoRASwitchLinear):
                    n_switch += 1
        if found:
            layer_ids.add(i)
            suffixes[i] = sorted(found)

    trainable = tree_flatten(model.trainable_parameters())
    n_tensors = sum(1 for k, _ in trainable if k.endswith(("lora_a", "lora_b")))
    return {
        "layer_ids": sorted(layer_ids),
        "suffixes_by_layer": suffixes,
        "n_switch": n_switch,
        "n_lora_tensors": n_tensors,
        "trainable_params": int(sum(v.size for _, v in trainable)),
        "total_params": int(sum(v.size for _, v in tree_flatten(model.parameters()))),
    }


def verify_layers(model, layer_ids: Sequence[int],
                  expect_tensors: Optional[int] = EXPECTED_LORA_TENSORS,
                  expect_trainable: Optional[int] = None,
                  expect_switch_per_layer: int = 3) -> tuple:
    """Return (ok, problems). Every assertion from spectrum-plan.md §6.4."""
    want = sorted({int(i) for i in layer_ids})
    rep = collect_lora_report(model)
    problems: List[str] = []

    if rep["layer_ids"] != want:
        problems.append(f"layer set: got {rep['layer_ids']}, expected {want}")
    if expect_tensors is not None and rep["n_lora_tensors"] != expect_tensors:
        problems.append(
            f"LoRA tensor count: got {rep['n_lora_tensors']}, expected {expect_tensors}"
        )
    if expect_trainable is not None and rep["trainable_params"] != expect_trainable:
        problems.append(
            f"trainable params: got {rep['trainable_params']}, expected "
            f"{expect_trainable} -- the selection changed CAPACITY, not just "
            f"placement, and the comparison is invalid"
        )
    want_switch = expect_switch_per_layer * len(want)
    if rep["n_switch"] != want_switch:
        problems.append(
            f"LoRASwitchLinear count: got {rep['n_switch']}, expected {want_switch} "
            f"({expect_switch_per_layer}/layer) -- the MoE branch did not fire"
        )
    for L in want:
        got = rep["suffixes_by_layer"].get(L, [])
        if got != list(EXPECTED_SUFFIXES):
            problems.append(f"layer {L} module suffixes: got {got}, "
                            f"expected {list(EXPECTED_SUFFIXES)}")
    return (not problems), problems


def _print_report(tag: str, rep: dict) -> None:
    total_m = rep["total_params"] / 1e6
    train_m = rep["trainable_params"] / 1e6
    print(f"[{tag}] layers          = {rep['layer_ids']}")
    print(f"[{tag}] lora tensors    = {rep['n_lora_tensors']}")
    print(f"[{tag}] LoRASwitchLinear= {rep['n_switch']}")
    print(f"[{tag}] trainable       = {train_m:.3f}M / {total_m:.3f}M "
          f"({train_m * 100 / total_m:.3f}%)")


def _run_verify(model_path: str, layers_path: str, lora_config: Dict,
                skip_control: bool) -> int:
    from mlx_lm.utils import load

    ids = load_layer_ids(layers_path)
    print(f"spectrum picks ({len(ids)}): {ids}")
    print(f"overlap with trailing-16 {{32..47}}: "
          f"{len(set(ids) & set(range(32, 48)))}/{len(ids)}")

    _assert_upstream_still_matches()
    print("upstream guard: OK "
          f"(mlx-lm {getattr(mlx_lm, '__version__', '?')}, source hash matches pin)")

    model, _ = load(model_path)
    model.freeze()                      # mirrors mlx_lm/lora.py:224
    make_linear_to_lora_layers(ids)(model, len(ids), lora_config)
    rep = collect_lora_report(model)
    _print_report("spectrum", rep)
    ok, problems = verify_layers(model, ids,
                                 expect_tensors=EXPECTED_LORA_TENSORS,
                                 expect_trainable=EXPECTED_TRAINABLE_PARAMS)
    del model

    if not skip_control:
        control, _ = load(model_path)
        control.freeze()
        _tu.linear_to_lora_layers(control, len(ids), lora_config)   # STOCK
        crep = collect_lora_report(control)
        _print_report("control", crep)
        expected_trailing = list(range(48 - len(ids), 48))
        if crep["layer_ids"] != expected_trailing:
            problems.append(
                f"stock control: got {crep['layer_ids']}, expected "
                f"{expected_trailing} -- the harness is not measuring placement"
            )
        if crep["trainable_params"] != rep["trainable_params"]:
            problems.append(
                f"control trainable {crep['trainable_params']} != spectrum "
                f"{rep['trainable_params']} -- capacity is not placement-invariant"
            )
        del control
        ok = not problems

    for p in problems:
        print(f"  !! {p}")
    print("VERIFY:", "PASS" if ok else f"FAIL -- {len(problems)} problem(s)")
    return 0 if ok else 1


def _pop_flag(argv: List[str], flag: str) -> Optional[str]:
    """Remove `--flag VALUE` from argv in place and return VALUE."""
    if flag in argv:
        i = argv.index(flag)
        value = argv[i + 1]
        del argv[i:i + 2]
        return value
    for i, tok in enumerate(list(argv)):
        if tok.startswith(flag + "="):
            del argv[i]
            return tok.split("=", 1)[1]
    return None


DEFAULT_LAYERS_PATH = str(Path(__file__).resolve().parent / "configs" / "spectrum_layers.json")


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    layers_path = (_pop_flag(argv, "--spectrum-layers")
                   or os.environ.get("SPECTRUM_LAYERS")
                   or DEFAULT_LAYERS_PATH)

    if "--verify-layers" in argv:
        argv.remove("--verify-layers")
        ap = argparse.ArgumentParser(prog="spectrum_lora_layers.py --verify-layers")
        ap.add_argument("--model", default="models/qwen-q4")
        ap.add_argument("--rank", type=int, default=16)
        ap.add_argument("--scale", type=float, default=2.0)
        ap.add_argument("--dropout", type=float, default=0.05)
        ap.add_argument("--skip-control", action="store_true")
        a = ap.parse_args(argv)
        return _run_verify(a.model, layers_path,
                           {"rank": a.rank, "scale": a.scale, "dropout": a.dropout},
                           a.skip_control)

    ids = load_layer_ids(layers_path)
    apply_patch(ids)
    print(f"[spectrum_lora_layers] LoRA layers rebound to {ids} "
          f"(from {layers_path}); mlx_lm.lora and mlx_lm.tuner.utils both patched",
          flush=True)

    from mlx_lm.lora import main as _upstream_main

    sys.argv = [sys.argv[0]] + argv   # stock argv for the stock parser
    _upstream_main()                  # parses sys.argv exactly as `mlx_lm.lora` does
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [x] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/test_spectrum_lora_layers.py -v`
Expected: all PASS. The two decisive ones are `test_rebind_actually_intercepts_the_training_path` (call through `mlx_lm.lora`'s own namespace lands on `[0, 2]`, not the stock `[2, 3]`) and `test_stock_control_reproduces_the_trailing_slice`.

- [x] **Step 5: Confirm the guard's pinned hash is real, not copied from the plan**

Run:
```bash
.venv/bin/python -c "
import hashlib, inspect, mlx_lm.tuner.utils as U
print(hashlib.sha256(''.join(inspect.getsource(U.linear_to_lora_layers).split()).encode()).hexdigest())"
```
Expected: `96aa5d61a790a24b228127f5b5eaf2205a833e636bfa1827ab8f65fbc6ddd111`, identical to `UPSTREAM_SRC_SHA256`.

- [x] **Step 6: Checkpoint** — full suite green, changes left unstaged.

---

### Task 4: Eval-time adapter_config rewrite + loaded-key assertion

**Files:**
- Create: `model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/adapter_config_fix.py`
- Test: `model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/test_adapter_config_fix.py`

**Interfaces:**
- Consumes: `spectrum_lora_layers.load_layer_ids`, `spectrum_lora_layers.make_linear_to_lora_layers`.
- Produces:
  - `spectrum_num_layers(picks: Sequence[int], num_blocks: int = 48) -> int` → `num_blocks - min(picks)`
  - `rewrite_adapter_config(adapter_dir, picks, num_blocks: int = 48) -> dict`
  - `adapter_keys(adapter_file) -> list[str]`
  - `assert_adapter_keys_present(model, adapter_file) -> int` (raises `RuntimeError` naming the missing keys; returns the matched count)

- [x] **Step 1: Write the failing tests**

Create `model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/test_adapter_config_fix.py`:

```python
"""Unit tests for adapter_config_fix.py -- spectrum-plan.md §7's silent-drop trap."""

import json

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from adapter_config_fix import (
    adapter_keys,
    assert_adapter_keys_present,
    rewrite_adapter_config,
    spectrum_num_layers,
)
from spectrum_lora_layers import make_linear_to_lora_layers
from test_spectrum_lora_layers import LORA_CFG, tiny_model


def test_num_layers_covers_a_superset_of_the_picks():
    # min pick 7 -> 48 - 7 = 41 -> stock trailing slice covers blocks 7..47
    assert spectrum_num_layers([7, 20, 47]) == 41
    assert spectrum_num_layers(list(range(32, 48))) == 16   # matches the incumbent
    assert spectrum_num_layers([0, 47]) == 48               # worst case: all blocks


def test_num_layers_rejects_empty_picks():
    with pytest.raises(ValueError):
        spectrum_num_layers([])


def test_rewrite_adapter_config_sets_num_layers_and_provenance(tmp_path):
    cfg = tmp_path / "adapter_config.json"
    cfg.write_text(json.dumps({
        "fine_tune_type": "lora", "num_layers": 16,
        "lora_parameters": {"rank": 16, "scale": 2.0, "dropout": 0.05},
        "model": "models/qwen-q4",
    }))
    out = rewrite_adapter_config(tmp_path, [7, 11, 47])

    assert out["num_layers"] == 41
    assert out["spectrum_layers"] == [7, 11, 47]
    assert out["spectrum_num_layers_original"] == 16
    # everything load_adapters actually reads must survive untouched
    assert out["fine_tune_type"] == "lora"
    assert out["lora_parameters"] == {"rank": 16, "scale": 2.0, "dropout": 0.05}
    assert json.loads(cfg.read_text())["num_layers"] == 41


def test_rewrite_is_idempotent(tmp_path):
    cfg = tmp_path / "adapter_config.json"
    cfg.write_text(json.dumps({
        "fine_tune_type": "lora", "num_layers": 16,
        "lora_parameters": {"rank": 16, "scale": 2.0, "dropout": 0.05},
    }))
    rewrite_adapter_config(tmp_path, [7, 47])
    again = rewrite_adapter_config(tmp_path, [7, 47])
    assert again["num_layers"] == 41
    assert again["spectrum_num_layers_original"] == 16   # not overwritten with 41


def _save_adapter(model, path):
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(path), weights)
    return sorted(weights)


def test_assert_adapter_keys_present_passes_when_every_key_lands(tmp_path):
    trained = tiny_model(num_layers=4)
    trained.freeze()
    make_linear_to_lora_layers([0, 2])(trained, 2, LORA_CFG)
    f = tmp_path / "adapters.safetensors"
    saved = _save_adapter(trained, f)
    assert len(saved) == 32
    assert adapter_keys(f) == saved

    fresh = tiny_model(num_layers=4)
    fresh.freeze()
    make_linear_to_lora_layers([0, 2])(fresh, 2, LORA_CFG)
    assert assert_adapter_keys_present(fresh, f) == 32


def test_assert_adapter_keys_present_catches_the_silent_drop(tmp_path):
    """The exact failure spectrum-plan.md §7 describes: a spectrum adapter
    naming layer 0 loaded into a model whose LoRA was rebuilt on layers 2-3.
    `load_weights(strict=False)` would swallow it; this must not."""
    trained = tiny_model(num_layers=4)
    trained.freeze()
    make_linear_to_lora_layers([0, 2])(trained, 2, LORA_CFG)
    f = tmp_path / "adapters.safetensors"
    _save_adapter(trained, f)

    wrong = tiny_model(num_layers=4)
    wrong.freeze()
    make_linear_to_lora_layers([2, 3])(wrong, 2, LORA_CFG)   # the stock slice
    with pytest.raises(RuntimeError, match="layers.0"):
        assert_adapter_keys_present(wrong, f)


def test_superset_conversion_makes_every_key_land(tmp_path):
    """Proves §7's primary fix works: converting blocks min(picks)..n-1 covers
    the picks, so nothing is dropped."""
    trained = tiny_model(num_layers=4)
    trained.freeze()
    make_linear_to_lora_layers([1, 3])(trained, 2, LORA_CFG)
    f = tmp_path / "adapters.safetensors"
    _save_adapter(trained, f)

    n_layers = spectrum_num_layers([1, 3], num_blocks=4)   # 4 - 1 = 3 -> blocks 1,2,3
    superset = list(range(4 - n_layers, 4))
    assert superset == [1, 2, 3]
    covered = tiny_model(num_layers=4)
    covered.freeze()
    make_linear_to_lora_layers(superset)(covered, len(superset), LORA_CFG)
    assert assert_adapter_keys_present(covered, f) == 32


def test_untrained_covered_layer_has_a_zero_lora_b(tmp_path):
    """§7's zero-delta claim: a covered-but-untrained layer contributes nothing,
    because lora_b initialises to zeros in both LoRALinear and LoRASwitchLinear."""
    m = tiny_model(num_layers=4)
    m.freeze()
    make_linear_to_lora_layers([1, 2, 3])(m, 3, LORA_CFG)
    zeros = [k for k, v in tree_flatten(m.trainable_parameters())
             if k.endswith("lora_b")]
    assert zeros
    for k, v in tree_flatten(m.trainable_parameters()):
        if k.endswith("lora_b"):
            assert float(mx.abs(v).max().item()) == 0.0, k
```

- [x] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/test_adapter_config_fix.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'adapter_config_fix'`.

- [x] **Step 3: Write `adapter_config_fix.py`**

```python
#!/usr/bin/env python3
"""Eval-time layer reconstruction for a spectrum adapter -- spectrum-plan.md §7.

THE TRAP
--------
`eval_functional.jac` loads via `mlx_lm.load(mp, adapter_path=adapter)` ->
`mlx_lm/tuner/utils.py:113` `load_adapters` -> `linear_to_lora_layers(model,
config.num_layers, ...)`, reading `num_layers` straight out of the adapter's own
`adapter_config.json`. With `num_layers: 16` written by training, the eval
process rebuilds LoRA on blocks 32-47 REGARDLESS of where the weights actually
came from, and then line 137 calls `model.load_weights(..., strict=False)`.
`strict=False` silently ignores keys that match nothing, so a spectrum adapter
naming block 7 would have block 7 dropped without a warning and the arm scored as
a partially-trained model.

Same failure class as the `mlx_lm.fuse` bug in the 2026-07 comparison report §3:
a silent weight loss producing a plausible-looking, completely wrong number.

THE FIX (exact, not an approximation)
-------------------------------------
Rewrite `num_layers := 48 - min(picks)` so the stock trailing slice covers a
SUPERSET of the picks. Every covered block with no saved weights stays at LoRA
init, where the delta is IDENTICALLY ZERO -- `mlx_lm/tuner/lora.py` initialises
`lora_b` to zeros for both `LoRALinear` and `LoRASwitchLinear`, and both
`__call__`s compute `y + scale * (x @ a) @ b`. Dropout is inert (`mlx_lm/utils.py`
calls `model.eval()` at load). Extra keys in `adapter_config.json` are harmless:
`load_adapters` builds a `types.SimpleNamespace` and reads only
`fine_tune_type`, `num_layers`, `lora_parameters`.

Cost: 281.838M/16 = 17.615M F32 params ~= 70MB per covered-but-untrained block;
worst case (min(picks) == 0) is 845M params ~= 3.4GB on top of the 16GB q4 base.

AND A MANDATORY ASSERTION REGARDLESS
------------------------------------
`assert_adapter_keys_present` checks every key in `adapters.safetensors` matches
a parameter that exists in the loaded model. `strict=False` will not tell you.
This raises; it never warns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import mlx.core as mx
from mlx.utils import tree_flatten

DEFAULT_NUM_BLOCKS = 48


def spectrum_num_layers(picks: Sequence[int], num_blocks: int = DEFAULT_NUM_BLOCKS) -> int:
    """`num_blocks - min(picks)` -- the trailing slice that covers every pick."""
    ids = sorted({int(i) for i in picks})
    if not ids:
        raise ValueError("picks is empty -- nothing to cover")
    if ids[0] < 0 or ids[-1] >= num_blocks:
        raise ValueError(f"picks out of range for {num_blocks} blocks: {ids}")
    return num_blocks - ids[0]


def rewrite_adapter_config(adapter_dir, picks: Sequence[int],
                           num_blocks: int = DEFAULT_NUM_BLOCKS) -> dict:
    """Rewrite adapter_config.json in place so load_adapters covers the picks."""
    path = Path(adapter_dir) / "adapter_config.json"
    if not path.exists():
        raise FileNotFoundError(f"no adapter_config.json in {adapter_dir}")
    cfg = json.loads(path.read_text())
    ids = sorted({int(i) for i in picks})
    if "spectrum_num_layers_original" not in cfg:
        cfg["spectrum_num_layers_original"] = cfg.get("num_layers")
    cfg["num_layers"] = spectrum_num_layers(ids, num_blocks)
    cfg["spectrum_layers"] = ids
    path.write_text(json.dumps(cfg, indent=2))
    return cfg


def adapter_keys(adapter_file) -> List[str]:
    return sorted(mx.load(str(adapter_file)).keys())


def assert_adapter_keys_present(model, adapter_file) -> int:
    """Fail the eval if any saved adapter key has no home in the loaded model."""
    have = {k for k, _ in tree_flatten(model.parameters())}
    keys = adapter_keys(adapter_file)
    missing = [k for k in keys if k not in have]
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(keys)} adapter keys do not exist in the loaded "
            f"model -- load_weights(strict=False) would DROP them silently and the "
            f"arm would be scored as a partially-trained model (spectrum-plan.md "
            f"§7). First 10 missing: {missing[:10]}"
        )
    return len(keys)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--spectrum-layers", required=True,
                    help="configs/spectrum_layers.json")
    ap.add_argument("--num-blocks", type=int, default=DEFAULT_NUM_BLOCKS)
    a = ap.parse_args(argv)

    blob = json.loads(Path(a.spectrum_layers).read_text())
    picks = blob["layers"] if isinstance(blob, dict) else blob
    cfg = rewrite_adapter_config(a.adapter_dir, picks, a.num_blocks)
    print(f"rewrote {a.adapter_dir}/adapter_config.json: "
          f"num_layers {cfg['spectrum_num_layers_original']} -> {cfg['num_layers']} "
          f"(covers blocks {a.num_blocks - cfg['num_layers']}..{a.num_blocks - 1}), "
          f"spectrum_layers={cfg['spectrum_layers']}")
    extra = cfg["num_layers"] - len(cfg["spectrum_layers"])
    print(f"eval-time cost: {extra} covered-but-untrained blocks "
          f"~= {extra * 17.615:.1f}M F32 params ~= {extra * 70}MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [x] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/test_adapter_config_fix.py -v`
Expected: 8 PASS, including `test_assert_adapter_keys_present_catches_the_silent_drop`.

- [x] **Step 5: Checkpoint** — run the whole spectrum suite: `.venv/bin/python -m pytest model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/ -q`.

---

### Task 5: Phase-1 config, gated run script, eval script, README

**Files:**
- Create: `model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/configs/sft_spectrum.yaml`
- Create: `model-experiments/04-cpt-sft/sft_fresh_probe/run_sft_spectrum.sh`
- Create: `model-experiments/04-cpt-sft/sft_fresh_probe/eval_sft_spectrum.sh`
- Create: `model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/README.md`

**Interfaces:**
- Consumes: `spectrum_lora_layers.py`'s CLI (`--spectrum-layers PATH`, `--verify-layers`), `adapter_config_fix.py`'s CLI (`--adapter-dir`, `--spectrum-layers`), `layer_select.py`'s CLI, `snr_scan.py`'s CLI.
- Produces: no Python interfaces. Runtime contract: `run_sft_spectrum.sh` refuses to start real training unless `CONFIRM_FULL_RUN=1`; `eval_sft_spectrum.sh` rewrites `adapter_config.json` before any scoring.

- [x] **Step 1: Write `configs/sft_spectrum.yaml`**

Byte-identical to `sft_fresh_probe/configs/sft.yaml` except `adapter_path`:

```yaml
# sft_fresh_probe/configs/sft.yaml VERBATIM except `adapter_path`.
# num_layers stays 16 -- the COUNT is unchanged; only membership changes, and
# spectrum_lora_layers.py cross-checks this value against the length of
# configs/spectrum_layers.json (spectrum-plan.md §6.1).
model: "models/qwen-q4"
train: true
data: "model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft"
fine_tune_type: lora
num_layers: 16
lora_parameters:
  rank: 16
  scale: 2.0
  dropout: 0.05
batch_size: 1
iters: 8200
learning_rate: 2.0e-5
lr_schedule:
  name: cosine_decay
  warmup: 820
  arguments: [2.0e-5, 8200, 1.0e-6]
max_seq_length: 3072
adapter_path: "model-experiments/04-cpt-sft/sft_fresh_probe/adapters/sft-on-fresh-spectrum"
save_every: 820
steps_per_eval: 500
steps_per_report: 50
val_batches: 8
seed: 42
mask_prompt: true
grad_checkpoint: true
```

- [x] **Step 2: Verify the config differs from the incumbent in exactly one key**

Run:
```bash
.venv/bin/python -c "
import yaml
a=yaml.safe_load(open('model-experiments/04-cpt-sft/sft_fresh_probe/configs/sft.yaml'))
b=yaml.safe_load(open('model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/configs/sft_spectrum.yaml'))
diff={k for k in set(a)|set(b) if a.get(k)!=b.get(k)}
print('differing keys:', diff)
assert diff=={'adapter_path'}, diff
print('OK: only adapter_path differs')"
```
Expected: `differing keys: {'adapter_path'}` then `OK: only adapter_path differs`.

- [x] **Step 3: Write `run_sft_spectrum.sh`**

`run_sft.sh` with `CFG`/`ADAPTER`/`RDIR` retargeted, `mlx_lm.lora` swapped for the driver, and a `--verify-layers` preflight added ahead of the existing gate. The watchdog loop, the stall detector, the OOM ladder, the true-global-step checkpoint archiving and the `CONFIRM_FULL_RUN=1` gate are unchanged.

```bash
#!/usr/bin/env bash
# Spectrum-layer SFT probe runner (Phase 1, fresh arm) -- run_sft.sh with CFG /
# ADAPTER / RDIR retargeted and `mlx_lm.lora` swapped for spectrum_lora_layers.py.
#
# The driver is a DROP-IN for `mlx_lm.lora`: it rebinds
# linear_to_lora_layers on both mlx_lm.lora and mlx_lm.tuner.utils, then
# delegates to mlx_lm.lora.main(), which parses sys.argv exactly as the stock
# CLI does. So every flag this loop passes (--config / --iters / --adapter-path
# / --resume-adapter-file) behaves stock, and the watchdog/stall/OOM/checkpoint
# machinery below is byte-for-byte the same logic as run_sft.sh's.
#
# One addition ahead of the gate: a --verify-layers preflight. spectrum-plan.md
# §6.4 makes the trainable-parameter count the probe's cheapest correctness
# check -- a count that differs from 281.838M means the selection changed
# CAPACITY, not just placement, and invalidates the whole comparison. Training
# is gated on it, not merely advised by it.
set -euo pipefail

if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu env CAFFEINATED=1 "$0" "$@"
fi

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(cd "$SELF_DIR/../../.." && pwd)"   # repo root
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

SPEC_DIR="model-experiments/04-cpt-sft/sft_fresh_probe/spectrum"
DRIVER="$SPEC_DIR/spectrum_lora_layers.py"
LAYERS="$SPEC_DIR/configs/spectrum_layers.json"
CFG="$SPEC_DIR/configs/sft_spectrum.yaml"
ADAPTER="model-experiments/04-cpt-sft/sft_fresh_probe/adapters/sft-on-fresh-spectrum"
CKPT_DIR="$ADAPTER/checkpoints"
RDIR="model-experiments/04-cpt-sft/sft_fresh_probe/results/sft-spectrum"
TRAIN_LOG="$RDIR/train.log"
DRY_ITERS="${DRY_ITERS:-30}"
EVAL_EVERY="${EVAL_EVERY:-60}"
STALL_SECS="${SFT_STALL_SECS:-900}"
OOM_RECOVERY_ITERS="${SFT_OOM_RECOVERY_ITERS:-100}"

need() { command -v "$1" >/dev/null 2>&1 || { echo "MISSING: $1"; exit 1; }; }
need jac; need python
for f in model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/train.jsonl \
         model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl \
         "$CFG" "$DRIVER"; do
  [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }
done
if [ ! -f "$LAYERS" ]; then
  echo "MISSING: $LAYERS"
  echo "  The layer selection is produced by the SNR scan, not by this script."
  echo "  Run spectrum-workflow.md Phases 1-2 first:"
  echo "    python $SPEC_DIR/snr_scan.py --snapshot <bf16 snapshot> --out $SPEC_DIR/snr/snr_raw.json"
  echo "    python $SPEC_DIR/layer_select.py --snr $SPEC_DIR/snr/snr_raw.json \\"
  echo "      --layer-scores-out $SPEC_DIR/snr/layer_scores.json --selection-out $LAYERS"
  exit 1
fi

# Preflight: a competing resident model OOMs this 48GB machine regardless of
# anything this script does (documented dual-model-load gotcha from the CPT work).
if pgrep -f "jac start" >/dev/null 2>&1 || pgrep -f "mlx_lm" >/dev/null 2>&1; then
  echo "!!! another jac/mlx_lm process is already running -- stop it first."
  pgrep -fl "jac start|mlx_lm" || true
  exit 1
fi

mkdir -p "$RDIR" "$ADAPTER" "$CKPT_DIR"
done_mark() { touch "$RDIR/.$1.done"; }
is_done() { [ -f "$RDIR/.$1.done" ]; }
ADAPTER_FILE="$ADAPTER/adapters.safetensors"
PROGRESS_FILE="$RDIR/.sft_progress_steps"

# --- self-test gate (spectrum-plan.md §6.4) -------------------------------
if ! is_done verify && [ "${SKIP_VERIFY:-0}" != "1" ]; then
  echo ">>> --verify-layers self-test (loads models/qwen-q4 twice; a few minutes)"
  python "$DRIVER" --verify-layers --spectrum-layers "$LAYERS" \
    --model models/qwen-q4 2>&1 | tee "$RDIR/verify_layers.txt"
  if ! grep -q "^VERIFY: PASS" "$RDIR/verify_layers.txt"; then
    echo "!!! self-test FAILED -- see $RDIR/verify_layers.txt. Not training."
    exit 1
  fi
  done_mark verify
fi

# --- dry-run (advisory, does not gate) -------------------------------------
if [ ! -f "$ADAPTER_FILE" ] && ! is_done dry && [ "${SKIP_DRY:-0}" != "1" ]; then
  echo ">>> dry-run (${DRY_ITERS} iters) -- bail check"
  python "$DRIVER" --spectrum-layers "$LAYERS" --config "$CFG" --iters "$DRY_ITERS" \
    --adapter-path "model-experiments/04-cpt-sft/sft_fresh_probe/adapters/dry-spectrum" 2>&1 | tail -25
  echo ">>> dry-run complete."
  done_mark dry
fi

# --- the actual safety gate. UNCONDITIONAL on ADAPTER_FILE existence and NOT
# additionally gated on "did a dry-run happen" -- run_sft.sh's comment records
# why: with SKIP_DRY=1 on a fresh start an AND-gate never fires and training
# launches with ZERO confirmation required.
if [ ! -f "$ADAPTER_FILE" ] && [ "${CONFIRM_FULL_RUN:-}" != "1" ]; then
  echo "Self-test and dry-run complete (or skipped). Re-run with CONFIRM_FULL_RUN=1 to start the real multi-hour training run."
  exit 0
fi

TOTAL_ITERS="$(grep -E '^iters:' "$CFG" | grep -oE '[0-9]+' | head -1)"
TOTAL_ITERS="${TOTAL_ITERS:-8200}"
[ -f "$PROGRESS_FILE" ] || echo 0 > "$PROGRESS_FILE"

if is_done train && [ -f "$ADAPTER_FILE" ]; then
  echo ">>> training: already complete"
  exit 0
fi

consecutive_fails=0
oom_shrinks=0
attempt_iters_override=""
while true; do
  DONE_STEPS="$(cat "$PROGRESS_FILE")"
  REMAIN=$(( TOTAL_ITERS - DONE_STEPS ))
  if [ "$REMAIN" -le 0 ] && [ -f "$ADAPTER_FILE" ]; then
    done_mark train
    echo "=== spectrum SFT training done ($DONE_STEPS/$TOTAL_ITERS). Next: eval_sft_spectrum.sh ==="
    break
  fi
  ATTEMPT_ITERS="$REMAIN"
  if [ -n "$attempt_iters_override" ] && [ "$attempt_iters_override" -lt "$REMAIN" ]; then
    ATTEMPT_ITERS="$attempt_iters_override"
  fi
  echo ">>> SFT attempt: requesting ${ATTEMPT_ITERS} iters (${DONE_STEPS}/${TOTAL_ITERS} done, ${REMAIN} remaining)" | tee -a "$TRAIN_LOG"

  BEFORE_CKPTS="$(ls "$ADAPTER"/*_adapters.safetensors 2>/dev/null | xargs -n1 basename 2>/dev/null || true)"
  : > "$RDIR/.segment.log"
  # Two explicit branches, NOT an empty-array expansion -- macOS bash 3.2 treats
  # "${ARR[@]}" as an unbound-variable error under `set -u` when ARR is empty.
  if [ "$DONE_STEPS" -gt 0 ] && [ -f "$ADAPTER_FILE" ]; then
    python "$DRIVER" --spectrum-layers "$LAYERS" --config "$CFG" --adapter-path "$ADAPTER" \
      --iters "$ATTEMPT_ITERS" --resume-adapter-file "$ADAPTER_FILE" >> "$RDIR/.segment.log" 2>&1 &
  else
    python "$DRIVER" --spectrum-layers "$LAYERS" --config "$CFG" --adapter-path "$ADAPTER" \
      --iters "$ATTEMPT_ITERS" >> "$RDIR/.segment.log" 2>&1 &
  fi
  SEG_PID=$!

  last_growth=$(date +%s); last_size=0
  while kill -0 "$SEG_PID" 2>/dev/null; do
    sleep "$EVAL_EVERY"
    cur_size="$(wc -l < "$RDIR/.segment.log" 2>/dev/null || echo 0)"; now="$(date +%s)"
    if [ "$cur_size" -gt "$last_size" ]; then last_size="$cur_size"; last_growth="$now"; fi
    if [ $(( now - last_growth )) -ge "$STALL_SECS" ]; then
      echo "!!! stalled: no log growth for ${STALL_SECS}s -- killing PID $SEG_PID and treating as a failed attempt" | tee -a "$TRAIN_LOG"
      kill -9 "$SEG_PID" 2>/dev/null || true
      break
    fi
    JAC_TRAIN_LOG="$RDIR/.segment.log" JAC_METRICS="/dev/null" JAC_PLOT_DIR="$RDIR" \
      jac run model-experiments/01-sft-dpo/sft_dpo/jacgen/plot_metrics.jac >/dev/null 2>&1 || true
  done
  RC=0; wait "$SEG_PID" 2>/dev/null || RC=$?
  cat "$RDIR/.segment.log" >> "$TRAIN_LOG"

  AFTER_CKPTS="$(ls "$ADAPTER"/*_adapters.safetensors 2>/dev/null | xargs -n1 basename 2>/dev/null || true)"
  NEW_CKPTS="$(comm -13 <(echo "$BEFORE_CKPTS" | sort) <(echo "$AFTER_CKPTS" | sort) 2>/dev/null || true)"
  MAX_NEW_LOCAL=0
  for f in $NEW_CKPTS; do
    ln="$(echo "$f" | grep -oE '^[0-9]+' | sed 's/^0*//')"; ln="${ln:-0}"
    true_step=$(( DONE_STEPS + ln ))
    cp "$ADAPTER/$f" "$CKPT_DIR/$(printf '%07d' "$true_step")_adapters.safetensors"
    [ "$ln" -gt "$MAX_NEW_LOCAL" ] && MAX_NEW_LOCAL="$ln"
  done

  if [ "$RC" -ne 0 ]; then
    consecutive_fails=$(( consecutive_fails + 1 ))
    echo "!!! attempt failed (attempt ${consecutive_fails})" | tee -a "$TRAIN_LOG"
    if [ "$MAX_NEW_LOCAL" -gt 0 ]; then
      NEW_DONE=$(( DONE_STEPS + MAX_NEW_LOCAL ))
      echo "$NEW_DONE" > "$PROGRESS_FILE"
      echo "  real progress persisted before the crash: now at ${NEW_DONE}/${TOTAL_ITERS}" | tee -a "$TRAIN_LOG"
    fi
    if grep -qEi "out of memory|OutOfMemory|kIOGPUCommandBuffer|MTL::.*(OOM|Insufficient)" "$RDIR/.segment.log"; then
      echo "!!! OOM signature detected in segment log" | tee -a "$TRAIN_LOG"
      if [ "$oom_shrinks" -lt 2 ]; then
        oom_shrinks=$(( oom_shrinks + 1 ))
        attempt_iters_override="$OOM_RECOVERY_ITERS"
        echo "!!! OOM-recovery ${oom_shrinks}/2: next attempt(s) capped at ${OOM_RECOVERY_ITERS} iters" | tee -a "$TRAIN_LOG"
      else
        echo "!!! OOM persisted through the shrink ladder -- giving up." | tee -a "$TRAIN_LOG"
        exit 1
      fi
    fi
    if [ "$consecutive_fails" -ge 5 ]; then
      echo "!!! attempt failed 5x in a row at the same ${DONE_STEPS}/${TOTAL_ITERS} point -- giving up." | tee -a "$TRAIN_LOG"
      tail -20 "$TRAIN_LOG"; exit 1
    fi
    continue
  fi

  consecutive_fails=0
  attempt_iters_override=""
  NEW_DONE=$(( DONE_STEPS + ATTEMPT_ITERS ))
  echo "$NEW_DONE" > "$PROGRESS_FILE"
done
```

- [x] **Step 4: Write `eval_sft_spectrum.sh`**

```bash
#!/usr/bin/env bash
# Sequential per-checkpoint functional eval for the spectrum arm. Reuses
# sft_cptv2_probe/jacgen/eval_functional.jac unmodified (env-var driven), exactly
# as eval_sft_sweep.sh does, so this arm is scored by the same harness that
# produced every other number in RESULTS.md.
#
# THE ONE ADDITION: adapter_config.json is rewritten BEFORE any scoring
# (spectrum-plan.md §7). Without it, load_adapters rebuilds LoRA on blocks 32-47
# from the adapter's own `num_layers: 16` and load_weights(strict=False)
# SILENTLY DROPS every out-of-slice layer -- the same failure class as the
# mlx_lm.fuse bug in comparison report §3.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
[ -d ".venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"

SPEC_DIR="model-experiments/04-cpt-sft/sft_fresh_probe/spectrum"
LAYERS="$SPEC_DIR/configs/spectrum_layers.json"
ADAPTER="model-experiments/04-cpt-sft/sft_fresh_probe/adapters/sft-on-fresh-spectrum"
RDIR="model-experiments/04-cpt-sft/sft_fresh_probe/results/sft-spectrum"
HOLDOUT="model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl"
METRICS="$RDIR/metrics_functional.jsonl"
SUBSET="${SUBSET:-100}"

[ -f "$LAYERS" ] || { echo "MISSING: $LAYERS (run the SNR scan + layer_select.py first)"; exit 1; }
[ -f "$ADAPTER/adapter_config.json" ] || { echo "MISSING: $ADAPTER/adapter_config.json (train first)"; exit 1; }

mkdir -p "$RDIR/images"
: > "$METRICS"

echo ">>> rewriting adapter_config.json so load_adapters covers the spectrum picks (§7)"
python "$SPEC_DIR/adapter_config_fix.py" --adapter-dir "$ADAPTER" --spectrum-layers "$LAYERS" \
  | tee "$RDIR/adapter_config_rewrite.txt"

echo ">>> asserting every adapter key lands in the loaded model (strict=False will not tell you)"
python - "$ADAPTER" <<'PY' | tee "$RDIR/key_assertion.txt"
import sys
from mlx_lm.utils import load
from adapter_config_fix import assert_adapter_keys_present
adapter = sys.argv[1]
model, _ = load("models/qwen-q4", adapter_path=adapter)
n = assert_adapter_keys_present(model, adapter + "/adapters.safetensors")
print(f"OK: all {n} adapter keys present in the loaded model")
PY

echo ">>> base (plain Qwen, no CPT, no SFT) -- FULL holdout"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL=models/qwen-q4 JAC_EVAL_ADAPTER="" \
  JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP=0 \
  jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac | tee "$RDIR/base.txt"

TMPADP="model-experiments/04-cpt-sft/sft_fresh_probe/adapters/sft-spectrum-ckpt-eval"
CKPT_DIR="$ADAPTER/checkpoints"
for CK in "$CKPT_DIR"/*_adapters.safetensors; do
  [ -e "$CK" ] || continue
  # run_sft_spectrum.sh's watchdog already archives each checkpoint under its
  # TRUE global step name, so the filename IS the real step -- no offset needed.
  STEP="$(basename "$CK" | grep -oE '^[0-9]+' | sed 's/^0*//')"; STEP="${STEP:-0}"
  rm -rf "$TMPADP"; mkdir -p "$TMPADP"
  cp "$CK" "$TMPADP/adapters.safetensors"
  cp "$ADAPTER/adapter_config.json" "$TMPADP/adapter_config.json"   # already rewritten
  echo ">>> checkpoint $STEP (subset=$SUBSET)"
  JAC_EVAL_MODE=mlx JAC_EVAL_MODEL=models/qwen-q4 JAC_EVAL_ADAPTER="$TMPADP" \
    JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_LIMIT="$SUBSET" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$STEP" \
    jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac 2>/dev/null | tail -5
done
rm -rf "$TMPADP"

echo ">>> final spectrum SFT checkpoint -- FULL holdout"
TOTAL_ITERS="$(grep -E '^iters:' "$SPEC_DIR/configs/sft_spectrum.yaml" | grep -oE '[0-9]+' | head -1)"
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL=models/qwen-q4 JAC_EVAL_ADAPTER="$ADAPTER" \
  JAC_HOLDOUT="$HOLDOUT" JAC_EVAL_METRICS_OUT="$METRICS" JAC_EVAL_STEP="$TOTAL_ITERS" \
  jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac | tee "$RDIR/final.txt"

echo "=== functional eval sweep done: $METRICS ==="
echo "Next: paired McNemar vs sft-on-fresh (597/855) -> $RDIR/mcnemar.json, then the §8.2 gate."
```

- [x] **Step 5: Make both scripts executable and syntax-check them without running them**

Run:
```bash
chmod +x model-experiments/04-cpt-sft/sft_fresh_probe/run_sft_spectrum.sh \
         model-experiments/04-cpt-sft/sft_fresh_probe/eval_sft_spectrum.sh
bash -n model-experiments/04-cpt-sft/sft_fresh_probe/run_sft_spectrum.sh
bash -n model-experiments/04-cpt-sft/sft_fresh_probe/eval_sft_spectrum.sh
echo "bash syntax OK"
```
Expected: `bash syntax OK`, no output from `bash -n`.

- [x] **Step 6: Prove the run script refuses to train (gate + missing-selection guard)**

Run (with **no** `CONFIRM_FULL_RUN`, and with `configs/spectrum_layers.json` deliberately absent because the scan has not run):
```bash
model-experiments/04-cpt-sft/sft_fresh_probe/run_sft_spectrum.sh; echo "exit=$?"
```
Expected: `MISSING: .../configs/spectrum_layers.json` plus the two-command instruction block, `exit=1`. No model is loaded and no training starts.

- [x] **Step 7: Prove the `CONFIRM_FULL_RUN` gate itself fires**

Temporarily stage a throwaway selection file so the script gets past the missing-selection guard, and short-circuit the self-test and dry-run so nothing heavy loads:
```bash
mkdir -p /tmp/spectrum-gate-check
printf '{"layers": [3,7,11,15,19,23,27,31,33,35,38,40,43,45,46,47], "count": 16}' \
  > /tmp/spectrum-gate-check/spectrum_layers.json
cp /tmp/spectrum-gate-check/spectrum_layers.json \
   model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/configs/spectrum_layers.json
SKIP_VERIFY=1 SKIP_DRY=1 CAFFEINATED=1 \
  model-experiments/04-cpt-sft/sft_fresh_probe/run_sft_spectrum.sh; echo "exit=$?"
rm -f model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/configs/spectrum_layers.json
```
Expected: `Self-test and dry-run complete (or skipped). Re-run with CONFIRM_FULL_RUN=1 to start the real multi-hour training run.` and `exit=0`, with **no** `mlx_lm.lora` process ever launched. The throwaway file is deleted again on the last line — `spectrum_layers.json` must not exist in the tree, because it is the scan's output and inventing one would be exactly the "plausible-looking, completely wrong number" failure this probe is guarding against.

- [x] **Step 8: Write `spectrum/README.md`** — what is built, what is still the user's to run:

```markdown
# spectrum/ — Phase-1 scaffolding for the Spectrum layer-selection SFT probe

Specs: `../../docs/spectrum-plan.md` (design), `../../docs/spectrum-workflow.md`
(runbook). Plan: `docs/superpowers/plans/2026-08-02-spectrum-layer-sft-probe.md`.

## Built and self-tested

| File | What it does |
|---|---|
| `snr_scan.py` | Marchenko-Pastur bulk-edge SNR per weight matrix, streaming one tensor at a time out of the bf16 snapshot (hand-rolled safetensors header + mmap, BF16→F32 by bit-shift — no numpy bf16 dtype exists and both `safe_open` backends raise on this file). |
| `layer_select.py` | spectrum-plan.md §5.1's primary rule: z-score each dense module type across its 48 layers, mean the five, rank, take 16. Expert scores recorded, not used. `--rule q_proj-only` is the §5.1 fallback. |
| `spectrum_lora_layers.py` | `mlx_lm.lora` drop-in that LoRA-izes an explicit layer list. Five-check upstream guard, two-attribute rebind, `--verify-layers` self-test. |
| `adapter_config_fix.py` | spectrum-plan.md §7: `num_layers := 48 - min(picks)` rewrite + the mandatory all-keys-present assertion `strict=False` would otherwise swallow. |
| `configs/sft_spectrum.yaml` | `../configs/sft.yaml` with `adapter_path` retargeted. Nothing else differs. |
| `../run_sft_spectrum.sh` | `run_sft.sh` retargeted; adds a `--verify-layers` gate ahead of the existing `CONFIRM_FULL_RUN=1` gate. |
| `../eval_sft_spectrum.sh` | `eval_sft_sweep.sh` retargeted; runs the §7 rewrite and the key assertion before any scoring. |

Tests: `.venv/bin/python -m pytest model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/ -v`

## NOT run — these are yours, they need the GPU / hours

`configs/spectrum_layers.json` does not exist yet. It is the scan's output and is
frozen once training starts; nothing in this directory invents one.

1. **SNR scan** (spectrum-workflow.md Phase 1). 57GB snapshot, 18,867 tensors;
   wall-clock and peak RSS are unmeasured and belong in the write-up.
   ```bash
   SNAP=~/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/b2cff646eb4bb1d68355c01b18ae02e7cf42d120
   .venv/bin/python model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/snr_scan.py \
     --snapshot "$SNAP" \
     --out model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/snr/snr_raw.json \
     2>&1 | tee model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/snr/scan.log
   ```
   `--expert-sample 8` scores 8 of 128 experts per layer instead of all of them
   if the full expert pass is too slow; the choice is recorded in the output JSON.
   `--no-experts` skips them entirely (the ranking never uses them).

2. **Layer selection** (Phase 2), then write `snr/SELECTION.md` by hand with the
   rule used, the MoE finding, ties, and the overlap:
   ```bash
   .venv/bin/python model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/layer_select.py \
     --snr model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/snr/snr_raw.json \
     --layer-scores-out model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/snr/layer_scores.json \
     --selection-out model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/configs/spectrum_layers.json
   ```
   If it prints `picks == {32..47}`, stop: the probe is answered by construction.

3. **Self-test** (Phase 3) — loads `models/qwen-q4` twice, a few minutes:
   ```bash
   .venv/bin/python model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/spectrum_lora_layers.py \
     --verify-layers --model models/qwen-q4 \
     --spectrum-layers model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/configs/spectrum_layers.json
   ```
   Must print 281.838M / 0.923%, 256 tensors, 48 LoRASwitchLinear, and a control
   run reproducing `{32..47}`. A different trainable count means the selection
   changed capacity, not placement — stop and fix before spending compute.

4. **Training** (Phase 4, ~2-2.6h, one continuous process):
   ```bash
   CONFIRM_FULL_RUN=1 model-experiments/04-cpt-sft/sft_fresh_probe/run_sft_spectrum.sh
   ```

5. **Eval** (Phase 5):
   ```bash
   model-experiments/04-cpt-sft/sft_fresh_probe/eval_sft_spectrum.sh
   ```

6. **Gate** (Phase 6): paired McNemar vs `sft-on-fresh` (597/855, 69.8%),
   p < 0.05 AND |Δ| ≥ 2.8pp, decision written to `results/sft-spectrum/GATE.md`
   before any Phase-2 work. On a null: stop.

## Standing caveat

Spectrum's published wins are for full-parameter unfreezing of high-SNR modules,
not LoRA on selected layers, and its ranking is designed for dense decoders —
18,432 of this model's 18,867 tensors are sparsely-activated expert projections
whose SNR has no established interpretation (spectrum-plan.md §4.3, §11 risks 1
and 6). Both belong in the write-up regardless of the result.
```

- [x] **Step 9: Final full-suite run and status check**

Run:
```bash
.venv/bin/python -m pytest model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/ -v
git status --short model-experiments/04-cpt-sft/ docs/superpowers/plans/
```
Expected: every test PASS; every new file listed as untracked (`??`), nothing staged.

---

## Self-Review

**1. Spec coverage** — every in-scope requirement of `spectrum-plan.md` / `spectrum-workflow.md` mapped to a task:

| Spec requirement | Task |
|---|---|
| §4 / §4.1 SNR formula (MP bulk edge, above-edge over in-bulk energy), computed on bf16 not q4 | Task 1 (`matrix_snr`, `mp_upper_edge`; the scan CLI takes a `--snapshot` path and the README's command points at the bf16 snapshot) |
| §11 risk 4 / workflow Phase 1: stream per-tensor, never materialise 57GB | Task 1 (`read_safetensors_header` + `load_tensor_f32`, real-shard RSS test) |
| §4.3 HF→MLX expert-name mapping | Task 1 (`_classify` collapses `mlp.experts.<E>.<proj>` onto the stacked MLX name) |
| §5.1 primary rule: dense-only, z-scored per module type, mean-aggregated, top 16 | Task 2 (`dense_layer_scores`, `rank_layers`) |
| §5.1 expert scores recorded but not used | Task 2 (`expert_layer_stats`, `build_layer_scores_artifact`) |
| §5.1 q_proj-only fallback | Task 2 (`--rule q_proj-only`, tested) |
| §5.2 tie-break by lower layer index; early exit when picks == {32..47} | Task 2 (`rank_layers` sort key, `boundary_tie`, the `main()` early-exit message) |
| §5.3 selection artifact shape + `overlap_with_trailing16` | Task 2 (`build_selection_artifact`) |
| §6.1 verbatim reuse except the one loop; `num_layers == len(ids)` assertion | Task 3 (`make_linear_to_lora_layers`) |
| §6.2 rebind BOTH `mlx_lm.lora` and `mlx_lm.tuner.utils`; assert `mlx_lm_lora` unused | Task 3 (`apply_patch` + two intercept tests) |
| §6.3 all five upstream guard checks incl. pinned source sha256 | Task 3 (`_assert_upstream_still_matches`, one test per check) |
| §6.4 self-test: layer set / 256 tensors / 281.838M / 8 suffixes / 48 switch / stock control | Task 3 (`verify_layers`, `_run_verify`) |
| §7 `num_layers := 48 - min(picks)` + mandatory key assertion + zero-delta claim | Task 4 |
| §10 / workflow Phase 3: `sft_spectrum.yaml` differing only in `adapter_path` | Task 5 (Steps 1-2, with a diff assertion) |
| Workflow Phase 3-4: retargeted `run_sft_spectrum.sh` with the dry-run + `CONFIRM_FULL_RUN` gate | Task 5 (Steps 3, 6, 7) |
| Workflow Phase 5: retargeted `eval_sft_spectrum.sh` running the §7 rewrite first | Task 5 (Step 4) |
| §12 out-of-scope: no scan run, no training run, no Phase 2, no DPO | Global Constraints + Task 5 Step 7's explicit non-launch check |

Deliberate non-goals, each already out of scope in the spec and re-stated in Global Constraints: the actual SNR scan run (§12 bullet 1), the actual training/eval runs (§12 bullet 1), Phase-2 cptv2 scaffolding (§12 bullet 2), the DPO stage (§12 bullet 4). `snr/SELECTION.md` is written by hand at scan time from Phase 1's findings (spectrum-workflow.md Phase 1's checklist) — it records a judgement about output this scan has not produced yet, so pre-writing it would be fabrication; the README's step 2 names it as the user's.

**2. Placeholder scan** — no "TBD", "handle edge cases", "similar to Task N", or bare prose steps. Every code step carries complete, runnable code; every run step names the exact command and the exact expected output. The one "TBD" that survives is in quoted spec prose (§4.2's scan-time filenames), where it belongs, and it is resolved by this plan implementing the scan itself rather than shelling out to an unpinned script.

**3. Type consistency** — checked across tasks: `snr_scan.DENSE_MODULE_TYPES` (a `tuple`) is imported by `layer_select` and by both test files with the same spelling; `matrix_snr`'s return keys (`snr`, `n_signal`, `upper_edge`, `sigma_sq`, `gamma`, `shape`) are the exact keys `layer_select.dense_layer_scores` reads (`["snr"]`) and the exact keys the Task-1 tests assert; `scan_snapshot`'s `dense[str(layer)][module_type]` string-keyed shape is what `_scan()` in `test_layer_select.py` fabricates and what `dense_layer_scores` indexes; `build_selection_artifact`'s `layers` key is what `spectrum_lora_layers.load_layer_ids` and `adapter_config_fix.main` both read; `make_linear_to_lora_layers(ids)` returns a callable with upstream's exact `(model, num_layers, config, use_dora=False)` signature, which is what `apply_patch` rebinds and what `test_adapter_config_fix.py` calls positionally as `(m, 2, LORA_CFG)`; `collect_lora_report`'s keys (`layer_ids`, `n_lora_tensors`, `n_switch`, `trainable_params`, `total_params`, `suffixes_by_layer`) are consistent between `verify_layers`, `_print_report` and every test that reads them.
