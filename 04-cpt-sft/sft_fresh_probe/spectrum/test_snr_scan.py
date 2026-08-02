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
    """Read exactly two matrices out of a multi-GB shard and score one of them.

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
