"""SVD of the CPT-v2 LoRA adapter (rank=16, layers 32-47) at each of the 12
leg checkpoints -- diagnoses under-training vs. rank-capacity saturation by
tracking how much of the rank-16 budget each update actually uses.

For a low-rank update dW = P @ Q (P: (m,r), Q: (r,n), m>=r), dW's nonzero
singular values equal the singular values of R @ Q, where P = QR_p @ R (a
thin QR of P) -- QR_p has orthonormal columns so left-multiplying by it
doesn't change singular values, and R @ Q is only (r, n), so this avoids
ever forming or SVD-ing the full (m, n) matrix. Same trick handles both the
dense projections (q/k/v/o_proj, mlp.gate) and the per-expert MoE
projections (switch_mlp.{gate,up,down}_proj, 128 experts/layer) -- MoE
records aggregate (mean) across experts to keep output size sane; a mean
over 128 independent rank-16 spectra is still a meaningful per-layer signal.

stable_rank = (sum(s)^2) / sum(s^2) is a continuous effective-rank measure
in [1, 16]: near 16 means the update spreads across the whole rank-16
budget (spectrum ~flat -- can't rule out capacity as a limiting factor);
near 1 means one direction dominates (budget underused -- rank isn't the
bottleneck)."""
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ADAPTER_DIR = ROOT / "03-new" / "adapters" / "cpt-v2"
OUT = ROOT / "03-new" / "results" / "cpt-v2" / "json" / "lora_svd.json"

DENSE_PROJS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate"]
MOE_PROJS = ["mlp.switch_mlp.gate_proj", "mlp.switch_mlp.up_proj", "mlp.switch_mlp.down_proj"]


def lowrank_singular_values(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Singular values of P @ Q, P: (m, r) with m >= r, Q: (r, n)."""
    Qp, R = np.linalg.qr(P)
    return np.linalg.svd(R @ Q, compute_uv=False)


def stats(s: np.ndarray) -> dict:
    nuclear = float(s.sum())
    energy = float((s ** 2).sum())
    stable_rank = (nuclear ** 2) / energy if energy > 0 else 0.0
    return {"spectral_norm": float(s[0]), "nuclear_norm": nuclear, "stable_rank": stable_rank}


def leg_steps() -> list:
    ckpts = sorted(ADAPTER_DIR.glob("*_adapters.safetensors"))
    return [int(p.name.split("_")[0]) for p in ckpts]


def analyze_leg(step: int) -> list:
    w = mx.load(str(ADAPTER_DIR / f"{step:07d}_adapters.safetensors"))
    keys = list(w.keys())
    layers = sorted(set(int(k.split(".")[2]) for k in keys))
    records = []

    for layer in layers:
        for proj in DENSE_PROJS:
            a = np.array(w[f"model.layers.{layer}.{proj}.lora_a"])  # (in, r)
            b = np.array(w[f"model.layers.{layer}.{proj}.lora_b"])  # (r, out)
            s = lowrank_singular_values(a, b)
            rec = {"leg_step": step, "layer": layer, "proj": proj, "is_moe": False,
                   "singular_values": s.tolist()}
            rec.update(stats(s))
            records.append(rec)

        for proj in MOE_PROJS:
            a = np.array(w[f"model.layers.{layer}.{proj}.lora_a"])  # (experts, r, in) or (experts, r, out)... see below
            b = np.array(w[f"model.layers.{layer}.{proj}.lora_b"])  # (experts, out, r) or (experts, in, r)
            n_experts = a.shape[0]
            spectra = []
            for e in range(n_experts):
                a_e, b_e = a[e], b[e]  # a_e: (r, X), b_e: (Y, r), Y >= r
                s = lowrank_singular_values(b_e, a_e)
                spectra.append(s)
            spectra = np.array(spectra)  # (experts, r)
            mean_s = spectra.mean(axis=0)
            per_expert_stats = [stats(s) for s in spectra]
            rec = {
                "leg_step": step, "layer": layer, "proj": proj, "is_moe": True,
                "n_experts": n_experts,
                "singular_values_mean": mean_s.tolist(),
                "stable_rank_mean": float(np.mean([s["stable_rank"] for s in per_expert_stats])),
                "stable_rank_std": float(np.std([s["stable_rank"] for s in per_expert_stats])),
                "spectral_norm_mean": float(np.mean([s["spectral_norm"] for s in per_expert_stats])),
                "nuclear_norm_mean": float(np.mean([s["nuclear_norm"] for s in per_expert_stats])),
            }
            records.append(rec)
    return records


def main():
    steps = leg_steps()
    print(f"found {len(steps)} leg checkpoints: {steps}")
    all_records = []
    for i, step in enumerate(steps, 1):
        print(f"  [{i}/{len(steps)}] leg step {step} ...")
        all_records.extend(analyze_leg(step))
    out = {"rank": 16, "layers": list(range(32, 48)), "leg_steps": steps, "records": all_records}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    print(f"wrote {len(all_records)} records to {OUT}")


if __name__ == "__main__":
    main()
