"""SVD of self_attn.q_proj's LoRA A-matrix (rank=16, layers 32-47, adapter-only,
not merged with base W) across the three adapters this experiment phase produced --
same low-rank SVD trick and stable-rank diagnostic as
`03-cpt-only/cpt_train/eval_v2/lora_svd_analysis.py`, narrowed to one projection
type and compared across arms instead of across CPT-v2 training legs:

  - "CPT-v2 (no SFT)"   the standalone CPT-v2 continual-pretrain adapter
  - "Base+SFT (fresh)"  the SFT adapter trained on the unmodified base
  - "CPT+SFT (cptv2)"   the SFT adapter trained on the CPT-v2-adapted base

Question: once SFT converges both arms to statistically indistinguishable
functional pass rates (RESULTS.md, +2.8pp, p~0.20), does the SFT adapter
trained on top of CPT-v2 still carry a structural trace of CPT-v2 underneath,
or does it converge to the same weight geometry as an SFT adapter trained from
scratch? See docs/reports/2026-07-cpt-vs-fresh-comparison.md Phase 4 for the
answer this produced.

For a low-rank update dW = P @ Q (P: (in,r), Q: (r,out)), dW's nonzero singular
values equal the singular values of R @ Q, where P = Qp @ R (a thin QR of P) --
Qp has orthonormal columns so left-multiplying by it doesn't change singular
values, and R @ Q is only (r, out), so this avoids ever forming or SVD-ing the
full (in, out) matrix.

The "Base (untrained init)" reference line/column is a *synthetic null*, not a
real checkpoint: mlx-lm's LoRA init sets B to exact zeros at training start, so
a literal untrained ΔW is identically 0 and has no spectrum to plot. Instead we
sample both A and B from Uniform(-1/sqrt(in_dims), 1/sqrt(in_dims)) (matching
mlx-lm's own A-init scale, applied symmetrically to B purely for this null) and
average n=50 draws -- illustrating "what a random rank-16 update would look
like next to a trained one," not the model's real initial state. The exact
null construction used in the original (lost) version of this script is
unknown; treat this reference line as informal context, not a load-bearing
part of the finding -- the finding itself rests on comparing the three real
trained adapters against each other, which this script computes directly from
their checkpoint files.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
IMAGES_DIR = RESULTS_DIR / "images"
OUT_JSON = RESULTS_DIR / "lora_svd_qproj.json"
OUT_LINE_PNG = IMAGES_DIR / "lora_svd_qproj.png"
OUT_HEATMAP_PNG = IMAGES_DIR / "lora_svd_qproj_heatmap.png"

ADAPTERS = {
    "CPT-v2 (no SFT)": REPO_ROOT / "model-experiments/03-cpt-only/adapters/cpt-v2/adapters.safetensors",
    "Base+SFT (fresh)": REPO_ROOT / "model-experiments/04-cpt-sft/sft_fresh_probe/adapters/sft-on-fresh/adapters.safetensors",
    "CPT+SFT (cptv2)": REPO_ROOT / "model-experiments/04-cpt-sft/sft_cptv2_probe/adapters/sft-on-cptv2/adapters.safetensors",
}
PROJ = "self_attn.q_proj"
LAYERS = [32, 34, 36, 38, 40, 42, 44, 47]
RANK = 16
COLORS = {"Base": "#8A8F82", "CPT-v2 (no SFT)": "#C0392B", "Base+SFT (fresh)": "#3B6FB6", "CPT+SFT (cptv2)": "#3E9B5C"}


def lowrank_singular_values(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Singular values of P @ Q, P: (in, r) with in >= r, Q: (r, out)."""
    Qp, R = np.linalg.qr(P)
    return np.linalg.svd(R @ Q, compute_uv=False)


def _selftest():
    rng = np.random.default_rng(0)
    P, Q = rng.normal(size=(64, 5)), rng.normal(size=(5, 40))
    got = lowrank_singular_values(P, Q)
    want = np.linalg.svd(P @ Q, compute_uv=False)[:5]
    assert np.allclose(got, want, atol=1e-8), f"low-rank SVD trick mismatch: {got} vs {want}"


def real_spectra() -> dict:
    """Singular values of the *effective* ΔW = scale * (A @ B) mlx-lm actually
    adds to the base weight -- adapter_config.json's lora_parameters.scale,
    not the raw A@B product, since that scale is what makes the update real."""
    out = {}
    for label, path in ADAPTERS.items():
        w = mx.load(str(path))
        scale = json.loads((path.parent / "adapter_config.json").read_text())["lora_parameters"]["scale"]
        out[label] = {}
        for layer in LAYERS:
            a = np.array(w[f"model.layers.{layer}.{PROJ}.lora_a"])  # (in, r)
            b = np.array(w[f"model.layers.{layer}.{PROJ}.lora_b"])  # (r, out)
            out[label][layer] = (scale * lowrank_singular_values(a, b)).tolist()
    return out


def synthetic_null(in_dims: int, out_dims: int, r: int = RANK, n_trials: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    scale = 1 / np.sqrt(in_dims)
    specs = np.array([
        lowrank_singular_values(
            rng.uniform(-scale, scale, size=(in_dims, r)),
            rng.uniform(-scale, scale, size=(r, out_dims)),
        )
        for _ in range(n_trials)
    ])
    return specs.mean(axis=0).tolist(), specs.std(axis=0).tolist()


def plot_lines(spectra: dict, null_mean: list):
    x = np.arange(1, RANK + 1)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey=True)
    fig.suptitle(f"LoRA A-matrix singular-value spectra (adapter-only, not merged with W), {PROJ}, layers 32-47 -- shared y-axis")
    for ax, layer in zip(axes.flat, LAYERS):
        ax.plot(x, null_mean, "-o", color=COLORS["Base"], label="Base (untrained init, n=50 mean)", markersize=4)
        for arm in ADAPTERS:
            ax.plot(x, spectra[arm][layer], "-o", color=COLORS[arm], label=arm, markersize=4)
        ax.set_yscale("log")
        ax.set_title(f"layer {layer} (q_proj)")
        ax.set_xlabel("rank")
    axes[0, 0].set_ylabel("singular value")
    axes[1, 0].set_ylabel("singular value")
    axes[0, 0].legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_LINE_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_LINE_PNG}")


def plot_heatmap(spectra: dict, null_mean: list):
    panels = ["Base", *ADAPTERS.keys()]
    grids = {"Base": np.tile(null_mean, (len(LAYERS), 1))}
    for arm in ADAPTERS:
        grids[arm] = np.array([spectra[arm][layer] for layer in LAYERS])
    vmin = min(g.min() for g in grids.values())
    vmax = max(g.max() for g in grids.values())

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    fig.suptitle(f"LoRA A-matrix singular values, {PROJ} -- rank x layer, shared colorbar scale")
    im = None
    for ax, panel in zip(axes, panels):
        title = "Base (untrained init, expected)" if panel == "Base" else panel
        im = ax.imshow(grids[panel], aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
                        extent=(0.5, RANK + 0.5, LAYERS[-1] + 1, LAYERS[0] - 1))
        ax.set_yticks(LAYERS)
        ax.set_xlabel("rank")
        ax.set_title(title)
    axes[0].set_ylabel("layer")
    fig.colorbar(im, ax=axes, label="singular value", shrink=0.85)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_HEATMAP_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_HEATMAP_PNG}")


def main():
    _selftest()
    spectra = real_spectra()
    sample_layer = LAYERS[0]
    in_dims, out_dims = mx.load(str(next(iter(ADAPTERS.values()))))[f"model.layers.{sample_layer}.{PROJ}.lora_a"].shape[0], \
        mx.load(str(next(iter(ADAPTERS.values()))))[f"model.layers.{sample_layer}.{PROJ}.lora_b"].shape[1]
    null_mean, null_std = synthetic_null(in_dims, out_dims)

    out = {
        "layers": LAYERS,
        "rank": RANK,
        "proj": PROJ,
        "init_mean": null_mean,
        "init_std": null_std,
        **spectra,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_JSON}")

    plot_lines(spectra, null_mean)
    plot_heatmap(spectra, null_mean)


if __name__ == "__main__":
    main()
