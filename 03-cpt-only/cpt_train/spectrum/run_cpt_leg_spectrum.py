#!/usr/bin/env python3
"""CPT leg driver on the Spectrum-picked layers -- cpt-spectrum-plan.md §2.1.

Identical to run_cpt_leg.py in every respect except which 16 decoder blocks
get LoRA. It literally imports and calls run_cpt_leg.run_leg(), after
rebinding the layer-selection function that run_cpt_leg's own top-level
`from mlx_lm.tuner.utils import ... linear_to_lora_layers` will bind.

THE IMPORT-ORDER TRAP
----------------------
`from X import Y` binds `Y` into the importing module's OWN globals at
import time. Patching `mlx_lm.tuner.utils.linear_to_lora_layers` AFTER
`run_cpt_leg` has already been imported does nothing -- run_cpt_leg.py's
own `linear_to_lora_layers` name still points at the stock function, and
training silently proceeds on the trailing-16 slice while every log line
says "spectrum". This exact class of bug has bitten this project twice
already (see sft_fresh_probe/spectrum/dpo_spectrum_train.py's three-site
rebind for the DPO package). The fix here is ordering, made explicit and
asserted rather than assumed: the patch runs, and is verified to have
taken, before `import run_cpt_leg` executes anywhere in this file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SELF = Path(__file__).resolve().parent                    # .../03-cpt-only/cpt_train/spectrum
CPT_TRAIN = SELF.parent                                    # .../03-cpt-only/cpt_train
REPO_ROOT = SELF.parents[3]                                # repo root
# SELF is the spectrum DIRECTORY (Path(__file__).resolve().parent, not .../parents[N]
# off Path(__file__) itself), so parents[0]=cpt_train, [1]=03-cpt-only,
# [2]=model-experiments, [3]=repo root. Asserted, not trusted:
assert (REPO_ROOT / "models" / "qwen-q4").exists(), f"REPO_ROOT misresolved: {REPO_ROOT}"

FRESH_SPEC = REPO_ROOT / "model-experiments" / "04-cpt-sft" / "sft_fresh_probe" / "spectrum"
for _p in (str(FRESH_SPEC), str(CPT_TRAIN)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spectrum-layers", default=str(SELF / "configs" / "spectrum_layers.json"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--done-steps", type=int, required=True)
    ap.add_argument("--resume-adapter-file", default=None)
    ap.add_argument("--resume-optimizer-file", default=None)
    return ap.parse_args()


def _block_indices(keys):
    """model.layers.<N>.<...>.lora_a -> {N, ...}"""
    return sorted({int(k.split(".")[2]) for k in keys if k.startswith("model.layers.")})


def main() -> int:
    args = _parse()

    # ---- STEP 1: patch BEFORE run_cpt_leg is imported. This assertion is the
    # whole contract -- do not move any import above this line. ----
    assert "run_cpt_leg" not in sys.modules, (
        "run_cpt_leg was imported before the layer patch was applied -- its "
        "`from mlx_lm.tuner.utils import linear_to_lora_layers` has already bound "
        "the STOCK trailing-16 function into its own globals and the rebind below "
        "cannot reach it. Do not move any import above this line."
    )
    from spectrum_lora_layers import (  # noqa: E402  -- ordering is the point
        load_layer_ids, apply_patch, EXPECTED_LORA_TENSORS,
    )

    layer_ids = load_layer_ids(args.spectrum_layers)      # sorted, deduped, range-checked
    apply_patch(layer_ids)                                 # runs _assert_upstream_still_matches(),
                                                            # then rebinds BOTH
                                                            #   mlx_lm.tuner.utils.linear_to_lora_layers
                                                            #   mlx_lm.lora.linear_to_lora_layers
    print(f"[run_cpt_leg_spectrum] LoRA layers rebound to {layer_ids} "
          f"(from {args.spectrum_layers})", flush=True)

    # ---- STEP 2: import run_cpt_leg, then PROVE the patch took. ----
    import run_cpt_leg                                     # noqa: E402
    import mlx_lm.tuner.utils as _tu                        # noqa: E402
    assert run_cpt_leg.linear_to_lora_layers is _tu.linear_to_lora_layers, \
        "run_cpt_leg bound a different linear_to_lora_layers than the patched one"
    assert getattr(run_cpt_leg.linear_to_lora_layers, "spectrum_layer_ids", None) == layer_ids, \
        "run_cpt_leg is holding the STOCK linear_to_lora_layers -- the patch did not take"

    import mlx.core as mx                                   # noqa: E402

    # ---- §2.2(a): resume-file key check (pure file read, no model load). ----
    # run_cpt_leg.py applies --resume-adapter-file with strict=False AFTER
    # conversion, which silently drops any key naming a block that was not
    # converted. Within a single spectrum run this can only fail if a
    # checkpoint from a different layer set is passed by mistake -- assert
    # it rather than train silently on a partially-seeded model.
    if args.resume_adapter_file:
        keys = list(mx.load(args.resume_adapter_file).keys())
        blocks = _block_indices(keys)
        assert blocks == layer_ids, (
            f"resume file {args.resume_adapter_file} covers blocks {blocks}, "
            f"picks are {layer_ids}"
        )
        assert len(keys) == EXPECTED_LORA_TENSORS, (
            f"resume file has {len(keys)} keys, expected {EXPECTED_LORA_TENSORS}"
        )
        print(f"[run_cpt_leg_spectrum] resume file OK: {len(keys)} keys, blocks={blocks}",
              flush=True)

    run_cpt_leg.run_leg(args.config, args.adapter_path, args.iters, args.done_steps,
                         args.resume_adapter_file, args.resume_optimizer_file)

    # ---- §2.2(b): checkpoint post-condition -- verify weights are genuinely
    # nonzero at the FIRST checkpoint, not after 26 hours. ----
    final_it = args.done_steps + args.iters
    ckpt = Path(args.adapter_path) / f"{final_it:07d}_adapters.safetensors"
    w = dict(mx.load(str(ckpt)))
    mx.eval(list(w.values()))          # materialize BEFORE inspecting (lesson 1 -- the
                                        # bug that zeroed the cptv2 arm's SFT adapter was
                                        # exactly reading lazily-loaded arrays after a write
                                        # to the same path; this read has no matching write,
                                        # but eval-before-inspect is the cheap, correct habit)
    blocks = _block_indices(w.keys())
    assert blocks == layer_ids, f"checkpoint {ckpt} covers blocks {blocks}, expected {layer_ids}"
    assert len(w) == EXPECTED_LORA_TENSORS, (
        f"checkpoint {ckpt} has {len(w)} keys, expected {EXPECTED_LORA_TENSORS}"
    )
    zeros = [k for k, v in w.items() if float(mx.max(mx.abs(v))) == 0.0]
    assert not zeros, f"{len(zeros)}/{len(w)} checkpoint tensors are all-zero: {zeros[:5]}"
    print(f"[run_cpt_leg_spectrum] checkpoint post-condition OK: {ckpt.name}, "
          f"{len(w)} keys, blocks={blocks}, no all-zero tensors", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
