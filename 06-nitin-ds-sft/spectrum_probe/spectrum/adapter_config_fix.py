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
