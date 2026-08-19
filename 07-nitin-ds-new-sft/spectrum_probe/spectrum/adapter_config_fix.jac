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

import argparse;
import json;
import sys;
import from pathlib { Path }
import from typing { List, Optional, Sequence }

import mlx.core as mx;
import from mlx.utils { tree_flatten }

glob DEFAULT_NUM_BLOCKS: int = 48;


"""`num_blocks - min(picks)` -- the trailing slice that covers every pick."""
def spectrum_num_layers(picks: Sequence[int], num_blocks: int = DEFAULT_NUM_BLOCKS) -> int {
    ids = sorted({int(i) for i in picks});
    if not ids {
        raise ValueError("picks is empty -- nothing to cover");
    }
    if ids[0] < 0 or ids[-1] >= num_blocks {
        raise ValueError(f"picks out of range for {num_blocks} blocks: {ids}");
    }
    return num_blocks - ids[0];
}


"""Rewrite adapter_config.json in place so load_adapters covers the picks."""
def rewrite_adapter_config(adapter_dir: any, picks: Sequence[int],
                           num_blocks: int = DEFAULT_NUM_BLOCKS) -> dict {
    path = Path(adapter_dir) / "adapter_config.json";  # jac:ignore[E1053]
    if not path.exists() {
        raise FileNotFoundError(f"no adapter_config.json in {adapter_dir}");
    }
    cfg = json.loads(path.read_text());
    ids = sorted({int(i) for i in picks});
    if "spectrum_num_layers_original" not in cfg {
        cfg["spectrum_num_layers_original"] = cfg.get("num_layers");
    }
    cfg["num_layers"] = spectrum_num_layers(ids, num_blocks);
    cfg["spectrum_layers"] = ids;
    path.write_text(json.dumps(cfg, indent=2));
    return cfg;  # jac:ignore[E1002]
}


def adapter_keys(adapter_file: any) -> List[str] {
    return sorted(mx.load(str(adapter_file)).keys());  # jac:ignore[E1032,E1054,E1099]
}


"""Fail the eval if any saved adapter key has no home in the loaded model."""
def assert_adapter_keys_present(model: any, adapter_file: any) -> int {
    have = {k for (k, _) in tree_flatten(model.parameters())};
    keys = adapter_keys(adapter_file);
    missing = [k for k in keys if k not in have];
    if missing {
        raise RuntimeError(
            f"{len(missing)} of {len(keys)} adapter keys do not exist in the loaded "  # jac:ignore[E1053]
            f"model -- load_weights(strict=False) would DROP them silently and the "
            f"arm would be scored as a partially-trained model (spectrum-plan.md "
            f"§7). First 10 missing: {missing[:10]}"
        );
    }
    return len(keys);  # jac:ignore[E1053]
}


def main(argv: Optional[List[str]] = None) -> int {
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]);  # jac:ignore[E1099]
    ap.add_argument("--adapter-dir", required=True);
    ap.add_argument("--spectrum-layers", required=True,
                    help="configs/spectrum_layers.json");
    ap.add_argument("--num-blocks", type=int, default=DEFAULT_NUM_BLOCKS);  # jac:ignore[E1053]
    a = ap.parse_args(argv);  # jac:ignore[E1054]

    blob = json.loads(Path(a.spectrum_layers).read_text());  # jac:ignore[E1032,E1053]
    picks = blob["layers"] if isinstance(blob, dict) else blob;
    cfg = rewrite_adapter_config(a.adapter_dir, picks, a.num_blocks);  # jac:ignore[E1032,E1053]
    print(f"rewrote {a.adapter_dir}/adapter_config.json: "  # jac:ignore[E1032]
          f"num_layers {cfg['spectrum_num_layers_original']} -> {cfg['num_layers']} "
          f"(covers blocks {a.num_blocks - cfg['num_layers']}..{a.num_blocks - 1}), "  # jac:ignore[E1032,E1055]
          f"spectrum_layers={cfg['spectrum_layers']}");
    extra = cfg["num_layers"] - len(cfg["spectrum_layers"]);  # jac:ignore[E1053]
    print(f"eval-time cost: {extra} covered-but-untrained blocks "
          f"~= {extra * 17.615:.1f}M F32 params ~= {extra * 70}MB");  # jac:ignore[E1055]
    return 0;
}


with entry:__main__ {
    sys.exit(main());
}
