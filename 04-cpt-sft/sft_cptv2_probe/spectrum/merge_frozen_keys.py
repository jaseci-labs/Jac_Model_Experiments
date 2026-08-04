#!/usr/bin/env python3
"""Post-training merge for the cptv2 Spectrum arm -- spectrum-plan.md §8.3 step 5.

THE TRAP
--------
`mlx_lm/tuner/trainer.py` saves checkpoints and the final adapter as

    mx.save_safetensors(str(adapter_file),
                        dict(tree_flatten(model.trainable_parameters())))

**trainable** parameters only. `cptv2_spectrum_lora_layers.py` deliberately
freezes the CPT-v2 blocks in `{32..47} \\ picks` so they act as base weights and
do not consume the 16-block budget -- which means those blocks are absent from
every file training writes. Load that artifact for eval and the CPT-v2 delta is
gone: exactly the silent loss §8.3 exists to prevent, arriving one stage later.

THE FIX
-------
Merge the frozen CPT-v2 keys back in. The trained keys win on any collision
(picks inside 32-47 were *initialized* from CPT-v2 and then trained further --
the trained value supersedes), and by construction there are none, since the
frozen set and the trained set are disjoint by definition.

Result: an adapter covering the union `picks | {32..47}`, which is what
`adapter_config_fix.py` must then be pointed at -- with the UNION, not the picks
-- so `load_adapters` rebuilds a superset of it.

INVARIANTS ASSERTED (this script raises, it never warns)
--------------------------------------------------------
- input holds exactly the picks, 16 keys per block
- CPT-v2 source holds exactly {32..47}, 256 keys
- output holds exactly the union, 16 keys per block, and every frozen key is
  bit-identical to the CPT-v2 source it came from
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cptv2_spectrum_lora_layers import (  # noqa: E402
    CPT_V2_ADAPTER,
    adapter_layer_ids,
    assert_cpt_v2_shape,
    frozen_layers,
    union_layers,
)

KEYS_PER_BLOCK = 16   # 8 module types x {lora_a, lora_b}


def merge(trained_file, cpt_file, out_file, picks: Sequence[int],
          base_layers: Optional[Sequence[int]] = None) -> Dict[str, object]:
    """Write `trained | (CPT-v2 keys for base_layers \\ picks)` to `out_file`."""
    picked = sorted({int(i) for i in picks})
    if base_layers is None:
        base_layers = assert_cpt_v2_shape(cpt_file)
    to_merge = frozen_layers(picked, base_layers)
    union = union_layers(picked, base_layers)

    trained = dict(mx.load(str(trained_file)))
    mx.eval(list(trained.values()))  # force materialization NOW: mx.load is a lazy
    # mmap of trained_file, and merge()'s --in/--out are the same path in normal
    # use (in-place merge). Writing out_file before these arrays are evaluated
    # truncates the very mmap they lazily read from -- every trained key comes
    # back as all-zero. This is the actual cause of the cptv2-arm corruption
    # incident (2026-08-02/03): frozen keys, loaded from a *different* file,
    # were always correct; only trained keys -- read from the file being
    # overwritten -- went to zero. Root-caused by comparing an untouched
    # numbered checkpoint (always correct) against the post-merge adapters.safetensors
    # (always zero on trained keys, byte-identical to CPT-v2 source on frozen keys).
    got_trained_layers = adapter_layer_ids(trained_file)
    if got_trained_layers != picked:
        raise RuntimeError(
            f"{trained_file} covers layers {got_trained_layers}, expected the "
            f"picks {picked}. Either the wrong file was passed or training did "
            f"not train what the selection says it did -- do not merge."
        )
    if len(trained) != KEYS_PER_BLOCK * len(picked):
        raise RuntimeError(
            f"{trained_file} has {len(trained)} tensors, expected "
            f"{KEYS_PER_BLOCK * len(picked)} ({KEYS_PER_BLOCK}/block x "
            f"{len(picked)} blocks)"
        )

    cpt = dict(mx.load(str(cpt_file)))
    merge_prefixes = tuple(f"model.layers.{i}." for i in to_merge)
    frozen = {k: v for k, v in cpt.items() if k.startswith(merge_prefixes)}
    if len(frozen) != KEYS_PER_BLOCK * len(to_merge):
        raise RuntimeError(
            f"selected {len(frozen)} frozen keys from {cpt_file}, expected "
            f"{KEYS_PER_BLOCK * len(to_merge)} for blocks {to_merge}"
        )

    clash = set(frozen) & set(trained)
    if clash:
        raise RuntimeError(
            f"{len(clash)} keys are in BOTH the trained adapter and the frozen "
            f"CPT-v2 set; they are supposed to be disjoint. First: {sorted(clash)[:5]}"
        )

    merged = {**trained, **frozen}
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out), merged)

    # read back and prove it, rather than trusting the write
    back = dict(mx.load(str(out)))
    if adapter_layer_ids(out) != union:
        raise RuntimeError(
            f"merged adapter covers {adapter_layer_ids(out)}, expected the union {union}"
        )
    if len(back) != KEYS_PER_BLOCK * len(union):
        raise RuntimeError(
            f"merged adapter has {len(back)} tensors, expected "
            f"{KEYS_PER_BLOCK * len(union)}"
        )
    for k, v in frozen.items():
        if not mx.array_equal(back[k], v):
            raise RuntimeError(
                f"merged key {k} does not match the CPT-v2 source -- the frozen "
                f"delta was altered by the merge"
            )
    return {
        "out": str(out),
        "picks": picked,
        "base_layers": sorted(int(i) for i in base_layers),
        "merged_frozen_layers": to_merge,
        "union": union,
        "n_trained_keys": len(trained),
        "n_frozen_keys": len(frozen),
        "n_total_keys": len(back),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="in_file", required=True,
                    help="adapters.safetensors written by training (trainable keys only)")
    ap.add_argument("--out", dest="out_file", required=True,
                    help="merged output (may be the same path as --in)")
    ap.add_argument("--cpt-adapter", default=str(CPT_V2_ADAPTER))
    ap.add_argument("--spectrum-layers", required=True)
    a = ap.parse_args(argv)

    blob = json.loads(Path(a.spectrum_layers).read_text())
    picks = blob["layers"] if isinstance(blob, dict) else blob
    info = merge(a.in_file, a.cpt_adapter, a.out_file, picks)
    print(f"merged {info['n_trained_keys']} trained + {info['n_frozen_keys']} frozen "
          f"CPT-v2 keys (blocks {info['merged_frozen_layers']}) "
          f"-> {info['n_total_keys']} keys covering {info['union']}")
    print(f"wrote {info['out']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
