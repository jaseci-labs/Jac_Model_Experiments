#!/usr/bin/env python3
"""`mlx_lm.lora` drop-in for the **cptv2** Spectrum arm -- spectrum-plan.md §8.3.

WHY THIS IS NOT JUST THE FRESH ARM'S DRIVER WITH DIFFERENT PATHS
----------------------------------------------------------------
The fresh arm starts from nothing, so LoRA-izing exactly the 16 picks is the
whole job. The cptv2 arm starts from `03-cpt-only/adapters/cpt-v2/
adapters.safetensors` via `resume_adapter_file`, and `mlx_lm/lora.py:250` applies
that as

    model.load_weights(args.resume_adapter_file, strict=False)

**after** conversion. That CPT-v2 adapter covers exactly blocks 32-47 (verified
here at runtime, not assumed: 256 tensors, 8 module suffixes, rank 16). Both the
picks and {32..47} have exactly 16 members, so unless they are identical -- in
which case the probe is answered by construction and no training happens at all
-- the picks CANNOT be a superset. Every non-picked CPT-v2 block would be
silently dropped by `strict=False`, and the arm would train on a base that has
lost part of its CPT-v2 delta. That is not the treatment `sft-on-cptv2` (72.6%,
621/855) received, so the comparison it feeds would be invalid.

Same silent-weight-loss failure class as the `mlx_lm.fuse` bug in the 2026-07
comparison report §3.

THE RESOLUTION (spectrum-plan.md §8.3, steps 1-5)
-------------------------------------------------
1. Convert the **union** `picks | {32..47}` to LoRA.
2. Load the CPT-v2 weights -- all 256 keys now have a home. Assert the count.
3. **Freeze** the LoRA modules in `{32..47} \\ picks`: they carry the CPT-v2 delta
   as a frozen part of the base, exactly as they would have been in the fresh
   arm's base, but contribute no gradients. Picks that fall inside 32-47 stay
   trainable and are *initialized* from CPT-v2 -- precisely how the incumbent
   `sft-on-cptv2` arm's 16 layers were.
4. Assert trainable params still equal 281.838M. Capacity must be identical to
   both the fresh spectrum arm and the incumbent trailing-16 arms, or the
   four-way table in spectrum-workflow.md Phase 8 compares budgets, not placement.
5. Post-training, merge the frozen keys back (see `merge_frozen_keys.py`) --
   `mlx_lm/tuner/trainer.py` saves only `model.trainable_parameters()`, so the
   frozen CPT-v2 keys would vanish from the artifact and the delta would be lost
   at eval for the second time.

Steps 1-4 are this file. Step 5 is `merge_frozen_keys.py`. Step 5's `num_layers`
rewrite is `adapter_config_fix.py` (shared, fresh arm) driven with the UNION.

RELATIONSHIP TO THE FRESH ARM'S MODULE
--------------------------------------
`snr_scan.py`, `layer_select.py`, `spectrum_lora_layers.py` and
`adapter_config_fix.py` are NOT duplicated here. The SNR ranking is a property of
the base model's weights, which both arms share; duplicating it would create two
sources of truth for the layer picks, which is the one thing that must be
identical across arms. This module imports the shared driver and reuses its
upstream guard, its verbatim `to_lora` dispatch, and its reporting helpers.
`configs/spectrum_layers.json` is a relative symlink to the fresh arm's file for
the same reason.

USAGE
-----
    # self-test before spending compute
    python cptv2_spectrum_lora_layers.py --verify-layers --model models/qwen-q4

    # training: every other flag is stock `mlx_lm.lora`
    python cptv2_spectrum_lora_layers.py --config configs/sft_spectrum.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

# The shared Phase-1 modules live in the fresh arm's spectrum/ dir -- single
# source of truth for the picks and for the upstream-composition guard.
FRESH_SPECTRUM_DIR = (Path(__file__).resolve().parents[2]
                      / "sft_fresh_probe" / "spectrum")
if str(FRESH_SPECTRUM_DIR) not in sys.path:
    sys.path.insert(0, str(FRESH_SPECTRUM_DIR))

import mlx_lm
import mlx_lm.lora
import mlx_lm.tuner.utils as _tu

import spectrum_lora_layers as S  # noqa: E402  (path is set above)

REPO_ROOT = Path(__file__).resolve().parents[4]

CPT_V2_ADAPTER = (REPO_ROOT / "model-experiments" / "03-cpt-only" / "adapters"
                  / "cpt-v2" / "adapters.safetensors")

# spectrum-plan.md §8.3 / §3: verified facts about the CPT-v2 checkpoint. These
# are ASSERTED against the real file, never assumed.
EXPECTED_CPT_V2_KEYS = 256
EXPECTED_CPT_V2_LAYERS = list(range(32, 48))

DEFAULT_NUM_BLOCKS = S.DEFAULT_NUM_BLOCKS
EXPECTED_TRAINABLE_PARAMS = S.EXPECTED_TRAINABLE_PARAMS
EXPECTED_LORA_TENSORS = S.EXPECTED_LORA_TENSORS

_KEY_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)\.(lora_a|lora_b)$")


# --------------------------------------------------------------------------
# the CPT-v2 checkpoint: read it, do not trust the docs about it
# --------------------------------------------------------------------------
def adapter_layer_ids(adapter_file) -> List[int]:
    """Layer indices actually present in an adapter's key list."""
    keys = list(mx.load(str(adapter_file)).keys())
    ids: Set[int] = set()
    for k in keys:
        hit = _KEY_RE.match(k)
        if hit is None:
            raise ValueError(
                f"unparseable adapter key {k!r} in {adapter_file} -- this driver's "
                f"layer bookkeeping assumes model.layers.<N>.<module>.lora_{{a,b}}"
            )
        ids.add(int(hit.group(1)))
    return sorted(ids)


def assert_cpt_v2_shape(adapter_file=CPT_V2_ADAPTER) -> List[int]:
    """Fail loudly if the CPT-v2 checkpoint is not the artifact §8.3 describes."""
    path = Path(adapter_file)
    if not path.exists():
        raise FileNotFoundError(
            f"CPT-v2 checkpoint not found: {path}. The cptv2 arm's entire premise "
            f"is resuming from it (spectrum-plan.md §8.3)."
        )
    keys = list(mx.load(str(path)).keys())
    ids = adapter_layer_ids(path)
    if len(keys) != EXPECTED_CPT_V2_KEYS:
        raise RuntimeError(
            f"CPT-v2 checkpoint has {len(keys)} tensors, expected "
            f"{EXPECTED_CPT_V2_KEYS}. The base composition this driver builds "
            f"(union + freeze) is sized from that number -- re-verify §8.3 before "
            f"training."
        )
    if ids != EXPECTED_CPT_V2_LAYERS:
        raise RuntimeError(
            f"CPT-v2 checkpoint covers layers {ids}, expected "
            f"{EXPECTED_CPT_V2_LAYERS}. The union/freeze split would be wrong."
        )
    return ids


# --------------------------------------------------------------------------
# union conversion + selective freeze (§8.3 steps 1, 3, 4)
# --------------------------------------------------------------------------
def union_layers(picks: Sequence[int], base_layers: Sequence[int]) -> List[int]:
    return sorted({int(i) for i in picks} | {int(i) for i in base_layers})


def frozen_layers(picks: Sequence[int], base_layers: Sequence[int]) -> List[int]:
    """`base_layers \\ picks` -- converted so CPT-v2 lands, frozen so it stays base."""
    return sorted({int(i) for i in base_layers} - {int(i) for i in picks})


def make_union_linear_to_lora_layers(picks: Sequence[int],
                                     base_layers: Sequence[int]) -> Callable:
    """Convert `picks | base_layers`; freeze `base_layers \\ picks`.

    `num_layers` is checked against **len(picks)**, not against the union: the
    YAML's `num_layers: 16` is the TRAINABLE budget, and that is what must stay
    equal to the fresh arm's. The union is an implementation detail of getting
    the CPT-v2 delta into the model at all.
    """
    picked = sorted({int(i) for i in picks})
    union = union_layers(picked, base_layers)
    to_freeze = frozen_layers(picked, base_layers)
    inner = S.make_linear_to_lora_layers(union)

    def cptv2_linear_to_lora_layers(model: nn.Module, num_layers: int,
                                    config: Dict, use_dora: bool = False):
        if num_layers != len(picked):
            raise ValueError(
                f"num_layers={num_layers} but the spectrum selection has "
                f"{len(picked)} layers {picked}. On this arm num_layers is the "
                f"TRAINABLE budget; the union {union} is converted regardless. "
                f"Fix the config or the selection, do not proceed."
            )
        inner(model, len(union), config, use_dora)
        for i in to_freeze:
            model.layers[i].freeze(recurse=True)

    cptv2_linear_to_lora_layers.spectrum_layer_ids = picked
    cptv2_linear_to_lora_layers.union_layer_ids = union
    cptv2_linear_to_lora_layers.frozen_layer_ids = to_freeze
    return cptv2_linear_to_lora_layers


def apply_patch(picks: Sequence[int], base_layers: Sequence[int]) -> None:
    """Guard, then rebind BOTH call sites (spectrum-plan.md §6.2)."""
    if "mlx_lm_lora" in sys.modules:
        raise RuntimeError(
            "mlx_lm_lora (the DPO package) is imported. It has its own "
            "linear_to_lora_layers call path which this driver deliberately does "
            "NOT rebind -- DPO is out of scope (spectrum-plan.md §12)."
        )
    S._assert_upstream_still_matches()
    replacement = make_union_linear_to_lora_layers(picks, base_layers)
    _tu.linear_to_lora_layers = replacement
    mlx_lm.lora.linear_to_lora_layers = replacement


# --------------------------------------------------------------------------
# verification (§8.3 steps 2 and 4)
# --------------------------------------------------------------------------
def trainable_layer_ids(model) -> List[int]:
    """Blocks that actually contribute trainable parameters."""
    ids: Set[int] = set()
    for k, _ in tree_flatten(model.trainable_parameters()):
        hit = re.match(r"^layers\.(\d+)\.", k) or re.match(r"^model\.layers\.(\d+)\.", k)
        if hit:
            ids.add(int(hit.group(1)))
    return sorted(ids)


def verify_composition(model, picks: Sequence[int], base_layers: Sequence[int],
                       expect_trainable: Optional[int] = None,
                       expect_tensors: Optional[int] = None) -> tuple:
    """Every assertion §8.3 steps 1-4 asks for. Returns (ok, problems)."""
    picked = sorted({int(i) for i in picks})
    union = union_layers(picked, base_layers)
    to_freeze = frozen_layers(picked, base_layers)
    problems: List[str] = []

    rep = S.collect_lora_report(model)
    if rep["layer_ids"] != union:
        problems.append(
            f"converted layer set: got {rep['layer_ids']}, expected the union "
            f"{union} -- CPT-v2 keys outside it would be dropped by strict=False"
        )
    for L in union:
        got = rep["suffixes_by_layer"].get(L, [])
        if got != list(S.EXPECTED_SUFFIXES):
            problems.append(f"layer {L} module suffixes: got {got}, "
                            f"expected {list(S.EXPECTED_SUFFIXES)}")

    got_trainable = trainable_layer_ids(model)
    if got_trainable != picked:
        problems.append(
            f"TRAINABLE layer set: got {got_trainable}, expected the picks "
            f"{picked}. Frozen union members {to_freeze} must not train -- "
            f"capacity would exceed the fresh arm's and the comparison is invalid"
        )

    n_trainable_tensors = sum(
        1 for k, _ in tree_flatten(model.trainable_parameters())
        if k.endswith(("lora_a", "lora_b"))
    )
    want_tensors = expect_tensors if expect_tensors is not None else 16 * len(picked)
    if n_trainable_tensors != want_tensors:
        problems.append(
            f"trainable LoRA tensors: got {n_trainable_tensors}, expected "
            f"{want_tensors}"
        )

    n_params = int(sum(v.size for _, v in tree_flatten(model.trainable_parameters())))
    if expect_trainable is not None and n_params != expect_trainable:
        problems.append(
            f"trainable params: got {n_params}, expected {expect_trainable} -- "
            f"the union/freeze split changed CAPACITY, not just placement"
        )
    return (not problems), problems


def _run_verify(model_path: str, layers_path: str, lora_config: Dict,
                cpt_adapter, skip_control: bool) -> int:
    from mlx_lm.utils import load
    from adapter_config_fix import assert_adapter_keys_present

    picks = S.load_layer_ids(layers_path)
    base = assert_cpt_v2_shape(cpt_adapter)
    union = union_layers(picks, base)
    to_freeze = frozen_layers(picks, base)

    print(f"spectrum picks ({len(picks)}): {picks}")
    print(f"CPT-v2 base layers ({len(base)}): {base}   [{EXPECTED_CPT_V2_KEYS} tensors, verified]")
    print(f"union to convert ({len(union)}): {union}")
    print(f"frozen (CPT-v2 delta kept, no gradients) ({len(to_freeze)}): {to_freeze}")
    print(f"overlap picks & trailing-16: {len(set(picks) & set(base))}/{len(picks)}")

    S._assert_upstream_still_matches()
    print("upstream guard: OK "
          f"(mlx-lm {getattr(mlx_lm, '__version__', '?')}, source hash matches pin)")

    model, _ = load(model_path)
    model.freeze()                     # mirrors mlx_lm/lora.py:224
    make_union_linear_to_lora_layers(picks, base)(model, len(picks), lora_config)

    # §8.3 step 2: the CPT-v2 weights must ALL land. This is the assertion the
    # whole union exists to make possible.
    n_keys = assert_adapter_keys_present(model, cpt_adapter)
    model.load_weights(str(cpt_adapter), strict=False)
    print(f"CPT-v2 resume: all {n_keys} keys present in the converted model")

    rep = S.collect_lora_report(model)
    S._print_report("cptv2-spectrum", rep)
    n_train = int(sum(v.size for _, v in tree_flatten(model.trainable_parameters())))
    print(f"[cptv2-spectrum] TRAINABLE layers = {trainable_layer_ids(model)}")
    print(f"[cptv2-spectrum] TRAINABLE params = {n_train / 1e6:.3f}M "
          f"(frozen union members contribute 0)")

    ok, problems = verify_composition(
        model, picks, base,
        expect_trainable=EXPECTED_TRAINABLE_PARAMS,
        expect_tensors=EXPECTED_LORA_TENSORS,
    )
    del model

    if not skip_control:
        control, _ = load(model_path)
        control.freeze()
        _tu.linear_to_lora_layers(control, len(picks), lora_config)   # STOCK
        crep = S.collect_lora_report(control)
        S._print_report("control(stock trailing)", crep)
        expected_trailing = list(range(DEFAULT_NUM_BLOCKS - len(picks), DEFAULT_NUM_BLOCKS))
        if crep["layer_ids"] != expected_trailing:
            problems.append(
                f"stock control: got {crep['layer_ids']}, expected "
                f"{expected_trailing} -- the harness is not measuring placement"
            )
        ctrain = int(sum(v.size for _, v in tree_flatten(control.trainable_parameters())))
        if ctrain != n_train:
            problems.append(
                f"control trainable {ctrain} != cptv2-spectrum {n_train} -- this "
                f"arm's budget differs from the incumbent sft-on-cptv2 recipe's"
            )
        del control
        ok = not problems

    for p in problems:
        print(f"  !! {p}")
    print("VERIFY:", "PASS" if ok else f"FAIL -- {len(problems)} problem(s)")
    return 0 if ok else 1


DEFAULT_LAYERS_PATH = str(Path(__file__).resolve().parent / "configs" / "spectrum_layers.json")


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    layers_path = (S._pop_flag(argv, "--spectrum-layers")
                   or os.environ.get("SPECTRUM_LAYERS")
                   or DEFAULT_LAYERS_PATH)
    cpt_adapter = (S._pop_flag(argv, "--cpt-adapter")
                   or os.environ.get("CPT_V2_ADAPTER")
                   or str(CPT_V2_ADAPTER))

    if "--verify-layers" in argv:
        argv.remove("--verify-layers")
        ap = argparse.ArgumentParser(prog="cptv2_spectrum_lora_layers.py --verify-layers")
        ap.add_argument("--model", default="models/qwen-q4")
        ap.add_argument("--rank", type=int, default=16)
        ap.add_argument("--scale", type=float, default=2.0)
        ap.add_argument("--dropout", type=float, default=0.05)
        ap.add_argument("--skip-control", action="store_true")
        a = ap.parse_args(argv)
        return _run_verify(a.model, layers_path,
                           {"rank": a.rank, "scale": a.scale, "dropout": a.dropout},
                           cpt_adapter, a.skip_control)

    picks = S.load_layer_ids(layers_path)
    base = assert_cpt_v2_shape(cpt_adapter)
    apply_patch(picks, base)
    print(f"[cptv2_spectrum_lora_layers] picks={picks} base={base}\n"
          f"  converting union {union_layers(picks, base)}, "
          f"freezing {frozen_layers(picks, base)} (CPT-v2 delta, no gradients)\n"
          f"  mlx_lm.lora and mlx_lm.tuner.utils both patched",
          flush=True)

    from mlx_lm.lora import main as _upstream_main

    sys.argv = [sys.argv[0]] + argv   # stock argv for the stock parser
    _upstream_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
