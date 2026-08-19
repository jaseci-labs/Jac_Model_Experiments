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

import argparse;
import hashlib;
import inspect;
import json;
import os;
import sys;
import from pathlib { Path }
import from typing { Callable, Dict, List, Optional, Sequence }

import mlx.nn as nn;
import from mlx.utils { tree_flatten, tree_unflatten }

import mlx_lm;
import mlx_lm.lora;
import mlx_lm.tuner.utils as _tu;
# [TODO: jac port blocked -- jaclang 0.16.1's type checker crashes with
# "'NoneType' object has no attribute 'is_instantiable_class'" on a Jac named
# import from mlx_lm.models.switch_layers / mlx_lm.tuner.dora / mlx_lm.tuner.lora
# (`import from mlx_lm.tuner.lora { LoRALinear }` alone reproduces it, and it
# aborts the whole check, not just these lines). Plain Python imports are exempt.
# Retry `import from ...` once jaclang fixes it -- semantics are identical.]
::py::
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear
from mlx_lm.tuner.dora import DoRAEmbedding, DoRALinear
from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear
::py::

# --- pinned upstream facts (verified against the installed venv 2026-08-02) ---
glob UPSTREAM_MLX_LM_VERSION: str = "0.31.3";
glob UPSTREAM_SRC_SHA256: str = "96aa5d61a790a24b228127f5b5eaf2205a833e636bfa1827ab8f65fbc6ddd111";
glob UPSTREAM_PARAMS: set = {"model", "num_layers", "config", "use_dora"};
glob TRAILING_SLICE_MARKER: str = "forlinmodel.layers[-max(num_layers,0):]";

# the 8 LoRA-eligible module suffixes present in every one of the 48 blocks,
# read off the trained `sft-on-fresh/adapters.safetensors` key list
glob EXPECTED_SUFFIXES: tuple = (
    "mlp.gate",
    "mlp.switch_mlp.down_proj",
    "mlp.switch_mlp.gate_proj",
    "mlp.switch_mlp.up_proj",
    "self_attn.k_proj",
    "self_attn.o_proj",
    "self_attn.q_proj",
    "self_attn.v_proj",
);

# spectrum-plan.md §6.4 / §3: the invariant that makes the comparison fair
glob EXPECTED_LORA_TENSORS: int = 256;
# EXACT count, re-derived against the real 30B model 2026-08-02 by --verify-layers
# and cross-checked against a real mlx_lm run's own banner
# (`02-rl-grpo/results/chain1.log:56`: "Trainable parameters: 0.923%
# (281.838M/30532.123M)" -- mlx_lm prints millions to 3dp, so 281.838M is
# 281_837_568 rounded, not a 5-significant-figure count).
#
#   per block, rank 16:
#     q_proj  2048*16 + 16*4096 =  98304      o_proj 4096*16 + 16*2048 =  98304
#     k_proj  2048*16 + 16* 512 =  40960      v_proj 2048*16 + 16* 512 =  40960
#     mlp.gate 2048*16 + 16*128 =  34816
#     switch_mlp {gate,up}_proj  128*(2048*16 + 16*768) = 5767168  each
#     switch_mlp  down_proj      128*( 768*16 + 16*2048) = 5767168
#   = 17_614_848 per block  x 16 blocks = 281_837_568
#
# NOTE the percentage: mlx_lm's banner divides by the DEQUANTIZED 30532.123M
# (0.923%). `--verify-layers` divides by tree_flatten(model.parameters()) on the
# q4 model, which counts packed uint32 words, so it prints ~5.576% of 5054.233M.
# Same numerator, different denominator -- do not "fix" one to match the other.
glob EXPECTED_TRAINABLE_PARAMS: int = 281_837_568;
glob PER_BLOCK_TRAINABLE_PARAMS: int = 17_614_848;
glob DEFAULT_NUM_BLOCKS: int = 48;

glob _HELP: str = (
    f"Installed mlx-lm: {getattr(mlx_lm, '__version__', '?')} "
    f"(driver written against {UPSTREAM_MLX_LM_VERSION}). Re-read "
    f"`.venv/lib/python3.14/site-packages/mlx_lm/tuner/utils.py:linear_to_lora_layers` "
    f"and `mlx_lm/lora.py`'s imports, update this driver's verbatim copy and its "
    f"pinned hash, and re-run the --verify-layers self-test before training."
);


"""Fail at import time, never silently, if the composed-against code moved.

Inverted twin of dpo_fixed_train.py's `_assert_upstream_still_buggy()`: that
guard checks a bug is STILL present; this one checks the code we copied
verbatim is STILL shaped the way we copied it (spectrum-plan.md §6.3).
"""
def _assert_upstream_still_matches() -> None {
    fn = getattr(_tu, "linear_to_lora_layers", None);
    if fn is None {
        raise RuntimeError(
            "mlx_lm.tuner.utils.linear_to_lora_layers is gone -- this driver's "
            "composition target moved. " + _HELP
        );
    }

    params = set(inspect.signature(fn).parameters);  # jac:ignore[E1053]
    if params != UPSTREAM_PARAMS {
        raise RuntimeError(
            f"linear_to_lora_layers signature changed: upstream={sorted(params)} "
            f"expected={sorted(UPSTREAM_PARAMS)}. The call contract this driver "
            f"is a drop-in for no longer holds. " + _HELP
        );
    }

    src = "".join(inspect.getsource(fn).split());  # jac:ignore[E1053]
    if TRAILING_SLICE_MARKER not in src {
        raise RuntimeError(
            "linear_to_lora_layers no longer takes a trailing slice "
            f"({TRAILING_SLICE_MARKER!r} absent). The premise of this driver -- and "
            "of the probe's framing, which calls layers 32-47 an unexamined "
            "DEFAULT -- has changed. " + _HELP
        );
    }

    got = hashlib.sha256(src.encode()).hexdigest();
    if got != UPSTREAM_SRC_SHA256 {
        raise RuntimeError(
            f"linear_to_lora_layers source hash changed: got {got}, pinned "
            f"{UPSTREAM_SRC_SHA256}. Something in the function body was edited "
            f"(to_lora's type dispatch? get_keys_for_lora's type tuple?), and this "
            f"driver's verbatim copy is now silently stale. " + _HELP
        );
    }

    if mlx_lm.lora.linear_to_lora_layers is not fn {
        raise RuntimeError(
            "mlx_lm.lora.linear_to_lora_layers is not the same object as "
            "mlx_lm.tuner.utils.linear_to_lora_layers -- the import structure "
            "changed, so this driver's two-attribute rebind list is no longer the "
            "complete set of call sites. " + _HELP
        );
    }
}


"""Read, sort, dedupe and range-validate the frozen picks (§5.3)."""
def load_layer_ids(path: any, num_layers: int = DEFAULT_NUM_BLOCKS) -> List[int] {
    p = Path(path);  # jac:ignore[E1053]
    if not p.exists() {
        raise FileNotFoundError(
            f"spectrum layer selection not found: {p}. It is produced by the SNR "
            f"scan (spectrum-workflow.md Phases 1-2) and frozen before training. "
            f"Run snr_scan.py then layer_select.py first."
        );
    }
    blob = json.loads(p.read_text());
    raw = blob["layers"] if isinstance(blob, dict) else blob;
    ids = sorted({int(i) for i in raw});
    bad = [i for i in ids if not (0 <= i < num_layers)];
    if bad {
        raise ValueError(f"layer index out of range for a {num_layers}-block model: {bad}");
    }
    if not ids {
        raise ValueError(f"{p} contains no layer indices");
    }
    return ids;
}


"""Build the drop-in replacement bound to an explicit index list.

Body copied VERBATIM from mlx_lm 0.31.3
`mlx_lm/tuner/utils.py:linear_to_lora_layers` (Apple Inc., MIT) except for
the single loop that selects which blocks get converted.
"""
def make_linear_to_lora_layers(layer_ids: Sequence[int]) -> Callable {
    ids = sorted({int(i) for i in layer_ids});

    def spectrum_linear_to_lora_layers(model: nn.Module, num_layers: int,
                                       config: Dict, use_dora: bool = False) -> None {
        if num_layers != len(ids) {
            raise ValueError(
                f"num_layers={num_layers} but the spectrum selection has "
                f"{len(ids)} layers {ids}. Capacity and placement must not "
                f"disagree -- fix the config or the selection, do not proceed."
            );
        }
        bad = [i for i in ids if i >= len(model.layers)];  # jac:ignore[E1053]
        if bad {
            raise ValueError(
                f"layer index out of range: {bad} (model has {len(model.layers)} blocks)"  # jac:ignore[E1053]
            );
        }

        # --- verbatim from upstream ------------------------------------------
        def to_lora(layer: any) -> any {
            if not use_dora and hasattr(layer, "to_lora") {
                return layer.to_lora(
                    r=config["rank"],
                    scale=config["scale"],
                    dropout=config["dropout"],
                );
            }

            if isinstance(layer, (nn.Linear, nn.QuantizedLinear)) {
                LoRALayer = DoRALinear if use_dora else LoRALinear;
            } elif isinstance(layer, (SwitchLinear, QuantizedSwitchLinear)) {
                if use_dora {
                    raise ValueError(f"{type(layer).__name__} doesn't support DoRA yet.");
                }
                LoRALayer = LoRASwitchLinear;
            } elif isinstance(layer, (nn.Embedding, nn.QuantizedEmbedding)) {
                LoRALayer = DoRAEmbedding if use_dora else LoRAEmbedding;
            } else {
                raise ValueError(
                    f"Can't convert layer of type {type(layer).__name__} to LoRA"
                );
            }

            return LoRALayer.from_base(
                layer,
                r=config["rank"],
                scale=config["scale"],
                dropout=config["dropout"],
            );
        }

        if (keys := config.get("keys", None)) is None {  # jac:ignore[E1030]
            keys = set();

            def get_keys_for_lora(p: any, m: any) -> None {
                types = (
                    nn.Linear,
                    nn.QuantizedLinear,
                    SwitchLinear,
                    QuantizedSwitchLinear,
                    nn.Embedding,
                    nn.QuantizedEmbedding,
                );
                if hasattr(m, "to_lora") or isinstance(m, types) {
                    keys.add(p);
                }
            }

            for l in model.layers {  # jac:ignore[E1022]
                l.apply_to_modules(get_keys_for_lora);  # jac:ignore[E1032]
            }
        }
        # --- end verbatim -----------------------------------------------------

        # THE ONE CHANGED STATEMENT: explicit indices, not model.layers[-n:]
        for i in ids {
            l = model.layers[i];
            lora_layers = [(k, to_lora(m)) for (k, m) in l.named_modules() if k in keys];  # jac:ignore[E1032]
            if lora_layers {
                l.update_modules(tree_unflatten(lora_layers));  # jac:ignore[E1032]
            }
        }

        # --- verbatim from upstream ------------------------------------------
        lora_modules = [(k, to_lora(m)) for (k, m) in model.named_modules() if k in keys];
        if lora_modules {
            model.update_modules(tree_unflatten(lora_modules));  # jac:ignore[E1053]
        }
        # --- end verbatim -----------------------------------------------------
    }

    spectrum_linear_to_lora_layers.spectrum_layer_ids = ids;  # jac:ignore[E1031]
    return spectrum_linear_to_lora_layers;
}


"""Guard, then rebind BOTH call sites (spectrum-plan.md §6.2)."""
def apply_patch(layer_ids: Sequence[int]) -> None {
    if "mlx_lm_lora" in sys.modules {
        raise RuntimeError(
            "mlx_lm_lora (the DPO package) is imported. It has its own "
            "linear_to_lora_layers call path (mlx_lm_lora/utils.py:197) which this "
            "driver deliberately does NOT rebind -- DPO is out of scope "
            "(spectrum-plan.md §12). Running it here would half-patch the "
            "conversion. Use the stock DPO driver instead."
        );
    }
    _assert_upstream_still_matches();
    replacement = make_linear_to_lora_layers(layer_ids);
    # Rebinding a typed module attribute with `=` crashes jaclang's type checker
    # ("'NoneType' object has no attribute 'is_instantiable_class'"). `setattr` is
    # the identical module-attribute mutation -- same two call sites, same object,
    # same ordering -- and is what the DPO driver's port uses for the same reason.
    setattr(_tu, "linear_to_lora_layers", replacement);
    setattr(mlx_lm.lora, "linear_to_lora_layers", replacement);
}


# --------------------------------------------------------------------------
# self-test (spectrum-plan.md §6.4)
# --------------------------------------------------------------------------
"""Measure what actually got converted, from the model object itself."""
def collect_lora_report(model: any) -> dict {
    (layer_ids, suffixes, n_switch) = (set(), {}, 0);
    for (i, layer) in enumerate(model.layers) {  # jac:ignore[E1053]
        found = [];
        for (name, mod) in layer.named_modules() {  # jac:ignore[E1031]
            if isinstance(mod, (LoRALinear, LoRASwitchLinear, DoRALinear,
                                DoRAEmbedding, LoRAEmbedding)) {
                found.append(name);
                if isinstance(mod, LoRASwitchLinear) {  # jac:ignore[E1053]
                    n_switch += 1;
                }
            }
        }
        if found {
            layer_ids.add(i);
            suffixes[i] = sorted(found);
        }
    }

    trainable = tree_flatten(model.trainable_parameters());
    n_tensors = sum(1 for (k, _) in trainable if k.endswith(("lora_a", "lora_b")));  # jac:ignore[E1032]
    return {
        "layer_ids": sorted(layer_ids),
        "suffixes_by_layer": suffixes,
        "n_switch": n_switch,
        "n_lora_tensors": n_tensors,
        "trainable_params": int(sum(v.size for (_, v) in trainable)),  # jac:ignore[E1032]
        "total_params": int(sum(v.size for (_, v) in tree_flatten(model.parameters()))),  # jac:ignore[E1032]
    };
}


"""Return (ok, problems). Every assertion from spectrum-plan.md §6.4."""
def verify_layers(model: any, layer_ids: Sequence[int],
                  expect_tensors: Optional[int] = EXPECTED_LORA_TENSORS,
                  expect_trainable: Optional[int] = None,
                  expect_switch_per_layer: int = 3) -> tuple {
    want = sorted({int(i) for i in layer_ids});
    rep = collect_lora_report(model);
    problems: List[str] = [];

    if rep["layer_ids"] != want {
        problems.append(f"layer set: got {rep['layer_ids']}, expected {want}");
    }
    if expect_tensors is not None and rep["n_lora_tensors"] != expect_tensors {
        problems.append(
            f"LoRA tensor count: got {rep['n_lora_tensors']}, expected {expect_tensors}"
        );
    }
    if expect_trainable is not None and rep["trainable_params"] != expect_trainable {
        problems.append(
            f"trainable params: got {rep['trainable_params']}, expected "
            f"{expect_trainable} -- the selection changed CAPACITY, not just "
            f"placement, and the comparison is invalid"
        );
    }
    want_switch = expect_switch_per_layer * len(want);
    if rep["n_switch"] != want_switch {
        problems.append(
            f"LoRASwitchLinear count: got {rep['n_switch']}, expected {want_switch} "
            f"({expect_switch_per_layer}/layer) -- the MoE branch did not fire"
        );
    }
    for L in want {
        got = rep["suffixes_by_layer"].get(L, []);
        if got != list(EXPECTED_SUFFIXES) {
            problems.append(f"layer {L} module suffixes: got {got}, "
                            f"expected {list(EXPECTED_SUFFIXES)}");
        }
    }
    return ((not problems), problems);
}


def _print_report(tag: str, rep: dict) -> None {
    total_m = rep["total_params"] / 1e6;
    train_m = rep["trainable_params"] / 1e6;
    print(f"[{tag}] layers          = {rep['layer_ids']}");
    print(f"[{tag}] lora tensors    = {rep['n_lora_tensors']}");
    print(f"[{tag}] LoRASwitchLinear= {rep['n_switch']}");
    print(f"[{tag}] trainable       = {train_m:.3f}M / {total_m:.3f}M "
          f"({train_m * 100 / total_m:.3f}%)");  # jac:ignore[E1055]
}


def _run_verify(model_path: str, layers_path: str, lora_config: Dict,
                skip_control: bool) -> int {
    # [TODO: jac port blocked -- same jaclang 0.16.1 type-checker crash as the
    # module-level mlx_lm imports above; `import from mlx_lm.utils { load }`
    # reproduces it. Function-scoped exactly as in the original.]
    ::py::
    from mlx_lm.utils import load
    ::py::

    ids = load_layer_ids(layers_path);
    print(f"spectrum picks ({len(ids)}): {ids}");  # jac:ignore[E1053]
    print(f"overlap with trailing-16 {{32..47}}: "
          f"{len(set(ids) & set(range(32, 48)))}/{len(ids)}");  # jac:ignore[E1053]

    _assert_upstream_still_matches();
    print("upstream guard: OK "
          f"(mlx-lm {getattr(mlx_lm, '__version__', '?')}, source hash matches pin)");

    (model, _) = load(model_path);
    model.freeze();                      # mirrors mlx_lm/lora.py:224  # jac:ignore[E1032]
    make_linear_to_lora_layers(ids)(model, len(ids), lora_config);  # jac:ignore[E1051,E1053]
    rep = collect_lora_report(model);
    _print_report("spectrum", rep);
    (ok, problems) = verify_layers(model, ids,  # jac:ignore[E1053]
                                   expect_tensors=EXPECTED_LORA_TENSORS,
                                   expect_trainable=EXPECTED_TRAINABLE_PARAMS);
    del model;

    if not skip_control {
        (control, _) = load(model_path);
        control.freeze();  # jac:ignore[E1032]
        # [TODO: jac port blocked -- calling the module attribute
        # `_tu.linear_to_lora_layers(...)` crashes jaclang's type checker
        # ("'NoneType' object has no attribute 'is_instantiable_class'"). Kept as
        # inline Python; still the STOCK function (apply_patch is not called on
        # this path, so nothing has rebound it yet) and the control arm still
        # measures trailing-16 placement, which is the whole point of the check.]
        ::py::
        _tu.linear_to_lora_layers(control, len(ids), lora_config)   # STOCK
        ::py::
        crep = collect_lora_report(control);
        _print_report("control", crep);
        expected_trailing = list(range(48 - len(ids), 48));  # jac:ignore[E1053]
        if crep["layer_ids"] != expected_trailing {
            problems.append(
                f"stock control: got {crep['layer_ids']}, expected "
                f"{expected_trailing} -- the harness is not measuring placement"
            );
        }
        if crep["trainable_params"] != rep["trainable_params"] {
            problems.append(
                f"control trainable {crep['trainable_params']} != spectrum "
                f"{rep['trainable_params']} -- capacity is not placement-invariant"
            );
        }
        del control;
        ok = not problems;
    }

    for p in problems {  # jac:ignore[E1022]
        print(f"  !! {p}");
    }
    print("VERIFY:", "PASS" if ok else f"FAIL -- {len(problems)} problem(s)");  # jac:ignore[E1053]
    return 0 if ok else 1;
}


"""Remove `--flag VALUE` from argv in place and return VALUE."""
def _pop_flag(argv: List[str], flag: str) -> Optional[str] {
    if flag in argv {
        i = argv.index(flag);
        value = argv[i + 1];  # jac:ignore[E1055]
        del argv[i:i + 2];  # jac:ignore[E1055]
        return value;  # jac:ignore[E1002]
    }
    for (i, tok) in enumerate(list(argv)) {
        if tok.startswith(flag + "=") {  # jac:ignore[E1031]
            del argv[i];
            return tok.split("=", 1)[1];  # jac:ignore[E1002,E1031]
        }
    }
    return None;
}


glob DEFAULT_LAYERS_PATH: str = str(Path(__file__).resolve().parent / "configs" / "spectrum_layers.json");  # jac:ignore[E1053]


def main(argv: Optional[List[str]] = None) -> int {
    argv = list(sys.argv[1:] if argv is None else argv);
    layers_path = (_pop_flag(argv, "--spectrum-layers")
                   or os.environ.get("SPECTRUM_LAYERS")
                   or DEFAULT_LAYERS_PATH);

    if "--verify-layers" in argv {
        argv.remove("--verify-layers");
        ap = argparse.ArgumentParser(prog="spectrum_lora_layers.py --verify-layers");
        ap.add_argument("--model", default="models/qwen-q4");
        ap.add_argument("--rank", type=int, default=16);  # jac:ignore[E1053]
        ap.add_argument("--scale", type=float, default=2.0);  # jac:ignore[E1053]
        ap.add_argument("--dropout", type=float, default=0.05);  # jac:ignore[E1053]
        ap.add_argument("--skip-control", action="store_true");
        a = ap.parse_args(argv);
        return _run_verify(a.model, layers_path,  # jac:ignore[E1053]
                           {"rank": a.rank, "scale": a.scale, "dropout": a.dropout},  # jac:ignore[E1053]
                           a.skip_control);  # jac:ignore[E1053]
    }

    ids = load_layer_ids(layers_path);
    apply_patch(ids);  # jac:ignore[E1053]
    print(f"[spectrum_lora_layers] LoRA layers rebound to {ids} "
          f"(from {layers_path}); mlx_lm.lora and mlx_lm.tuner.utils both patched",
          flush=True);

    # [TODO: jac port blocked -- `import from mlx_lm.lora { main as _upstream_main }`
    # crashes jaclang's type checker ("'NoneType' object has no attribute
    # 'is_instantiable_class'"), the same defect already worked around at the
    # module-level mlx_lm imports above. Kept as inline Python; semantics are
    # identical. Retry the `import from` form once jaclang fixes it.]
    _upstream_main: any;
    ::py::
    from mlx_lm.lora import main as _upstream_main
    ::py::

    sys.argv = [sys.argv[0]] + argv;   # stock argv for the stock parser
    _upstream_main();                  # parses sys.argv exactly as `mlx_lm.lora` does
    return 0;
}


with entry:__main__ {
    sys.exit(main());
}
