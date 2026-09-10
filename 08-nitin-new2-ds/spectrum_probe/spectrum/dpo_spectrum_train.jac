"""DPO on the Spectrum-picked layers -- `dpo_fixed_train.jac` + the layer rebind.

SCOPE NOTE (read this first)
----------------------------
`spectrum-plan.md` §12 lists "a DPO stage on the spectrum adapter" as OUT of
scope, and §6.2's rebind table marks `mlx_lm_lora.utils.linear_to_lora_layers`
"no -- DPO is out of scope; the driver asserts it is *not* being used rather than
rebinding it". That decision was reversed by the user on 2026-08-02: DPO for this
probe continues the SAME Spectrum-picked layers as SFT (same LoRA lineage, not
the stock trailing-16), so it is "the existing stock DPO recipe with Spectrum's
layer set", exactly mirroring how SFT-Spectrum is "the existing stock SFT recipe
with Spectrum's layer set". This module is that third rebind target, plus the
composition with the chat-template fix. `spectrum_lora_layers.py` is UNCHANGED:
its `apply_patch()` still refuses to run with `mlx_lm_lora` imported, because the
SFT path genuinely must not half-patch the DPO package. DPO comes through here.

DOES `mlx_lm_lora` SHARE `mlx_lm`'s `linear_to_lora_layers`? -- ANSWERED, NOT ASSUMED
------------------------------------------------------------------------------------
Read from the installed source, `mlx_lm_lora` 2.1.0 / `mlx_lm` 0.31.3:

    mlx_lm_lora/utils.py:15   from mlx_lm.tuner.utils import linear_to_lora_layers
    mlx_lm_lora/utils.py:197  linear_to_lora_layers(model=model, num_layers=..., ...)
    mlx_lm_lora/train.py:49   from .utils import from_pretrained, ...
    mlx_lm_lora/train.py:~1081 run() -> from_pretrained(...)  # THE conversion
    mlx_lm_lora/train.py:527  train_model() -> model.load_weights(
                                  args.resume_adapter_file, strict=False)

So the answer is BOTH halves, and the second half is the load-bearing one:

  * It is the SAME FUNCTION OBJECT -- `mlx_lm_lora` has no conversion code of its
    own, it reuses `mlx_lm.tuner.utils.linear_to_lora_layers` verbatim. The
    verbatim body copied into `spectrum_lora_layers.py` is therefore correct for
    the DPO path too; no second copy is needed.
  * It is bound into `mlx_lm_lora.utils`'s OWN module globals by a `from ...
    import`. Rebinding `mlx_lm.tuner.utils.linear_to_lora_layers` after that
    module has been imported does NOT reach it. `dpo_fixed_train.jac` imports
    `mlx_lm_lora.trainer.datasets` at module scope, so by the time any patch
    runs, `mlx_lm_lora` is already in `sys.modules` -- "patch early and let the
    from-import pick it up" would be an import-order coincidence, not a design.

Hence THREE explicit rebind sites, asserted after the fact rather than assumed:

    mlx_lm.tuner.utils.linear_to_lora_layers  -- load_adapters (utils.py:131)
    mlx_lm.lora.linear_to_lora_layers         -- mlx_lm's own train_model
    mlx_lm_lora.utils.linear_to_lora_layers   -- from_pretrained (utils.py:197)
                                                 = the DPO training path

COMPOSING TWO MONKEY-PATCHES
----------------------------
Both fixes are real and both must be live in the same process:

  1. `dpo_fixed_train.FixedDPODataset` -> `mlx_lm_lora.trainer.datasets.DPODataset`
     (`add_generation_prompt=False`; see that module for the diagnosis)
  2. the Spectrum layer list -> the three attributes above

They touch disjoint attributes, so they cannot clobber each other -- but each
one's GUARD is a one-shot that only reads true against unpatched upstream
(`_assert_upstream_still_buggy` requires `add_generation_prompt=True` to still be
in the source; `_assert_upstream_still_matches` requires the trailing slice to
still be in the source). Apply either patch twice, or run a guard after its own
patch, and it raises a message about upstream having changed -- which would be a
lie. `apply_patches()` therefore runs BOTH guards BEFORE EITHER patch, refuses a
double-apply with an accurate message, and calls `assert_patches_active()`
afterwards so a half-applied composition fails here instead of silently training
the wrong thing. `test_dpo_spectrum_train.py` covers all of that, in both
application orders.

WHY THE RESUME PATH IS THE DANGEROUS ONE
----------------------------------------
`train.py:527` loads `--resume-adapter-file` with `strict=False`, AFTER
conversion. Any key naming a block that was not converted is dropped without a
word -- the same silent-loss class as the `mlx_lm.fuse` bug (comparison report
§3) and the eval-time trap (`adapter_config_fix.py`). The SFT-Spectrum adapter
this run resumes from names the picks, and the picks are what we convert, so the
fresh arm is safe by construction -- `--verify-layers` asserts it against the
real file rather than trusting that sentence. The cptv2 arm is NOT safe by
construction; see `sft_cptv2_probe/spectrum/cptv2_dpo_spectrum_train.py`.

USAGE
-----
    # cheap gate, no model load: are both patches live and on the right layers?
    jac run dpo_spectrum_train.jac --verify-patches --spectrum-layers configs/spectrum_layers.json

    # the chat-template half, against real training rows (from dpo_fixed_train.jac)
    jac run dpo_spectrum_train.jac --verify-tokenization --model models/qwen-q4 --data <dpo dir>

    # the expensive gate: real 30B model through the REAL DPO conversion path
    jac run dpo_spectrum_train.jac --verify-layers --model models/qwen-q4 \\
        --spectrum-layers configs/spectrum_layers.json \\
        --resume-adapter-file .../adapters/sft-on-fresh-spectrum/adapters.safetensors

    # training: every other flag is stock `python -m mlx_lm_lora.train`
    jac run dpo_spectrum_train.jac --spectrum-layers configs/spectrum_layers.json \\
        --model models/qwen-q4 --train --train-mode dpo ...
"""

import argparse;
import hashlib;
import inspect;
import json;
import os;
import sys;
import tempfile;
import from pathlib { Path }
import from typing { Callable, Dict, List, Optional, Sequence }

glob SELF_DIR: Path = Path(__file__ as str).resolve().parent;   # .../sft_fresh_probe/spectrum
glob PROBE_DIR: Path = SELF_DIR.parent;                         # .../sft_fresh_probe
glob REPO_ROOT: Path = SELF_DIR.parents[3];                     # repo root

with entry {
    for _p in (str(SELF_DIR), str(PROBE_DIR)) {
        if _p not in sys.path {
            sys.path.insert(0, _p);
        }
    }
}

import mlx_lm;
import mlx_lm.lora;
import mlx_lm.tuner.utils as _tu;

import spectrum_lora_layers as S;            # noqa: E402  (sys.path set above)
import dpo_fixed_train as F;                 # noqa: E402  imports mlx_lm_lora.trainer.datasets

import mlx_lm_lora;                          # noqa: E402
import mlx_lm_lora.train as _mt;             # noqa: E402
import mlx_lm_lora.utils as _mu;             # noqa: E402
import mlx_lm_lora.trainer.datasets as _ds;  # noqa: E402

# --- pinned upstream facts (verified against the installed venv 2026-08-02) ---
glob UPSTREAM_MLX_LM_LORA_VERSION: str = "2.1.0";
# whitespace-stripped source hashes of the two functions this driver composes with
glob FROM_PRETRAINED_SHA256: str = "e71fe3158508c6c9c70cda0c3ec6b9cabc254669ce9538338aea147b58b4fdb9";
glob TRAIN_MODEL_SHA256: str = "e13834497df5a28cd32c384574ff83d61ecab926da5c5fdd995e973ecd399a7d";
# the exact call this driver intercepts (mlx_lm_lora/utils.py:197), whitespace-stripped
glob CONVERT_CALL_MARKER: str = 'linear_to_lora_layers(model=model,num_layers=lora_config.get("num_layers",None),';
# the resume that runs AFTER conversion with strict=False (mlx_lm_lora/train.py:527)
glob RESUME_MARKER: str = "model.load_weights(args.resume_adapter_file,strict=False)";

glob DEFAULT_LAYERS_PATH: str = str(SELF_DIR / "configs" / "spectrum_layers.json");

glob _HELP: str = (
    f"Installed mlx-lm-lora: {getattr(mlx_lm_lora, '__version__', '?')} "
    f"(driver written against {UPSTREAM_MLX_LM_LORA_VERSION}). Re-read "
    f"`.venv/lib/python3.14/site-packages/mlx_lm_lora/utils.py:from_pretrained` and "
    f"`mlx_lm_lora/train.py:{{run,train_model}}`, update this driver's pins, and "
    f"re-run --verify-patches and --verify-layers before training."
);


def _stripped_source(fn: any) -> str {
    return "".join(inspect.getsource(fn as Callable).split());
}


# --------------------------------------------------------------------------
# guards -- fail at import/patch time, never silently
# --------------------------------------------------------------------------
def assert_dpo_upstream_still_matches() -> None {
    """The `mlx_lm_lora` half of the composition contract.

    Twin of `spectrum_lora_layers._assert_upstream_still_matches()` (the mlx_lm
    half) and of `dpo_fixed_train._assert_upstream_still_buggy()` (the dataset
    half). Runs BEFORE any patch: every check below reads the STOCK source.
    """;
    ver = getattr(mlx_lm_lora, "__version__", None);
    if ver != UPSTREAM_MLX_LM_LORA_VERSION {
        raise RuntimeError(
            f"mlx-lm-lora is {ver}, this driver is pinned to "
            f"{UPSTREAM_MLX_LM_LORA_VERSION}. " + _HELP
        );
    }

    fp = getattr(_mu, "from_pretrained", None);
    if fp is None {
        raise RuntimeError(
            "mlx_lm_lora.utils.from_pretrained is gone -- the DPO conversion path "
            "this driver intercepts moved. " + _HELP
        );
    }

    fp_src = _stripped_source(fp);
    if CONVERT_CALL_MARKER not in fp_src {
        raise RuntimeError(
            "mlx_lm_lora.utils.from_pretrained no longer calls "
            f"{CONVERT_CALL_MARKER!r}. Either it stopped converting to LoRA there, "
            "or the call shape changed, so rebinding the module attribute may no "
            "longer intercept the DPO training path. " + _HELP
        );
    }
    got = hashlib.sha256(fp_src.encode()).hexdigest();
    if got != FROM_PRETRAINED_SHA256 {
        raise RuntimeError(
            f"from_pretrained source hash changed: got {got}, pinned "
            f"{FROM_PRETRAINED_SHA256}. " + _HELP
        );
    }

    tm = getattr(_mt, "train_model", None);
    if tm is None {
        raise RuntimeError("mlx_lm_lora.train.train_model is gone. " + _HELP);
    }
    tm_src = _stripped_source(tm);
    if RESUME_MARKER not in tm_src {
        raise RuntimeError(
            f"mlx_lm_lora.train.train_model no longer does {RESUME_MARKER!r}. "
            "The --resume-adapter-file contract this probe's whole no-fuse recipe "
            "rests on (comparison report §3) has changed. " + _HELP
        );
    }
    got = hashlib.sha256(tm_src.encode()).hexdigest();
    if got != TRAIN_MODEL_SHA256 {
        raise RuntimeError(
            f"train_model source hash changed: got {got}, pinned "
            f"{TRAIN_MODEL_SHA256}. " + _HELP
        );
    }

    run_src = _stripped_source((_mt as any).run);
    (i_conv, i_train) = (run_src.find("from_pretrained("), run_src.find("train_model("));
    if i_conv < 0 or i_train < 0 or i_conv > i_train {
        raise RuntimeError(
            "mlx_lm_lora.train.run no longer converts (from_pretrained) BEFORE it "
            "resumes (train_model). The ordering is what makes an unconverted "
            "block's resume keys vanish under strict=False -- if it changed, the "
            "union/freeze reasoning on the cptv2 arm needs re-deriving. " + _HELP
        );
    }

    if (_mu as any).linear_to_lora_layers is not (_tu as any).linear_to_lora_layers {
        raise RuntimeError(
            "mlx_lm_lora.utils.linear_to_lora_layers is not the same object as "
            "mlx_lm.tuner.utils.linear_to_lora_layers. Either this driver already "
            "ran (call apply_patches once, not twice) or mlx_lm_lora grew its own "
            "conversion implementation, in which case spectrum_lora_layers.py's "
            "verbatim copy is no longer the right body for the DPO path. " + _HELP
        );
    }
}


# --------------------------------------------------------------------------
# the patch
# --------------------------------------------------------------------------
def patch_sites() -> list[tuple[any, str]] {
    """(module, attribute) for every name that resolves the conversion function.

    Three, not two: `mlx_lm_lora/utils.py:15` is a `from ... import`, so it holds
    an INDEPENDENT binding that a rebind of `mlx_lm.tuner.utils` cannot reach.
    """;
    return [
        (_tu, "linear_to_lora_layers"),
        ((mlx_lm as any).lora, "linear_to_lora_layers"),
        (_mu, "linear_to_lora_layers")
    ];
}


def already_patched() -> bool {
    return (any(
        [hasattr(getattr(mod, attr, None), "spectrum_layer_ids")
         for (mod, attr) in patch_sites()]
    ) or ((_ds as any).DPODataset is (F as any).FixedDPODataset)) as bool;
}


def apply_patches(layer_ids: Optional[Sequence[int]] = None,
                  replacement: Optional[Callable] = None) -> Callable {
    """Guard both halves, then apply both. Returns the installed replacement.

    `replacement` is the extension point the cptv2 arm uses: it passes
    `cptv2_spectrum_lora_layers.make_union_linear_to_lora_layers(picks, base)`
    instead of the plain picks-only conversion, and everything else -- the
    guards, the three rebind sites, the chat-template fix, the post-condition --
    is shared rather than reimplemented.
    """;
    if (layer_ids is None) == (replacement is None) {
        raise ValueError("pass exactly one of layer_ids= or replacement=");
    }

    if already_patched() {
        raise RuntimeError(
            "apply_patches() has already run in this process. Both guards read "
            "the STOCK upstream source, so a second call would raise a message "
            "claiming upstream changed, which would be false. Patch once per "
            "process."
        );
    }

    # both guards first, while upstream is still stock -----------------------
    (S as any)._assert_upstream_still_matches();      # mlx_lm side (verbatim-copy contract)
    assert_dpo_upstream_still_matches();              # mlx_lm_lora side (call-site contract)
    (F as any)._assert_upstream_still_buggy();        # dataset side (the bug is still there)

    # then both patches -------------------------------------------------------
    (F as any).apply_patch();                          # DPODataset -> FixedDPODataset
    fn = (replacement if replacement is not None else (S as any).make_linear_to_lora_layers(layer_ids)) as Callable;
    for (mod, attr) in patch_sites() {
        setattr(mod, attr, fn);
    }

    assert_patches_active(getattr(fn, "spectrum_layer_ids", None) as (Sequence[int] | None));
    return fn;
}


def assert_patches_active(expect_ids: Optional[Sequence[int]] = None) -> dict {
    """Post-condition. A half-applied composition raises HERE, not at iter 200.""";
    problems: list[str] = [];

    if (_ds as any).DPODataset is not (F as any).FixedDPODataset {
        problems.append(
            "mlx_lm_lora.trainer.datasets.DPODataset is NOT FixedDPODataset -- the "
            "chat-template fix is not live, so every preference pair would get a "
            "spurious trailing <|im_start|>assistant tag (dpo_fixed_train.jac)"
        );
    }

    installed = [];
    for (mod, attr) in patch_sites() {
        fn = getattr(mod, attr, None);
        installed.append(fn);
        if not hasattr(fn, "spectrum_layer_ids") {
            problems.append(
                f"{mod.__name__}.{attr} is not the spectrum replacement -- that "
                f"call site would convert the stock trailing-16 instead"
            );
        }
    }
    if len({id(f) for f in installed}) != 1 {
        problems.append(
            "the three call sites do not hold the SAME object -- one of them was "
            "rebound independently and the arms would disagree on placement"
        );
    }

    ids = getattr(installed[0], "spectrum_layer_ids", None);
    if expect_ids is not None and ids != sorted({int(i) for i in expect_ids}) {
        problems.append(f"layer ids: installed {ids}, expected {sorted(expect_ids)}");
    }

    if problems {
        raise RuntimeError(
            "DPO-spectrum patch composition is incomplete:\n  - "
            + "\n  - ".join(problems)
        );
    }
    return {
        "chat_template_fix": True,
        "layer_sites": [f"{m.__name__}.{a}" for (m, a) in patch_sites()],
        "spectrum_layer_ids": ids,
        "union_layer_ids": getattr(installed[0], "union_layer_ids", None),
        "frozen_layer_ids": getattr(installed[0], "frozen_layer_ids", None)
    };
}


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------
def _print_patch_report(rep: dict) -> None {
    print(f"[patches] chat-template fix : DPODataset -> FixedDPODataset  (OK)");
    for s in (rep["layer_sites"] as list) {
        print(f"[patches] layer rebind      : {s}  (OK)");
    }
    print(f"[patches] spectrum layers   : {rep['spectrum_layer_ids']}");
    if rep.get("union_layer_ids") is not None {
        print(f"[patches] union converted   : {rep['union_layer_ids']}");
        print(f"[patches] frozen (no grad)  : {rep['frozen_layer_ids']}");
    }
}


def _run_verify_patches(layers_path: str) -> int {
    ids = (S as any).load_layer_ids(layers_path) as list[int];
    print(f"spectrum picks ({len(ids)}): {ids}");
    print(f"overlap with trailing-16 {{32..47}}: "
          f"{len(set(ids) & set(range(32, 48)))}/{len(ids)}");
    apply_patches(ids);
    rep = assert_patches_active(ids);
    _print_patch_report(rep);
    print(f"upstream guards: OK (mlx-lm {getattr(mlx_lm, '__version__', '?')}, "
          f"mlx-lm-lora {getattr(mlx_lm_lora, '__version__', '?')})");
    print("VERIFY: PASS");
    return 0;
}


def dpo_convert(model_path: str, lora_config: dict, tmpdir: str) -> any {
    """Run the REAL DPO conversion path (`mlx_lm_lora.utils.from_pretrained`).

    Not a stand-in call to `linear_to_lora_layers` -- this is the exact function
    `mlx_lm_lora.train.run()` calls, with the exact `lora_config` shape
    `build_lora_config()` produces, so the intercept is proven where it matters.
    """;
    return (_mu as any).from_pretrained(
        model=model_path,
        new_adapter_path=tmpdir,
        lora_config=lora_config
    );
}


def _run_verify_layers(model_path: str, layers_path: str, lora_params: dict,
                       resume_file: Optional[str], skip_control: bool) -> int {
    import from adapter_config_fix { assert_adapter_keys_present }

    ids = (S as any).load_layer_ids(layers_path) as list[int];
    stock = (_tu as any).linear_to_lora_layers;        # captured BEFORE the rebind

    print(f"spectrum picks ({len(ids)}): {ids}");
    apply_patches(ids);
    _print_patch_report(assert_patches_active(ids));

    lora_config = dict(lora_params);
    lora_config.update({"use_dora": False, "num_layers": len(ids)});

    with tempfile.TemporaryDirectory() as td {
        (model, _tok, _af) = dpo_convert(model_path, lora_config, td as str) as tuple[any, any, any];
        written = json.loads((Path(td as str) / "adapter_config.json").read_text());
    }
    print(f"[dpo] adapter_config.json written by from_pretrained: "
          f"num_layers={written.get('num_layers')} "
          f"(eval-time rewrite to {48 - min(ids)} is adapter_config_fix.py's job)");

    rep = (S as any).collect_lora_report(model);
    (S as any)._print_report("dpo-spectrum", rep);
    (ok, problems) = (S as any).verify_layers(model, ids,
                                   expect_tensors=(S as any).EXPECTED_LORA_TENSORS,
                                   expect_trainable=(S as any).EXPECTED_TRAINABLE_PARAMS) as tuple[bool, list[str]];

    if resume_file {
        # train.py:527 does this with strict=False, i.e. it will NOT tell you.
        n = assert_adapter_keys_present(model, resume_file);
        model.load_weights(str(resume_file), strict=False);
        print(f"[dpo] resume: all {n} keys of {resume_file} land in the converted model");
        if n != (S as any).EXPECTED_LORA_TENSORS {
            problems.append(
                f"resume adapter holds {n} keys, expected {(S as any).EXPECTED_LORA_TENSORS} "
                f"-- the SFT-spectrum artifact is not the shape this arm resumes from"
            );
            ok = False;
        }
    } else {
        print("[dpo] no --resume-adapter-file given: the strict=False key check "
              "(the whole point of the DPO verify) was SKIPPED");
    }

    del model;

    if not skip_control {
        # [TODO: jac port blocked -- `import from mlx_lm.utils { load }` crashes the
        # jaclang 0.16.1 type checker ("'NoneType' object has no attribute
        # 'is_instantiable_class'") on that one symbol; the Python form is identical.]
        ::py::
        from mlx_lm.utils import load
        ::py::
        (control, _) = load(model_path) as tuple[any, any];
        control.freeze();
        stock(control, len(ids), lora_params);            # STOCK trailing slice
        crep = (S as any).collect_lora_report(control);
        (S as any)._print_report("control(stock trailing)", crep);
        expected_trailing = list(range(48 - len(ids), 48));
        if crep["layer_ids"] != expected_trailing {
            problems.append(
                f"stock control: got {crep['layer_ids']}, expected "
                f"{expected_trailing} -- the harness is not measuring placement"
            );
        }
        if crep["trainable_params"] != rep["trainable_params"] {
            problems.append(
                f"control trainable {crep['trainable_params']} != dpo-spectrum "
                f"{rep['trainable_params']} -- capacity is not placement-invariant"
            );
        }
        del control;
        ok = not problems;
    }

    for p in problems {
        print(f"  !! {p}");
    }
    print("VERIFY:", "PASS" if ok else f"FAIL -- {len(problems)} problem(s)");
    return 0 if ok else 1;
}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_verify_parser(prog: str) -> argparse.ArgumentParser {
    ap = argparse.ArgumentParser(prog=prog);
    ap.add_argument("--model", default="models/qwen-q4");
    ap.add_argument("--rank", type=(int as Callable[[str], any]), default=16);
    ap.add_argument("--scale", type=(float as Callable[[str], any]), default=2.0);
    ap.add_argument("--dropout", type=(float as Callable[[str], any]), default=0.05);
    ap.add_argument("--resume-adapter-file", default=None);
    ap.add_argument("--skip-control", action="store_true");
    return ap;
}


def main(argv: Optional[List[str]] = None) -> int {
    argv = list(sys.argv[1:] if argv is None else argv);
    layers_path = ((S as any)._pop_flag(argv, "--spectrum-layers")
                   or os.environ.get("SPECTRUM_LAYERS")
                   or DEFAULT_LAYERS_PATH) as str;

    if "--verify-patches" in argv {
        argv.remove("--verify-patches");
        return _run_verify_patches(layers_path);
    }

    if "--verify-tokenization" in argv {
        # the chat-template half only, against REAL rows -- dpo_fixed_train.jac's
        # own check, reachable through this driver so the runner has one entry point
        model = argv[argv.index("--model") + 1];
        data = argv[argv.index("--data") + 1];
        (F as any)._assert_upstream_still_buggy();
        return (F as any)._verify_tokenization(model, data) as int;
    }

    if "--verify-layers" in argv {
        argv.remove("--verify-layers");
        a = build_verify_parser("dpo_spectrum_train.jac --verify-layers").parse_args(argv);
        return _run_verify_layers(
            a.model as str, layers_path,
            {"rank": a.rank, "scale": a.scale, "dropout": a.dropout},
            a.resume_adapter_file as (str | None), a.skip_control as bool
        );
    }

    ids = (S as any).load_layer_ids(layers_path) as list[int];
    apply_patches(ids);
    rep = assert_patches_active(ids);
    print(f"[dpo_spectrum_train] LoRA layers rebound to {ids} (from {layers_path});\n"
          f"  patched: {', '.join(rep['layer_sites'] as list[str])}\n"
          f"  DPODataset patched: add_generation_prompt=False (no spurious tag)",
          flush=True);

    import from mlx_lm_lora.train { main as _upstream_main }

    sys.argv = [sys.argv[0]] + argv;   # stock argv for the stock parser
    _upstream_main();
    return 0;
}


with entry:__main__ {
    sys.exit(main());
}
