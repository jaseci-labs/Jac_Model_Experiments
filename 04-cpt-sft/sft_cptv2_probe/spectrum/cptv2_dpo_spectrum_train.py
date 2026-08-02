#!/usr/bin/env python3
"""DPO on the Spectrum-picked layers, **cptv2 arm** -- union-convert + freeze.

WHY THIS IS NOT THE FRESH ARM'S DPO DRIVER WITH DIFFERENT PATHS
---------------------------------------------------------------
DPO here continues the SAME LoRA lineage as this arm's SFT, and that lineage is
not 16 blocks -- it is the union `picks | {32..47}` (21 blocks). Read the chain:

  * `cptv2_spectrum_lora_layers.py` converts the union and FREEZES
    `{32..47} \\ picks` (5 blocks) so the CPT-v2 delta stays in the model as base
    weights without consuming the 16-block trainable budget.
  * `mlx_lm/tuner/trainer.py` saves only `model.trainable_parameters()`, so
    `merge_frozen_keys.py` puts those 5 blocks (80 tensors) back into every SFT
    artifact. The adapter this DPO run resumes from therefore holds **336** keys
    over 21 blocks, not 256 over 16.
  * `mlx_lm_lora/train.py:527` applies `--resume-adapter-file` with
    `model.load_weights(..., strict=False)`, **after** conversion
    (`mlx_lm_lora/utils.py:197`, reached from `run()` before `train_model()` --
    asserted by `dpo_spectrum_train.assert_dpo_upstream_still_matches()`).

So a DPO driver that converted only the 16 picks would silently drop those same
80 tensors on the way in -- re-creating, one stage later, the exact bug the SFT
side already fixed. `strict=False` will not say a word. Same silent-weight-loss
class as the `mlx_lm.fuse` bug (comparison report §3) and the eval-time trap
(`adapter_config_fix.py`).

WHAT THIS DRIVER DOES
---------------------
1. Convert the union `picks | {32..47}`; freeze `{32..47} \\ picks`. Identical
   composition to the SFT arm -- literally
   `cptv2_spectrum_lora_layers.make_union_linear_to_lora_layers`, injected into
   `dpo_spectrum_train.apply_patches(replacement=...)` so the three rebind sites,
   both upstream guards, the chat-template fix and the post-condition are shared
   code rather than a second copy.
2. Assert the resume artifact is union-shaped AND that its frozen blocks are
   bit-identical to the CPT-v2 source (`assert_frozen_blocks_match_cpt_v2`).
   That is the proof the frozen-block discipline survived the SFT stage; it runs
   before DPO spends any compute.
3. Assert trainable == 281.838M, i.e. the same budget as the fresh spectrum arm
   and both incumbents. The union is placement plumbing, never extra capacity.
4. Post-training, `run_dpo_spectrum.sh` re-merges the frozen keys into every DPO
   artifact (final, snapshots, best) -- same `merge_frozen_keys.py`, same reason.

SCOPE NOTE
----------
`spectrum-plan.md` §12 marks DPO-on-spectrum out of scope; the user reversed that
on 2026-08-02 (DPO continues the same picked layers as SFT). See
`sft_fresh_probe/spectrum/dpo_spectrum_train.py`'s header for the full rationale
and for the `mlx_lm_lora` import-graph finding that drives the three-site rebind.

USAGE
-----
    # cheap gates, no model load
    python cptv2_dpo_spectrum_train.py --verify-patches
    python cptv2_dpo_spectrum_train.py --verify-resume <sft-spectrum>/adapters.safetensors

    # expensive gate: real 30B model through the REAL DPO conversion path
    python cptv2_dpo_spectrum_train.py --verify-layers --model models/qwen-q4 \\
        --resume-adapter-file <sft-spectrum>/adapters.safetensors

    # training: every other flag is stock `python -m mlx_lm_lora.train`
    python cptv2_dpo_spectrum_train.py --model models/qwen-q4 --train --train-mode dpo ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import mlx.core as mx
from mlx.utils import tree_flatten

SELF_DIR = Path(__file__).resolve().parent
if str(SELF_DIR) not in sys.path:
    sys.path.insert(0, str(SELF_DIR))

# C puts sft_fresh_probe/spectrum on sys.path (single source of truth for the
# picks and for the upstream-composition guard); D then puts sft_fresh_probe on
# it (where dpo_fixed_train.py lives). Import order matters.
import cptv2_spectrum_lora_layers as C   # noqa: E402
import merge_frozen_keys as M            # noqa: E402

FRESH_SPECTRUM_DIR = C.FRESH_SPECTRUM_DIR
FRESH_PROBE_DIR = Path(FRESH_SPECTRUM_DIR).parent
for _p in (str(FRESH_SPECTRUM_DIR), str(FRESH_PROBE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dpo_spectrum_train as D           # noqa: E402
import spectrum_lora_layers as S         # noqa: E402

import mlx_lm                            # noqa: E402
import mlx_lm.tuner.utils as _tu         # noqa: E402
import mlx_lm_lora.utils as _mu          # noqa: E402

REPO_ROOT = C.REPO_ROOT
CPT_V2_ADAPTER = C.CPT_V2_ADAPTER
KEYS_PER_BLOCK = M.KEYS_PER_BLOCK
DEFAULT_LAYERS_PATH = str(SELF_DIR / "configs" / "spectrum_layers.json")
DEFAULT_SFT_ADAPTER = (REPO_ROOT / "model-experiments" / "04-cpt-sft"
                       / "sft_cptv2_probe" / "adapters" / "sft-on-cptv2-spectrum"
                       / "adapters.safetensors")


# --------------------------------------------------------------------------
# the resume artifact: read it, do not trust the pipeline that wrote it
# --------------------------------------------------------------------------
def assert_frozen_blocks_match_cpt_v2(adapter_file, picks: Sequence[int],
                                      base_layers: Optional[Sequence[int]] = None,
                                      cpt_file=CPT_V2_ADAPTER) -> Dict:
    """Prove the frozen CPT-v2 blocks survived, byte for byte, into `adapter_file`.

    This is the check that makes "DPO continues the same lineage" a fact rather
    than an intention. `merge_frozen_keys.py` is supposed to have put blocks
    `{32..47} \\ picks` back after SFT; if the merge was skipped, half-run, or
    run against the wrong file, the adapter still LOADS fine (strict=False) and
    DPO would quietly start from a model that lost part of its CPT-v2 delta.
    """
    path = Path(adapter_file)
    if not path.exists():
        raise FileNotFoundError(
            f"resume adapter not found: {path}. The cptv2 DPO arm resumes from the "
            f"cptv2 SFT-spectrum adapter -- run run_sft_spectrum.sh (including its "
            f"§8.3 step-5 merge) and eval it first."
        )
    picked = sorted({int(i) for i in picks})
    if base_layers is None:
        base_layers = C.assert_cpt_v2_shape(cpt_file)
    union = C.union_layers(picked, base_layers)
    frozen = C.frozen_layers(picked, base_layers)

    got_layers = C.adapter_layer_ids(path)
    if got_layers != union:
        raise RuntimeError(
            f"{path} covers layers {got_layers}, expected the union {union}. "
            f"Either the §8.3 step-5 merge never ran (adapter holds only the "
            f"trainable picks -- the CPT-v2 delta is GONE) or this is the wrong "
            f"file. Do not start DPO from it."
        )
    blob = dict(mx.load(str(path)))
    if len(blob) != KEYS_PER_BLOCK * len(union):
        raise RuntimeError(
            f"{path} holds {len(blob)} tensors, expected "
            f"{KEYS_PER_BLOCK * len(union)} ({KEYS_PER_BLOCK}/block x {len(union)} "
            f"union blocks)"
        )

    cpt = dict(mx.load(str(cpt_file)))
    prefixes = tuple(f"model.layers.{i}." for i in frozen)
    checked, mismatched, missing = 0, [], []
    for k, v in cpt.items():
        if not k.startswith(prefixes):
            continue
        if k not in blob:
            missing.append(k)
            continue
        checked += 1
        if not mx.array_equal(blob[k], v):
            mismatched.append(k)
    if missing:
        raise RuntimeError(
            f"{len(missing)} frozen CPT-v2 keys are absent from {path} -- the "
            f"frozen blocks {frozen} did not survive. First: {sorted(missing)[:5]}"
        )
    if mismatched:
        raise RuntimeError(
            f"{len(mismatched)} frozen CPT-v2 keys in {path} differ from "
            f"{cpt_file}. They were supposed to carry NO gradients, so they must "
            f"be bit-identical. First: {sorted(mismatched)[:5]}"
        )
    if checked != KEYS_PER_BLOCK * len(frozen):
        raise RuntimeError(
            f"checked {checked} frozen keys, expected {KEYS_PER_BLOCK * len(frozen)}"
        )
    return {
        "adapter": str(path),
        "picks": picked,
        "union": union,
        "frozen_layers": frozen,
        "n_keys": len(blob),
        "n_frozen_verified": checked,
    }


# --------------------------------------------------------------------------
# the patch
# --------------------------------------------------------------------------
def apply_patches(picks: Sequence[int], base_layers: Sequence[int]):
    """Union-convert + freeze, through the shared three-site DPO rebind."""
    return D.apply_patches(
        replacement=C.make_union_linear_to_lora_layers(picks, base_layers)
    )


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------
def _print_header(picks, base, union, frozen) -> None:
    print(f"spectrum picks ({len(picks)}): {picks}")
    print(f"CPT-v2 base layers ({len(base)}): {base}   "
          f"[{C.EXPECTED_CPT_V2_KEYS} tensors, verified]")
    print(f"union to convert ({len(union)}): {union}")
    print(f"frozen (CPT-v2 delta kept, no gradients) ({len(frozen)}): {frozen}")


def _run_verify_patches(layers_path: str, cpt_adapter) -> int:
    picks = S.load_layer_ids(layers_path)
    base = C.assert_cpt_v2_shape(cpt_adapter)
    _print_header(picks, base, C.union_layers(picks, base), C.frozen_layers(picks, base))
    apply_patches(picks, base)
    rep = D.assert_patches_active(picks)
    D._print_patch_report(rep)
    print(f"upstream guards: OK (mlx-lm {getattr(mlx_lm, '__version__', '?')}, "
          f"mlx-lm-lora {__import__('mlx_lm_lora').__version__})")
    print("VERIFY: PASS")
    return 0


def _run_verify_resume(resume_file, layers_path: str, cpt_adapter) -> int:
    picks = S.load_layer_ids(layers_path)
    base = C.assert_cpt_v2_shape(cpt_adapter)
    info = assert_frozen_blocks_match_cpt_v2(resume_file, picks, base, cpt_adapter)
    print(f"resume adapter: {info['adapter']}")
    print(f"  covers the union {info['union']} ({info['n_keys']} tensors)")
    print(f"  frozen blocks {info['frozen_layers']}: all "
          f"{info['n_frozen_verified']} keys bit-identical to {cpt_adapter}")
    print("VERIFY: PASS")
    return 0


def _run_verify_layers(model_path: str, layers_path: str, lora_params: Dict,
                       cpt_adapter, resume_file: Optional[str],
                       skip_control: bool) -> int:
    from adapter_config_fix import assert_adapter_keys_present

    picks = S.load_layer_ids(layers_path)
    base = C.assert_cpt_v2_shape(cpt_adapter)
    union = C.union_layers(picks, base)
    frozen = C.frozen_layers(picks, base)
    stock = _tu.linear_to_lora_layers          # captured BEFORE the rebind
    _print_header(picks, base, union, frozen)

    if resume_file:
        info = assert_frozen_blocks_match_cpt_v2(resume_file, picks, base, cpt_adapter)
        print(f"resume adapter {resume_file}: union-shaped, "
              f"{info['n_frozen_verified']} frozen keys bit-match CPT-v2")

    apply_patches(picks, base)
    D._print_patch_report(D.assert_patches_active(picks))

    lora_config = dict(lora_params)
    lora_config.update({"use_dora": False, "num_layers": len(picks)})

    with tempfile.TemporaryDirectory() as td:
        model, _tok, _af = D.dpo_convert(model_path, lora_config, td)
        written = json.loads((Path(td) / "adapter_config.json").read_text())
    print(f"[dpo] adapter_config.json written by from_pretrained: "
          f"num_layers={written.get('num_layers')} "
          f"(eval-time rewrite to {48 - min(union)} over the UNION is "
          f"adapter_config_fix.py's job)")

    rep = S.collect_lora_report(model)
    S._print_report("cptv2-dpo-spectrum", rep)
    n_train = int(sum(v.size for _, v in tree_flatten(model.trainable_parameters())))
    print(f"[cptv2-dpo-spectrum] TRAINABLE layers = {C.trainable_layer_ids(model)}")
    print(f"[cptv2-dpo-spectrum] TRAINABLE params = {n_train / 1e6:.3f}M "
          f"(frozen union members contribute 0)")

    ok, problems = C.verify_composition(
        model, picks, base,
        expect_trainable=C.EXPECTED_TRAINABLE_PARAMS,
        expect_tensors=C.EXPECTED_LORA_TENSORS,
    )

    if resume_file:
        # train.py:527 does this with strict=False, i.e. it will NOT tell you.
        n = assert_adapter_keys_present(model, resume_file)
        model.load_weights(str(resume_file), strict=False)
        print(f"[dpo] resume: all {n} keys of {resume_file} land in the converted model")
        want = KEYS_PER_BLOCK * len(union)
        if n != want:
            problems.append(
                f"resume adapter holds {n} keys, expected {want} for the union"
            )
            ok = False
    else:
        print("[dpo] no --resume-adapter-file given: the strict=False key check "
              "(the whole reason this arm converts the union) was SKIPPED")

    del model

    if not skip_control:
        from mlx_lm.utils import load
        control, _ = load(model_path)
        control.freeze()
        stock(control, len(picks), lora_params)          # STOCK trailing slice
        crep = S.collect_lora_report(control)
        S._print_report("control(stock trailing)", crep)
        expected_trailing = list(range(C.DEFAULT_NUM_BLOCKS - len(picks),
                                       C.DEFAULT_NUM_BLOCKS))
        if crep["layer_ids"] != expected_trailing:
            problems.append(
                f"stock control: got {crep['layer_ids']}, expected "
                f"{expected_trailing} -- the harness is not measuring placement"
            )
        ctrain = int(sum(v.size for _, v in tree_flatten(control.trainable_parameters())))
        if ctrain != n_train:
            problems.append(
                f"control trainable {ctrain} != cptv2-dpo-spectrum {n_train} -- "
                f"this arm's budget differs from the incumbent DPO recipe's"
            )
        del control
        ok = not problems

    for p in problems:
        print(f"  !! {p}")
    print("VERIFY:", "PASS" if ok else f"FAIL -- {len(problems)} problem(s)")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    layers_path = (S._pop_flag(argv, "--spectrum-layers")
                   or os.environ.get("SPECTRUM_LAYERS")
                   or DEFAULT_LAYERS_PATH)
    cpt_adapter = (S._pop_flag(argv, "--cpt-adapter")
                   or os.environ.get("CPT_V2_ADAPTER")
                   or str(CPT_V2_ADAPTER))

    if "--verify-patches" in argv:
        argv.remove("--verify-patches")
        return _run_verify_patches(layers_path, cpt_adapter)

    if "--verify-resume" in argv:
        i = argv.index("--verify-resume")
        resume = argv[i + 1]
        return _run_verify_resume(resume, layers_path, cpt_adapter)

    if "--verify-tokenization" in argv:
        import dpo_fixed_train as F
        model = argv[argv.index("--model") + 1]
        data = argv[argv.index("--data") + 1]
        F._assert_upstream_still_buggy()
        return F._verify_tokenization(model, data)

    if "--verify-layers" in argv:
        argv.remove("--verify-layers")
        ap = argparse.ArgumentParser(prog="cptv2_dpo_spectrum_train.py --verify-layers")
        ap.add_argument("--model", default="models/qwen-q4")
        ap.add_argument("--rank", type=int, default=16)
        ap.add_argument("--scale", type=float, default=2.0)
        ap.add_argument("--dropout", type=float, default=0.05)
        ap.add_argument("--resume-adapter-file", default=None)
        ap.add_argument("--skip-control", action="store_true")
        a = ap.parse_args(argv)
        return _run_verify_layers(
            a.model, layers_path,
            {"rank": a.rank, "scale": a.scale, "dropout": a.dropout},
            cpt_adapter, a.resume_adapter_file, a.skip_control,
        )

    picks = S.load_layer_ids(layers_path)
    base = C.assert_cpt_v2_shape(cpt_adapter)
    apply_patches(picks, base)
    rep = D.assert_patches_active(picks)
    print(f"[cptv2_dpo_spectrum_train] picks={picks} base={base}\n"
          f"  converting union {rep['union_layer_ids']}, "
          f"freezing {rep['frozen_layer_ids']} (CPT-v2 delta, no gradients)\n"
          f"  patched: {', '.join(rep['layer_sites'])}\n"
          f"  DPODataset patched: add_generation_prompt=False (no spurious tag)",
          flush=True)

    from mlx_lm_lora.train import main as _upstream_main

    sys.argv = [sys.argv[0]] + argv
    _upstream_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
