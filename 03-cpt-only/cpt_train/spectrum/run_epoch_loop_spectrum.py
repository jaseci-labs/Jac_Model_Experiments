"""Spectrum-arm fork of run_epoch_loop.py (cpt-spectrum-plan.md §2.4). Runs the
CPT-on-Spectrum epoch loop end-to-end, leg 1 through the stop-loss gate or
ceiling 12 -- the exact same floor/ceiling stop-loss semantics as CPT-v2
(epoch_loop_gate.py, unchanged, imported not copied), on Spectrum's 16 picked
layers instead of the stock trailing-16.

A FORK, not a parameterized reuse of run_epoch_loop.py: that file has no
arguments -- every path is a module-level constant built from a stale
`"03-new"` literal, and the leg script name is hardcoded. Parameterising it
would mean rewriting CPT-v2's own historical driver. This file changes
exactly the paths/leg-script/env needed for the Spectrum arm; every other
line -- windows_per_epoch, resume_point, load_state/save_state,
parse_last_losses, revert_to_previous, append_review, FLOOR/CEILING, and the
whole main() control flow -- is kept byte-for-byte, plus one addition: the
adapter_config_fix.py rewrite between run_leg() and run_cf_check(), which
CPT-v2's own loop never needed (see cpt-spectrum-plan.md §2.5 for why it's
mandatory here -- without it, the CF-check silently scores a partially-loaded
model)."""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SPEC_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = REPO_ROOT / "model-experiments/03-cpt-only/adapters/cpt-v2-spectrum"
RESULTS_DIR = REPO_ROOT / "model-experiments/03-cpt-only/results/cpt-v2-spectrum"
JSON_DIR = RESULTS_DIR / "json"
LOGS_DIR = RESULTS_DIR / "logs"
REVIEWS_FILE = RESULTS_DIR / "leg_reviews.md"
MANIFEST = REPO_ROOT / "model-experiments/03-cpt-only/dataset/cpt-v2/manifest.json"
CONFIG_DIR = SPEC_DIR / "configs"
STATE_FILE = JSON_DIR / "training_state.json"
PY = str(REPO_ROOT / ".venv" / "bin" / "python3")
LEG_SCRIPT = SPEC_DIR / "run_cpt_leg_spectrum.py"
LAYERS = CONFIG_DIR / "spectrum_layers.json"
CFG_FIX = REPO_ROOT / "model-experiments/04-cpt-sft/sft_fresh_probe/spectrum/adapter_config_fix.py"
REWRITE_LOG = RESULTS_DIR / "adapter_config_rewrite.txt"

FLOOR, CEILING = 6, 12

sys.path.insert(0, str(SPEC_DIR.parent))  # cpt_train/, for epoch_loop_gate
from epoch_loop_gate import decide_next_action


def windows_per_epoch() -> int:
    m = json.loads(MANIFEST.read_text())
    return int(m["packed"]["train"])


def resume_point():
    """(done_steps, resume_adapter_file, resume_optimizer_file)."""
    ckpts = sorted(ADAPTER_DIR.glob("*_adapters.safetensors"))
    if not ckpts:
        return 0, None, None
    latest = ckpts[-1]
    steps = int(latest.name.split("_")[0])
    opt = ADAPTER_DIR / latest.name.replace("_adapters.safetensors", "_optimizer.safetensors")
    return steps, str(latest), str(opt) if opt.exists() else None


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"legs": [], "status": "running"}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def parse_last_losses(log_text: str):
    train_losses = re.findall(r"Train loss ([\d.]+)", log_text)
    val_losses = re.findall(r"Val loss ([\d.]+)", log_text)
    lrs = re.findall(r"Learning Rate ([\d.eE+-]+)", log_text)
    return (float(train_losses[-1]) if train_losses else None,
            float(val_losses[-1]) if val_losses else None,
            float(lrs[-1]) if lrs else None)


def run_leg(leg: int, windows: int, done_steps: int, resume_adapter, resume_optimizer) -> dict:
    config = CONFIG_DIR / f"config_v2_leg{leg}.yaml"
    cmd = [PY, str(LEG_SCRIPT),
           "--config", str(config), "--adapter-path", str(ADAPTER_DIR),
           "--iters", str(windows), "--done-steps", str(done_steps),
           "--spectrum-layers", str(LAYERS)]
    if resume_adapter:
        cmd += ["--resume-adapter-file", resume_adapter]
    if resume_optimizer:
        cmd += ["--resume-optimizer-file", resume_optimizer]
    print(f"=== leg {leg}/{CEILING}: {' '.join(cmd)} ===", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    dt = time.time() - t0
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / f"leg{leg}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    print(r.stdout[-4000:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-4000:], flush=True)
        raise RuntimeError(f"leg {leg} training failed, exit {r.returncode}")
    train_loss, val_loss, lr = parse_last_losses(r.stdout)
    return {"duration_s": round(dt, 1), "train_loss": train_loss, "val_loss": val_loss, "final_lr": lr}


def rewrite_adapter_config(leg: int):
    """cpt-spectrum-plan.md §2.5: load_adapters reads num_layers out of the
    adapter's own adapter_config.json, which run_cpt_leg.py writes as 16
    (the COUNT) every leg -- and rebuilds LoRA on the trailing slice {32..47}
    unless told otherwise, silently dropping Spectrum's non-trailing picks
    (0, 22, 23, 27, 30) via strict=False. Must run before every CF-check."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REWRITE_LOG, "a") as f:
        f.write(f"\n=== leg {leg} ===\n")
        r = subprocess.run(
            [PY, str(CFG_FIX), "--adapter-dir", str(ADAPTER_DIR), "--spectrum-layers", str(LAYERS)],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        f.write(r.stdout)
        f.write(r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"leg {leg} adapter_config_fix.py failed, exit {r.returncode}: {r.stderr[-2000:]}")


def run_cf_check(leg: int):
    cmd = [PY, str(SPEC_DIR.parent / "cf_check" / "run_leg_cf_check.py"), str(ADAPTER_DIR)]
    print(f"=== CF-check: {' '.join(cmd)} ===", flush=True)
    env = {**os.environ, "JAC_CF_RESULTS_DIR": str(JSON_DIR)}
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), env=env)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / f"cf{leg}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    print(r.stdout[-2000:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-2000:], flush=True)
        raise RuntimeError(f"CF-check failed, exit {r.returncode}")
    m = re.search(r"CF-check: (\d+)/(\d+)", r.stdout)
    if not m:
        raise RuntimeError(f"couldn't parse CF-check output: {r.stdout[-500:]}")
    return int(m.group(1)), int(m.group(2))


def revert_to_previous(prev_ckpt: str):
    """halt_keep_previous: run_cpt_leg.py's train() call always overwrites
    the shared rolling-latest adapters.safetensors with the JUST-COMPLETED
    (rejected) leg's weights. Copy the last ACCEPTED leg's numbered
    checkpoint back over it so downstream CF-check/eval see the right
    weights -- matches cf_check's documented "shared dir reflects most
    recently completed leg" contract."""
    shutil.copy(prev_ckpt, ADAPTER_DIR / "adapters.safetensors")
    print(f"reverted shared adapters.safetensors -> {prev_ckpt}", flush=True)


def append_review(leg: int, result: dict, cf_passed: bool, cf_score: str, decision: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REVIEWS_FILE, "a") as f:
        f.write(f"\n## Leg {leg}\n\n")
        f.write(f"- train loss (last): {result['train_loss']}, val loss (last): {result['val_loss']}, "
                f"final LR: {result['final_lr']}\n")
        f.write(f"- duration: {result['duration_s']}s\n")
        f.write(f"- CF-check: {cf_score} ({'PASS' if cf_passed else 'FAIL'})\n")
        f.write(f"- gate decision: {decision}\n")
        f.write("- _[objective log only]_\n")


def main():
    windows = windows_per_epoch()
    state = load_state()
    done_steps, resume_adapter, resume_optimizer = resume_point()
    leg = done_steps // windows + 1
    print(f"windows/epoch={windows}, resuming at done_steps={done_steps} -> leg {leg}", flush=True)

    while leg <= CEILING:
        result = run_leg(leg, windows, done_steps, resume_adapter, resume_optimizer)
        rewrite_adapter_config(leg)
        passed, total = run_cf_check(leg)
        cf_passed = passed == total
        decision = decide_next_action(leg, cf_passed, floor=FLOOR, ceiling=CEILING)

        state["legs"].append({
            "leg": leg, "done_steps_after": done_steps + windows, **result,
            "cf_passed": cf_passed, "cf_score": f"{passed}/{total}", "decision": decision,
        })
        state["status"] = decision
        save_state(state)
        append_review(leg, result, cf_passed, f"{passed}/{total}", decision)

        if decision == "halt_keep_previous":
            prev_ckpt = ADAPTER_DIR / f"{done_steps:07d}_adapters.safetensors"
            if prev_ckpt.exists():
                revert_to_previous(str(prev_ckpt))
            else:
                print(f"WARNING: halt_keep_previous at leg {leg} but no previous checkpoint "
                      "exists -- shouldn't happen, floor prevents halting before leg 6", flush=True)
            print(f"HALTED at leg {leg}: keeping leg {leg - 1}'s checkpoint (CF regression).", flush=True)
            break
        if decision == "halt_keep_this":
            print(f"HALTED at leg {leg}: keeping this leg's checkpoint.", flush=True)
            break

        done_steps += windows
        resume_adapter = str(ADAPTER_DIR / f"{done_steps:07d}_adapters.safetensors")
        resume_optimizer = str(ADAPTER_DIR / f"{done_steps:07d}_optimizer.safetensors")
        leg += 1

    print(f"=== epoch loop finished: status={state['status']} ===", flush=True)


if __name__ == "__main__":
    main()
