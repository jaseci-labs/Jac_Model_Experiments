"""Task 13 operational driver: runs the CPT-v2 epoch loop end-to-end, leg 1
through the stop-loss gate or ceiling 12 (design.md section 4.3). One
subprocess per leg (clean memory state each time -- mlx doesn't reliably
release all GPU memory across in-process model reloads) and one per
CF-check. Resumable by construction: restarting this script re-derives the
current leg from the adapter directory's on-disk checkpoints (same
convention as cpt.sv.jac's _cpt_leg_resume_point), so a kill/crash/sleep
mid-run loses at most the in-flight leg, never earlier progress.

Writes structured per-leg results to training_state.json (source of truth
for resuming and for the final verdict) and a mechanical log line to
leg_reviews.md per leg. The narrative/qualitative Sonnet review design.md
section 5 calls for is added separately and interactively (this script only
logs objective data -- loss, CF score, timing, gate decision)."""
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_DIR = ROOT / "03-new" / "adapters" / "cpt-v2"
RESULTS_DIR = ROOT / "03-new" / "results" / "cpt-v2"
STATE_FILE = RESULTS_DIR / "training_state.json"
REVIEWS_FILE = RESULTS_DIR / "leg_reviews.md"
MANIFEST = ROOT / "03-new" / "dataset" / "cpt-v2" / "manifest.json"
CONFIG_DIR = ROOT / "03-new" / "cpt_train"
PY = str(ROOT / ".venv" / "bin" / "python3")

FLOOR, CEILING = 6, 12

sys.path.insert(0, str(CONFIG_DIR))
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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
    cmd = [PY, str(CONFIG_DIR / "run_cpt_leg.py"),
           "--config", str(config), "--adapter-path", str(ADAPTER_DIR),
           "--iters", str(windows), "--done-steps", str(done_steps)]
    if resume_adapter:
        cmd += ["--resume-adapter-file", resume_adapter]
    if resume_optimizer:
        cmd += ["--resume-optimizer-file", resume_optimizer]
    print(f"=== leg {leg}/{CEILING}: {' '.join(cmd)} ===", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    print(r.stdout[-4000:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-4000:], flush=True)
        raise RuntimeError(f"leg {leg} training failed, exit {r.returncode}")
    train_loss, val_loss, lr = parse_last_losses(r.stdout)
    return {"duration_s": round(dt, 1), "train_loss": train_loss, "val_loss": val_loss, "final_lr": lr}


def run_cf_check():
    cmd = [PY, str(CONFIG_DIR / "cf_check" / "run_leg_cf_check.py"), str(ADAPTER_DIR)]
    print(f"=== CF-check: {' '.join(cmd)} ===", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
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
    checkpoint back over it so downstream CF-check/fuse see the right
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
        f.write("- _[objective log only -- qualitative review pending, see design.md section 5]_\n")


def main():
    windows = windows_per_epoch()
    state = load_state()
    done_steps, resume_adapter, resume_optimizer = resume_point()
    leg = done_steps // windows + 1
    print(f"windows/epoch={windows}, resuming at done_steps={done_steps} -> leg {leg}", flush=True)

    while leg <= CEILING:
        result = run_leg(leg, windows, done_steps, resume_adapter, resume_optimizer)
        passed, total = run_cf_check()
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
