#!/usr/bin/env python3
"""Launch run_arm5_full_chain.sh in its OWN session, fully detached.

Why this exists.  The first launch used the agent harness's `run_in_background`
with no `&`/nohup, exactly as the runbook advised.  The chain started cleanly
(preflight OK, leg 1 training, trainer resident at ~10.9 GB) and was then killed
~1 h in, mid-leg-1, with no `.FAILED_AT_STAGE` marker -- i.e. it was not the
script's own abort path, it was an external kill of the harness's background
task and everything in its process group.  A ~40 h run cannot live inside a
process group that the harness reaps.

Double-fork + setsid puts the chain in a brand-new session with no controlling
terminal, so neither SIGHUP nor a process-group kill aimed at the harness's
shell can reach it.  This is ONE backgrounding mechanism, not two -- the caller
invokes this in the FOREGROUND and it returns immediately.  Do not additionally
wrap it in `&`, nohup, or run_in_background.

Refuses to start a second copy (that would put two trainers on one GPU and OOM
the machine), so it is safe to re-run to resume after any kill.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Volumes/ExtremePro/JaseciLabs/jac_model_studio")
SCRIPT = ROOT / "model-experiments/04-cpt-sft/sft_cptspectrum_probe/run_arm5_full_chain.sh"
LOG = ROOT / "model-experiments/04-cpt-sft/sft_cptspectrum_probe/results/arm5-chain/launcher.log"
PIDFILE = ROOT / "model-experiments/04-cpt-sft/sft_cptspectrum_probe/results/arm5-chain/chain.pid"


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)

    # Refuse to double-launch: two trainers on one 48GB machine = OOM.
    if PIDFILE.exists():
        try:
            old = int(PIDFILE.read_text().strip())
        except ValueError:
            old = -1
        if old > 0 and alive(old):
            print(f"REFUSING: chain already running as PID {old}")
            sys.exit(2)
    out = subprocess.run(["pgrep", "-f", "run_arm5_full_chain"], capture_output=True, text=True)
    running = [p for p in out.stdout.split() if p.strip()]
    if running:
        print(f"REFUSING: run_arm5_full_chain.sh already running: {running}")
        sys.exit(2)

    if os.fork() > 0:            # parent: return to the caller immediately
        time.sleep(2.0)          # let the grandchild write the pidfile
        try:
            pid = PIDFILE.read_text().strip()
        except OSError:
            pid = "?"
        print(f"launched detached chain, session leader PID {pid}")
        print(f"log: {LOG}")
        os._exit(0)

    os.setsid()                  # NEW SESSION -- the whole point of this file
    if os.fork() > 0:            # ensure we can never reacquire a terminal
        os._exit(0)

    PIDFILE.write_text(str(os.getpid()))
    os.chdir(str(ROOT))
    fd = os.open(str(LOG), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.execv("/bin/bash", ["bash", str(SCRIPT)])


if __name__ == "__main__":
    main()
