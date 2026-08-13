#!/usr/bin/env python3
"""Halt the Arm 5 chain cleanly at the CPT-snapshot boundary (scope change).

New scope: run CPT (legs 1-12) and the Stage 2 snapshot into
cpt-v2-spectrum-final/, then STOP.  Stages 3-6 (SFT / SFT eval / DPO / DPO eval)
are held pending a further go-ahead.

Editing run_arm5_full_chain.sh cannot achieve this: a concurrent session
unlinked that file mid-run (commit ab73938) and the running bash still reads the
old, now-unlinked inode, so it will never see an edit at the path.  The stop has
to come from outside the process.

Mechanism: poll for the `.2-snapshot.done` marker, which `mark_done 2-snapshot`
writes only AFTER the snapshot has been copied, config-rewritten, verified
non-zero and sha256-matched against its source.  The instant it appears, SIGTERM
the chain's process group.  At that moment Stage 3 has either not started or is
at most a few seconds into `--verify-layers`, which only reads the base model and
writes a transcript -- nothing precious is mid-write.

Deliberately does NOT kill during Stage 1 or Stage 2: a kill mid-leg would cost
~2.4h of CPT, and a kill mid-snapshot could leave a truncated adapter.
"""
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path("/Volumes/ExtremePro/JaseciLabs/jac_model_studio")
STATE = ROOT / "model-experiments/04-cpt-sft/sft_cptspectrum_probe/results/arm5-chain"
MARKER = STATE / ".2-snapshot.done"
LOG = STATE / "stop_watcher.log"
CHAIN_PGID = 16861
CHAIN_PID = 16862


def say(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
    with open(LOG, "a") as f:
        f.write(line)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    if os.fork() > 0:
        os._exit(0)
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    say(f"stop-watcher armed: will SIGTERM pgid {CHAIN_PGID} when {MARKER.name} appears")

    while True:
        if not alive(CHAIN_PID):
            say("chain already exited on its own; nothing to stop")
            return
        if MARKER.exists():
            say("snapshot marker present -- Stage 2 complete and verified. Stopping chain.")
            try:
                os.killpg(CHAIN_PGID, signal.SIGTERM)
            except ProcessLookupError:
                say("process group already gone")
            for _ in range(30):
                time.sleep(1)
                if not alive(CHAIN_PID):
                    break
            if alive(CHAIN_PID):
                say("SIGTERM did not take; escalating to SIGKILL")
                try:
                    os.killpg(CHAIN_PGID, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                time.sleep(3)
            say(f"chain stopped at the CPT-snapshot boundary (alive={alive(CHAIN_PID)})")
            (STATE / ".STOPPED_AFTER_SNAPSHOT").write_text(
                "Chain intentionally halted after Stage 2 (CPT snapshot) per scope change.\n"
                "Stages 3-6 (SFT/SFT-eval/DPO/DPO-eval) NOT run, pending go-ahead.\n"
            )
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
