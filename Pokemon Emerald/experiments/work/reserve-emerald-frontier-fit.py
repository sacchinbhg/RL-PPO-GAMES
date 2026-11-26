"""Let the current game finish, then reserve its guest RAM for the higher-return training batch."""
import fcntl
import json
import os
from pathlib import Path
import signal
import time

base=Path(__file__).resolve().parents[1]
root=base/"outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
out=root/"controls/overnight-resource-recovery/frontier-fit-slot.json"
shared=root/"earned-experience-comparison/shared-opening/fresh-65536/result.json"
trained=root/"earned-frontier-curriculum/stage-003847-01f5c6813216/official-fit/result.json"

def command(proc):
    try:
        return (proc/"cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
    except (OSError,UnicodeError):
        return []

def record(status,**facts):
    draft=out.with_suffix(".tmp")
    draft.write_text(json.dumps(dict(time=time.time(),status=status,**facts),indent=2))
    draft.replace(out)

def main():
    if shared.exists() or trained.exists():
        record("slot already passed")
        return
    argv=["bash","work/run-emerald-diverse-experience.sh"]
    found=[p for p in Path("/proc").glob("[0-9]*") if command(p)==argv]
    if len(found)!=1:
        raise RuntimeError("Expected exactly one registered comparison supervisor")
    parent,=found
    pid=int(parent.name)
    children=(parent/"task"/parent.name/"children").read_text().split()
    jobs=[command(Path("/proc")/child) for child in children]
    if not any(any(a.endswith("/play_frozen_dreamer.py") for a in args) and
               any(a.endswith("/shared-opening/fresh-65536") for a in args) for args in jobs):
        raise RuntimeError("Supervisor is not waiting for the expected live game")
    start=time.monotonic()
    os.kill(pid,signal.SIGSTOP)
    try:
        while not shared.exists() and time.monotonic()-start<2400:
            record("current four games running; next evaluation held",supervisor_pid=pid,child_pids=children)
            time.sleep(30)
        if shared.exists():
            deadline=time.monotonic()+2700
            while not trained.exists() and time.monotonic()<deadline:
                record("current four games complete; capacity reserved for the frontier fit",supervisor_pid=pid)
                time.sleep(30)
    finally:
        if command(parent)==argv:
            os.kill(pid,signal.SIGCONT)
        record("comparison supervisor released",supervisor_pid=pid,shared_complete=shared.exists(),frontier_fit_complete=trained.exists())

if __name__=="__main__":
    def finish(_signal,_frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,finish)
    with out.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
