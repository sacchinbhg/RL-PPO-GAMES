"""Retire the slower reporter and forward independently verified improvements to the footage watcher."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from train_earned_dreamer import write_json
root=project/"runs/pixels-only-01/official-wsl/controls/trainer-contacts/frontier-reviews"
new=root/"all-branches"
read=lambda p:json.loads(p.read_text())


def matching(name):
    found=[]
    for file in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args=file.read_bytes().decode().strip("\0").split("\0")
        except (OSError,UnicodeError):
            continue
        if args and "python" in Path(args[0]).name and name in [Path(a).name for a in args[1:]]:
            found.append(int(file.parent.name))
    return found


if __name__=="__main__":
    with (new/"reporter.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert (new/"ready.json").exists() and len(matching("review-emerald-all-branches.py"))==1
        prior=matching("monitor-emerald-frontier-approaches.py")
        for pid in prior:
            os.kill(pid,signal.SIGTERM)
        write_json(new/"reporter.json",dict(old_reporting_pids=prior,replacement_reporting_only=True,time=time.time()))
        while True:
            if (new/"closest.json").exists():
                candidate=read(new/"closest.json")
                if candidate["distance"]<read(root/"closest.json")["distance"]:
                    check=read(Path(candidate["verification"]))
                    tape=Path(candidate["audit"]).with_suffix(".bin").read_bytes()
                    assert check["in_game_restores"]==0 and hashlib.sha256(tape).hexdigest()==check["input_sha256"]
                    assert (candidate["distance"],candidate["decision"])==(check["nearest_template_manhattan_tiles"],check["first_button_at_nearest"])
                    write_json(root/"closest.json",candidate)
            time.sleep(20)
