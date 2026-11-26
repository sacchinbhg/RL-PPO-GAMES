"""Evaluate the collected motion model and record its predeclared criterion for a fresh search trial."""
import fcntl
import json
from pathlib import Path
import sys
import time

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from fit_local_motion import run
from train_earned_dreamer import write_json
out=project/"runs/pixels-only-01/official-wsl/controls/local-screen-motion"
if __name__=="__main__":
    out.mkdir(exist_ok=True)
    with (out/"score.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(out/"trial-criterion.json",dict(minimum_sensitive_positions=25,required_absolute_gain=.025,
            comparator="strongest of uniform movement, most successful training direction, and whole-screen retrieval",
            purpose="screen scroll is a proxy for predicted movement; real exploration must still earn new rewards"))
        while not all((out/k/"completed.json").exists() for k in ("train","test")):
            time.sleep(30)
        run(out/"train",out/"test",out/"fitted")
        result=json.loads((out/"fitted/heldout.json").read_text())
        write_json(out/"decision.json",dict(qualifies_for_live_comparison=result["action_sensitive_positions"]>=25 and
            result["local_improvement_over_strongest_baseline"]>=.025,measured=result))
