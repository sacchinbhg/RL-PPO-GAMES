"""Give the live frontier pilot CPU priority while all registered comparisons continue."""
import json
import os
from pathlib import Path
import time

base=Path(__file__).resolve().parents[1]
root=base/"outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
pilot=root/"paired-self-earned-frontiers/systematic-reward-frontiers"
dest=root/"controls/reward-frontier-expansion/cpu-priority.json"

def args(pid):
    try:
        return (Path("/proc")/str(pid)/"cmdline").read_bytes().decode().strip("\0").split("\0")
    except (OSError,UnicodeError):
        return []

def change(pid,priority):
    ids=[pid]
    children=Path("/proc")/str(pid)/"task"/str(pid)/"children"
    for child in map(int,children.read_text().split()):
        if any("multiprocessing." in a for a in args(child)):
            ids.append(child)
    updates=[]
    for n in ids:
        old=os.getpriority(os.PRIO_PROCESS,n)
        os.setpriority(os.PRIO_PROCESS,n,priority)
        updates.append(dict(pid=n,before=old,after=os.getpriority(os.PRIO_PROCESS,n)))
    return updates

if dest.exists():
    raise SystemExit(0)
jobs={}
for proc in Path("/proc").glob("[0-9]*"):
    pid=int(proc.name)
    argv=args(pid)
    if len(argv)<3 or Path(argv[0]).name!="python":
        continue
    if Path(argv[2]).name=="run-emerald-option-comparison.py" and argv[3:]==["--motion","--landmarks","64"]:
        jobs["older_motion_sample"]=(pid,12)
    if Path(argv[2]).name=="explore_reward_frontiers.py" and "--output" in argv and argv[argv.index("--output")+1]==str(pilot):
        jobs["frontier_pilot"]=(pid,0)
if set(jobs)!={"older_motion_sample","frontier_pilot"}:
    raise SystemExit("Live processes did not match both registered experiments")
result={label:change(*job) for label,job in jobs.items()}
dest.write_text(json.dumps(dict(time=time.time(),updates=result),indent=2))
