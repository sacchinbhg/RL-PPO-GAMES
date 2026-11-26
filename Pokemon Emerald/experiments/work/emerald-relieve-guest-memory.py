"""Stop explicitly named, checkpoint-backed auxiliary jobs after a guest memory stall."""
import json
import os
from pathlib import Path
import signal
import time

base=Path(__file__).resolve().parents[1]
root=base/"outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
out=root/"controls/overnight-resource-recovery"

def inspect():
    rows=[]
    for p in Path("/proc").glob("[0-9]*"):
        try:
            argv=(p/"cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
            status=(p/"status").read_text()
            ppid=int(next(s.split()[1] for s in status.splitlines() if s.startswith("PPid:")))
            rss=next((int(s.split()[1]) for s in status.splitlines() if s.startswith("VmRSS:")),0)
        except (ValueError,UnicodeDecodeError,OSError,StopIteration):
            continue
        rows.append((int(p.name),ppid,argv,rss))
    return rows

def run():
    out.mkdir(parents=True,exist_ok=True)
    if (out/"relief.json").exists():
        return
    rows=inspect()
    selected=[]
    for pid,ppid,argv,rss in rows:
        names={Path(a).name for a in argv}
        training=any(a.endswith("/seed-71299-h15-staggered-screen-collection/train") for a in argv)
        probe="probe-emerald-reward-model.py" in names and "--adapted" in argv
        auxiliary="run-emerald-heldout-earned.sh" in names and len(argv)==2
        if training and "upstream_emerald.py" in names or probe or auxiliary:
            selected.append(pid)
    victims=set(selected)
    for _ in range(4):
        victims.update(pid for pid,ppid,argv,rss in rows if ppid in victims)
    (out/"attempt.json").write_text(json.dumps(dict(time=time.time(),selected=selected,victims=sorted(victims),
            pressure=Path("/proc/pressure/memory").read_text(),mem=Path("/proc/meminfo").read_text().splitlines()[:7]),indent=2))
    for pid in victims:
        try:
            os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:
            pass
    (out/"relief.json").write_text(json.dumps(dict(time=time.time(),selected=selected,
         stopped=[dict(pid=pid,command=argv,rss_kb=rss) for pid,ppid,argv,rss in rows if pid in victims]),indent=2))

if __name__=="__main__":
    run()
