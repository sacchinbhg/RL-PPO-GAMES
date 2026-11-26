"""Run a fresh recovery trial, then verify the original control and pre-intervention buttons."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from train_earned_dreamer import write_json
root=project/"runs/pixels-only-01/official-wsl"
family=root/"earned-route-recovery"
baseline=root/"earned-route-extension/trial-197461"
read=lambda p:json.loads(p.read_text())

def available():
    return int(next(s.split()[1] for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:")))

def last(arm,i):
    p=max((arm/"games"/f"env-{i}").glob("step-*/*.json"),key=lambda a:int(a.parent.name[5:]))
    return read(p),p.with_suffix(".bin").read_bytes()

def same_prefix(a,b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))

def run():
    check=read(root/"controls/visual-route-recovery/verification.json")
    assert check["fixture_reaches_target"] and check["heldout"]["matched_advice"]>0
    started=time.monotonic()
    trials=sorted(family.glob("trial-*"))
    trial=next((p for p in trials if (p/"result.json").exists()),None)
    if trial is None:
        while True:
            mem=available()
            prior=(root/"earned-official-updates/uncapped-training/adaptation/result.json").exists()
            if (prior and mem>=1750000) or mem>=3900000 or (time.monotonic()-started>=2400 and mem>=2300000):
                break
            write_json(family/"queue.json",dict(time=time.time(),available_kib=mem,waiting_for="a free WSL policy slot"))
            time.sleep(30)
        trial=family/f"trial-197461-{len(trials)+1:03d}"
        write_json(family/"queue.json",dict(time=time.time(),playing=str(trial)))
        with (family/"run.log").open("a") as log:
            subprocess.run([sys.executable,"-u",str(project/"extend_earned_route.py"),"--model",str(root/"earned-screen-memory/model"),
                "--checkpoint",str(root/"earned-official-updates/adaptation/frozen-policy"),"--parent",str(root/"seed-71299-h15/eval-65536"),
                "--upstream","/opt/emerald-rl/source","--output",str(trial),"--limit","65536","--seed","197461","--patience","512",
                "--visual-route-recovery"],stdout=log,stderr=subprocess.STDOUT,check=True)
    original=read(baseline/"provenance.json")
    current=read(trial/"provenance.json")
    for key in ("weight_sha256","memory_sha256","memory_sources","seed","decisions_per_game","memory_patience","in_game_restores"):
        assert original[key]==current[key]
    assert read(trial/"handoff.json")==read(baseline/"handoff.json")
    changed=trial/"first-recovery-change.json"
    first=read(changed)["decision"] if changed.exists() else None
    baseline_game,before=last(baseline,0)
    recovery_game,after=last(trial,0)
    _,ordinary_uniform=last(baseline,1)
    _,recovery_uniform=last(trial,1)
    overlap=same_prefix(before,after)
    uniform_overlap=same_prefix(ordinary_uniform,recovery_uniform)
    expected=min(len(before),len(after)) if first is None else first-1
    comparison=dict(first_changed_decision=first,unchanged_dreamer_prefix=overlap,
           matched_dreamer_before_intervention=overlap==expected,matched_uniform_control=uniform_overlap==len(recovery_uniform),
           uniform_buttons=uniform_overlap,baseline=baseline_game,recovery=recovery_game,trial=str(trial),
           recovery_actions=read(trial/"result.json").get("recovery_actions",{}))
    write_json(family/"comparison.json",comparison)
    if not comparison["matched_dreamer_before_intervention"] or not comparison["matched_uniform_control"]:
        raise RuntimeError("The two runs diverged outside the registered intervention")
    write_json(family/"queue.json",dict(time=time.time(),status="matched comparison complete"))

if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so",
                      OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",JAX_PLATFORMS="cuda",XLA_PYTHON_CLIENT_PREALLOCATE="false")
    family.mkdir(exist_ok=True)
    with (family/"job.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
