"""Train a paired official Dreamer adaptation with the uncapped visit component, then play from boot."""
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
capped=root/"earned-official-updates"
out=capped/"uncapped-training"
upstream=Path("/opt/emerald-rl/source")
parent=root/"seed-71299-h15/eval-65536"

def read(path):
    return json.loads(path.read_text())

def available():
    return int(next(s.split()[1] for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:")))

def call(script,*args):
    with (out/"job.log").open("a") as log:
        subprocess.run([sys.executable,"-u",str(project/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True)

def wait(reason):
    write_json(out/"queue.json",dict(time=time.time(),waiting_for=reason,guest_available_kib=available()))
    time.sleep(60)

def run():
    reference=read(capped/"adaptation/result.json")
    check=read(root/"controls/earned-recurrent-stream/uncapped-adaptation.json")
    assert [r["tape_sha256"] for r in check["episodes"]]==reference["training_tapes"]
    dirs=(root/"reward-only-training-forks/verified-training-tapes",
          root/"paired-self-earned-frontiers/visual-slots-0/verified-training-tapes")
    data=[d/sha for d,sha in zip(dirs,reference["training_tapes"],strict=True)]
    fit=out/"adaptation"
    while not (fit/"result.json").exists():
        if not (root/"earned-experience-comparison/shared-opening/fresh-65536/result.json").exists():
            wait("shared-opening frozen games to finish")
            continue
        primary=root/"earned-frontier-curriculum/stage-003847-01f5c6813216"
        if (primary/"selection.json").exists() and not (primary/"official-fit/result.json").exists():
            wait("the selected higher-return trajectory's Dreamer optimizer")
            continue
        if available()<3000000:
            wait("3000000 KiB of available guest memory")
            continue
        with (root/"controls/large-optimizer.lock").open("a") as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                wait("existing optimizer to finish")
                continue
            if available()<3000000:
                continue
            write_json(out/"queue.json",dict(time=time.time(),status="training"))
            call("train_earned_dreamer.py","--dataset",data[0],"--dataset",data[1],"--source-run",parent,
                 "--upstream",upstream,"--output",fit,"--updates",reference["optimizer_updates"],
                 "--uncapped-training-exploration")
    candidate=read(fit/"result.json")
    for key in ("training_tapes","optimizer_updates","upstream_revision","parent_sha256","observations_trained","imagination_length"):
        if candidate[key]!=reference[key]:
            raise ValueError(f"Unmatched adaptation: {key}")
    while available()<1700000:
        wait("1700000 KiB before frozen evaluation")
    trials=sorted(out.glob("fresh-*"))
    trial=next((p for p in trials if (p/"result.json").exists()),None)
    if trial is None:
        trial=out/f"fresh-{len(trials)+1:03d}"
        write_json(out/"queue.json",dict(time=time.time(),status="fresh evaluation",trial=str(trial)))
        call("play_frozen_dreamer.py",fit/"frozen-policy",parent,trial,"--upstream",upstream,"--limit",65536)
    results={}
    for name,arm,model in (("capped",capped/"fresh-65536",reference),("uncapped",trial,candidate)):
        result=read(arm/"result.json")
        if (result["seed"],result["games"],result["in_game_restores"],result["weight_sha256"])!=(271340,4,0,model["trained_sha256"]):
            raise ValueError("Fresh evaluation identity mismatch")
        workers=[]
        for i in range(4):
            audits=list((arm/"games"/f"env-{i}").glob("step-*/*episode-00001.json"))
            final=max((read(p) for p in audits),key=lambda a:a["steps"])
            assert final["steps"]==65536 or "stone_badge" in final["events"]
            workers.append(dict(steps=final["steps"],events=final["events"],maps=final.get("maps_first_reached",{})))
        results[name]=dict(weight_sha256=model["trained_sha256"],workers=workers)
    write_json(out/"comparison.json",dict(games_per_arm=4,button_limit=65536,fresh_seed=271340,
               optimizer_updates=reference["optimizer_updates"],results=results))
    write_json(out/"queue.json",dict(time=time.time(),status="completed"))

if __name__=="__main__":
    os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",JAX_PLATFORMS="cuda",XLA_PYTHON_CLIENT_PREALLOCATE="false",
                      EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
    out.mkdir(exist_ok=True)
    with (out/"job.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
