"""Keep fixed simulator and seed budgets while testing a visit reward shared across training branches."""
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
from reward_branch_search import write_json
root=project/"runs/pixels-only-01/official-wsl"
family=root/"paired-self-earned-frontiers"
out=root/"controls/shared-visit-reward"
control=family/"learned-visual-motion-landmarks-128-systematic-options-reward-landmarks-recent-32"
treatment=control.with_name(control.name+"-shared-visit-reward")
read=lambda p:json.loads(p.read_text())

def main():
    out.mkdir(parents=True,exist_ok=True)
    plan=dict(control=str(control),treatment=str(treatment),new_buttons_per_arm=262144,frames_per_button=24,
        seed=465721,initial_training_tape="e30822c294f0d5469126f3e250c9ad4ac83b232534b37d6f54b24b56078e37c5",
        change="a previously visited cell pays 0.02 only on its first discovery across all training branches",
        fixed="original training actions, visual motion model, action options, 128 landmark slots, 32 recent slots, seed, simulator budget",
        primary="furthest verified story event and new areas, followed by a separate frozen policy test if progress advances",
        scoring="independent game reports retain ordinary per-game scoring in both arms",evaluation_inputs_used=False)
    if (out/"plan.json").exists() and read(out/"plan.json")!=plan:
        raise ValueError("An incompatible registration already exists")
    write_json(out/"plan.json",plan)
    while not (out/"verified.json").exists():
        time.sleep(10)
    checked=read(out/"verified.json")
    assert checked["all_policy_frames_identical"] and checked["duplicated_location_payments_removed"]>0
    while int(next(row.split()[1] for row in Path("/proc/meminfo").read_text().splitlines() if row.startswith("MemAvailable:")))<1650000:
        time.sleep(20)
    write_json(out/"queue.json",dict(time=time.time(),status="starting fixed-budget treatment",calendar_start=time.time()))
    with (out/"training.log").open("a") as log:
        subprocess.run([sys.executable,"-u",str(base/"work/run-emerald-option-comparison.py"),"--motion","--landmarks","128",
            "--systematic-options","--reward-landmarks","--recent-landmarks","32","--shared-visit-reward"],check=True,stdout=log,stderr=subprocess.STDOUT)
    while not (control/"completed.json").exists():
        time.sleep(30)
    result={}
    for arm in (control,treatment):
        origin=read(arm/"branch-origin.json")
        complete=read(arm/"completed.json")
        work=complete["status"]["work"]-origin["inherited_training_buttons"]
        assert origin["parent_record"]["tape_sha256"]==plan["initial_training_tape"] and origin["seed"]==plan["seed"]
        assert work==plan["new_buttons_per_arm"] or complete["best"]["terminal"]
        result[arm.name]=dict(real_training_buttons=work,origin=origin["parent_record"]["tape_sha256"],
            best_persisted=complete["best"],maps_seen_in_training=sorted(p.name for p in (arm/"first-maps").iterdir() if (p/"game.json").is_file()))
    write_json(out/"comparison.json",result)

if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),
        EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
    out.mkdir(parents=True,exist_ok=True)
    with (out/"queue.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
