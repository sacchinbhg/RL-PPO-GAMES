"""Test whether keeping camera-scrolled screens helps cross gaps between scalar rewards."""
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
family=root/"paired-self-earned-frontiers"
out=root/"controls/unrewarded-motion-landmarks"
parent=root/"controls/reward-frontier-expansion"
probe=root/"controls/visual-motion/route-training.json"
read=lambda p:json.loads(p.read_text())


def run_arm(source,record,dest,budget,capacity,scroll):
    with (out/"experiment.log").open("a") as log:
        subprocess.run([sys.executable,"-u",str(project/"explore_reward_frontiers.py"),
                        "--source",str(source),"--record",str(record),"--output",str(dest),"--motion",str(probe),
                        "--steps",str(budget),"--capacity",str(capacity),"--scroll-slots",str(scroll),
                        "--reward-sweep","--seed","173741"],stdout=log,stderr=subprocess.STDOUT,check=True)
    return read(dest/"completed.json")


def run():
    assert read(out/"measured.json")["qualifies_for_comparison"]
    plan=read(parent/"plan.json")
    baseline=family/"sweep-reward-frontiers"
    dest=family/"sweep-unrewarded-scroll"
    baseline_settings=read(baseline/"settings.json")
    assert baseline_settings["seed"]==plan["seed"]==173741
    assert baseline_settings["source_sha256"]==plan["selected"]["tape_sha256"]
    assert baseline_settings["reward_slots"]==plan["capacity"]==768
    write_json(out/"live-registration.json",dict(seed=173741,source=plan["selected"]["tape_sha256"],
        comparison="reward sweep on the same independently earned history and same 768-slot maximum",
        treatment="640 rewarded saves plus up to 128 unrewarded, visually distinct learned camera movements",
        baseline="768 rewarded saves",steps=131072,frame_repeat=24,
        outcomes=["game progression","additional coarse cells at 16k, 32k, 65k, 131k"],
        scheduling="after the second ordinary worker completes, subject to available guest memory"))
    short=root/"controls/outcome-collection-smoke"
    smoke=out/"smoke"
    check=run_arm(short,short/"best.json",smoke,2048,16,4)
    stat=check["status"]
    assert stat["new_buttons"]==2048 and stat["active_slots"]<=16
    if stat["starts_by_kind"].get("scroll",0)==0:
        write_json(out/"smoke-stage-reset.json",dict(first_smoke=stat,
            observation="the short opening admitted scroll saves before its last large scalar reward, then retained only a rewarded save",
            followup="use the already measured self-earned 36,058-button history with confirmed unpaid movements after its last large reward"))
        short=family/"visual-slots-0"
        assert read(short/"best.json")["tape_sha256"]==read(out/"measured.json")["training_source"]
        smoke=out/"smoke-after-unrewarded-movement"
        check=run_arm(short,short/"best.json",smoke,2048,16,4)
        stat=check["status"]
        assert stat["new_buttons"]==2048 and stat["active_slots"]<=16
    assert stat["starts_by_kind"].get("scroll",0)>0
    with (out/"experiment.log").open("a") as log:
        best=check["best"]
        subprocess.run([sys.executable,str(project/"pixel_replay.py"),str(smoke/best["audit"]/"game.json"),
                        str(out/"smoke-original"),"--decision",str(best["length"])],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    proof=read(out/"smoke-original/verification.json")
    assert proof["input_sha256"]==best["tape_sha256"] and proof["restores_during_game"]==0
    while not (family/"systematic-reward-frontiers-seed-173742/completed.json").exists():
        time.sleep(30)
    while int(next(s.split()[1] for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:")))<1950000:
        time.sleep(30)
    check=run_arm(plan["source"],parent/"selected-record.json",dest,131072,768,128)
    assert check["status"]["active_slots"]<=768 and check["status"]["starts_by_kind"].get("scroll",0)>0
    write_json(out/"result.json",check)


if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),
                      EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
    with (out/"live.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
