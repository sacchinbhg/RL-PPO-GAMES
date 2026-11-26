"""Verify a small training fork first, then expand every rewarded discovery in the highest-return training history."""
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
out=root/"controls/reward-frontier-expansion"
destination=family/"systematic-reward-frontiers"
probe=root/"controls/visual-motion/route-training.json"
read=lambda p:json.loads(p.read_text())

def call(script,*args):
    with (out/"training.log").open("a") as log:
        subprocess.run([sys.executable,"-u",str(project/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True)

def main():
    if not (out/"plan.json").exists():
        eligible=[]
        for arm in [root/"reward-only-training-forks",*family.iterdir()]:
            if not (arm/"settings.json").is_file() or not (arm/"best.json").is_file() or arm==destination:
                continue
            cfg,best=read(arm/"settings.json"),read(arm/"best.json")
            if cfg.get("training_forks") and not cfg.get("receives_saves_from_other_runs") and best["training_only"]:
                eligible.append((arm,best))
        source,selected=max(eligible,key=lambda pair:(pair[1]["return_cents"],-pair[1]["length"]))
        write_json(out/"selected-record.json",selected)
        write_json(out/"plan.json",dict(source=str(source),selected=selected,seed=173741,steps=262144,capacity=768,
            rule="highest durable scalar training return then shortest history; consider every rewarded state after the latest large rewards",
            comparator="exploratory pilot from a later self-earned training history",inputs="RGB, scalar reward and done",
            training_restores=True,frozen_evaluation_data_used=False))
    plan=read(out/"plan.json")
    smoke=root/"controls/outcome-collection-smoke"
    small=out/"smoke"
    if not (out/"smoke-verified.json").exists():
        call("explore_reward_frontiers.py","--source",smoke,"--record",smoke/"best.json","--motion",probe,
            "--output",small,"--steps",512,"--capacity",16,"--seed",plan["seed"])
        result=read(small/"completed.json")
        source=small/result["best"]["audit"]
        call("pixel_replay.py",source/"game.json",out/"original-rom-smoke","--decision",result["best"]["length"])
        # The training fork writes game.bin and the reference emulator replays the same actions from boot.
        replay=read(out/"original-rom-smoke/verification.json")
        assert result["status"]["new_buttons"]==512 and result["status"]["branches"]>1
        assert replay["restores_during_game"]==0 and replay["input_sha256"]==result["best"]["tape_sha256"]
        write_json(out/"smoke-verified.json",dict(branches=result["status"]["branches"],new_buttons=512,
            replay_buttons=replay["buttons"],original_rom_sha1=replay["rom_sha1"],sha256=replay["input_sha256"]))
    while int(next(s.split()[1] for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:")))<1550000:
        time.sleep(20)
    call("explore_reward_frontiers.py","--source",plan["source"],"--record",out/"selected-record.json","--motion",probe,
         "--output",destination,"--steps",plan["steps"],"--capacity",plan["capacity"],"--seed",plan["seed"])

if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),
                      EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    out.mkdir(parents=True,exist_ok=True)
    with (out/"job.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
