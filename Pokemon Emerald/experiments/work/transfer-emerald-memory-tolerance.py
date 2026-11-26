"""Test a native memory ablation on the original game if it generalizes across four fresh native seeds."""
import fcntl
import json
import os
from pathlib import Path
import sys
import time

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from earned_memory import play
from train_earned_dreamer import write_json
root=project/"runs/pixels-only-01/official-wsl"
native=root/"controls/native-memory-tolerance"
family=root/"earned-screen-memory"
read=lambda p:json.loads(p.read_text())
if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so",
                      OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
    with (native/"original-check.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(native/"original-plan.json",dict(entry="at least two additional native games reached town with exact matching at four fixed unseen seeds",
            comparison="original-ROM robustness trial first, using the prior model, seed and extra wait every 257 buttons",
            schedule="65536 fixed decisions per game",fresh_boot=True,restores=0))
        while not (native/"result.json").exists():
            time.sleep(30)
        results=read(native/"result.json")["arms"]
        count=lambda mode:sum(row["town"] is not None for row in results[mode])
        if count("exact")<count("ordinary")+2:
            write_json(native/"original-decision.json",dict(run=False,ordinary_town=count("ordinary"),exact_town=count("exact")))
        else:
            for prior,name,wait in (("perturbed","exact-perturbed",257),("clean","exact-clean",0)):
                cfg=read(family/prior/"provenance.json")
                assert cfg["external_wait_interval"]==wait and cfg["button_budget"]==65536
                play(family/"model",family/name,cfg["seed"],65536,wait,False)
            write_json(native/"original-decision.json",dict(run=True,original_games=["exact-perturbed","exact-clean"]))
