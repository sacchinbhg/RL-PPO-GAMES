"""Change only the granularity of the scalar novelty payment in a matched reward sweep."""
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
out=root/"controls/individual-tile-reward"
previous=root/"controls/reward-frontier-expansion"
arm=root/"paired-self-earned-frontiers/sweep-individual-tiles"
if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so",
                      OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
    out.mkdir(exist_ok=True)
    with (out/"job.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=json.loads((previous/"plan.json").read_text())
        write_json(out/"registration.json",dict(seed=plan["seed"],source=plan["selected"]["tape_sha256"],steps=plan["steps"],
            control="sweep-reward-frontiers",only_change="0.005 once per individual map tile instead of 0.02 once per 2x2 cell",
            actor_observations="RGB, scalar reward, episode boundary",original_score_in_audits=True,coordinates_in_model=False))
        with (out/"run.log").open("a") as log:
            subprocess.run([sys.executable,"-u",str(base/"work/check-emerald-tile-rewards.py")],stdout=log,stderr=subprocess.STDOUT,check=True)
            while int(next(s.split()[1] for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:")))<1900000:
                time.sleep(30)
            subprocess.run([sys.executable,"-u",str(project/"explore_reward_frontiers.py"),"--source",plan["source"],
                "--record",str(previous/"selected-record.json"),"--output",str(arm),"--motion",str(root/"controls/visual-motion/route-training.json"),
                "--steps",str(plan["steps"]),"--capacity",str(plan["capacity"]),"--seed",str(plan["seed"]),"--reward-sweep","--individual-tile-rewards"],
                stdout=log,stderr=subprocess.STDOUT,check=True)
