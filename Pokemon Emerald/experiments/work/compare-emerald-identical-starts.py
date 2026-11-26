"""Start matched motion-planning searches from identical independently verified emulator states."""
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
from earned_starts import EarnedStarts
from pixel_env import game_paths
from emerald import Emerald
from reward_branch_search import visual_code
from train_earned_dreamer import write_json

root=project/"runs/pixels-only-01/official-wsl"
family=root/"paired-self-earned-frontiers"
out=root/"controls/local-screen-motion/identical-starts"
bankdir=root/"controls/dreamer-earned-resets/starts"
parent=root/"controls/reward-frontier-expansion"
read=lambda p:json.loads(p.read_text())


def run():
    suffix=os.environ.get("EMERALD_COMPARISON_SUFFIX","")
    if suffix and not suffix.isalnum():
        raise ValueError("Comparison suffix must be alphanumeric")
    label_suffix="-"+suffix if suffix else ""
    report=out/suffix if suffix else out
    report.mkdir(parents=True,exist_ok=True)
    plan=read(parent/"plan.json")
    write_json(report/"registration.json",dict(seed=173746,verified_reset_count=8,source=plan["selected"]["tape_sha256"],
       reason="the initial learned-movement worker did not reproduce the independently earned source tape from its own boot",
       comparison="both actors receive the same eight original emulator states and begin with empty tabular transition counts",
       difference="only the learned arm ranks three-decision movement options by its pixel model",
       budget_per_arm=131072,active_slots_per_arm=768,observations="RGB and scalar reward",frame_repeat=24))
    while not (bankdir/"ready.json").exists():
        time.sleep(30)
    bank=EarnedStarts(bankdir)
    game=Emerald(*game_paths())
    try:
        _,score,tape=bank.restore(game,len(bank.records)-1)
        start=len(tape)
        total=bank.records[-1]["scalar_return_cents"]
        for action in bank.tape[start:]:
            image,state=game.step(action)
            reward,done=score.read(game,state)
            total+=round(reward*100)
        assert not done and total==plan["selected"]["return_cents"] and visual_code(image).hex()==plan["selected"]["screen"]
        write_json(report/"matched-source.json",dict(final_screen=visual_code(image).hex(),source_reward_cents=total,
                   source_buttons=len(bank.tape),all_starts_verified=bank.info["all_starts_verified"],
                   snapshot_sha256=[r["sha256"] for r in bank.records]))
    finally:
        game.close()
    children=[]
    for treatment in (True,False):
        label="learned" if treatment else "ordinary"
        folder=family/("bank-movement-"+label+label_suffix)
        if (folder/"status.json").exists() and not (folder/"completed.json").exists():
            raise RuntimeError(f"Use a new suffix to preserve interrupted results in {folder}")
        cmd=[sys.executable,"-u",str(project/"explore_reward_frontiers.py"),"--source",plan["source"],
             "--record",str(parent/"selected-record.json"),"--output",str(folder),"--motion",str(root/"controls/visual-motion/route-training.json"),
             "--seed","173746","--steps","131072","--capacity","768","--reward-sweep","--movement-triples","--earned-starts",str(bankdir)]
        if treatment:
            cmd+=["--scroll-model",str(root/"controls/local-screen-motion/fitted")]
        while int(next(s.split()[1] for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:")))<1700000:
            time.sleep(30)
        with (report/(label+".log")).open("a") as log:
            children.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT))
    assert all(p.wait()==0 for p in children)
    a,b=(read(family/("bank-movement-"+n+label_suffix)/"initialization.json") for n in ("learned","ordinary"))
    assert a==b and a["verified_earned_starts"]==8
    write_json(report/"result.json",dict(shared_initialization=a,arms={n:read(family/("bank-movement-"+n+label_suffix)/"completed.json") for n in ("learned","ordinary")}))


if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
    out.mkdir(parents=True,exist_ok=True)
    with (out/"job.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
