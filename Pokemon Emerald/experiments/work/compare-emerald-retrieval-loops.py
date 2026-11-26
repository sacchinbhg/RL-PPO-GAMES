"""Test a visual-memory loop escape in matched, uninterrupted original games."""
from collections import Counter
import fcntl
import hashlib
from itertools import groupby
import json
import os
from pathlib import Path
import sys

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from earned_memory import play
from train_earned_dreamer import write_json

root=project/"runs/pixels-only-01/official-wsl"
family=root/"earned-screen-memory"
out=root/"controls/retrieval-loops"
read=lambda p:json.loads(p.read_text())


def measured(arm):
    result=read(arm/"result.json")
    step=result["decisions"]
    audit,=(arm/f"step-{step:07}").glob("*.json")
    data=read(audit)
    tape=audit.with_suffix(".bin").read_bytes()
    assert data["steps"]==len(tape)==step and result["in_game_restores"]==0
    longest,action=max((sum(1 for _ in group),key) for key,group in groupby(tape))
    return dict(buttons=step,sha256=hashlib.sha256(tape).hexdigest(),events=data["events"],
                maps=data["maps_first_reached"],longest_consecutive_action=action,
                longest_consecutive_length=longest,actions_in_last_8192=dict(Counter(tape[-8192:])),
                first_recovery=result.get("first_recovery"),recoveries=result.get("recoveries",0)),tape


def compare(prior,new):
    a,first=measured(prior)
    b,second=measured(new)
    prefix=next((i for i,(u,v) in enumerate(zip(first,second)) if u!=v),min(len(first),len(second)))
    trigger=b["first_recovery"]
    matched=first==second if trigger is None else prefix>=trigger-1
    return dict(control=a,with_recovery=b,identical_leading_buttons=prefix,matched_through_first_intervention=matched,
                interpretation="separate cold games; a divergence before intervention prevents attributing later differences to recovery")


def run():
    baseline,_=measured(family/"clean")
    assert baseline["longest_consecutive_length"]==25626
    write_json(out/"registration.json", dict(repeated_buttons_observed_in_control=baseline["longest_consecutive_length"],
        control_source=baseline["sha256"],recovery="8 visits to the same training index in 32 decisions trigger 16 exploratory actions",
        block_input="only the frozen controller's retrieved memory index; no coordinates, reward labels, or game-specific button",
        tests=["matched clean original game","matched extra wait every 257 decisions"],
        independent_replication="fresh held-back seed 197479, extra wait every 257 decisions, both policies",
        horizon=65536,frames_per_button=24,metrics=["game milestones","longest repeated button","map coverage"],
        original_run_from_boot=True,restores=0))
    for name,delay in (("clean",0),("perturbed",257)):
        dest=family/("loop-recovery-"+name)
        play(family/"model",dest,197461,65536,delay,True,True)
        write_json(out/(name+".json"),compare(family/name,dest))
    for suffix,guard in (("control",False),("treatment",True)):
        play(family/"model",family/("loop-replication-"+suffix),197479,65536,257,True,guard,True)
    write_json(out/"replication.json",compare(family/"loop-replication-control",family/"loop-replication-treatment"))


if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),
                      EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
    out.mkdir(parents=True,exist_ok=True)
    with (out/"job.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
