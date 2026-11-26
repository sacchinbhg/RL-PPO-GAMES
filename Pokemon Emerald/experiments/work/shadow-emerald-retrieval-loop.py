"""Compare two frozen memory policies on the same original-ROM frames and executed actions."""
from collections import Counter
import hashlib
from itertools import groupby
import json
import os
from pathlib import Path
import sys

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from earned_memory import Memory
from emerald import Emerald,ROM_SHA1
from pixel_env import Score,game_paths
from train_earned_dreamer import write_json

root=project/"runs/pixels-only-01/official-wsl"
family=root/"earned-screen-memory"
out=root/"controls/retrieval-loops/identical-frames"
read=lambda p:json.loads(p.read_text())


def run():
    audit,=(family/"clean/step-0065536").glob("*.json")
    original=read(audit)
    tape=audit.with_suffix(".bin").read_bytes()
    sha=hashlib.sha256(tape).hexdigest()
    assert sha==read(out.parent/"registration.json")["control_source"]
    at=0
    runs=[]
    for action,group in groupby(tape):
        n=sum(1 for _ in group)
        runs.append((n,at+1,action))
        at+=n
    length,start,repeated=max(runs)
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),
                      EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
    config=read(family/"clean/provenance.json")
    agents=[Memory(family/"model",config["seed"],True,guard) for guard in (False,True)]
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"registration.json",dict(original_sha256=sha,original_button_run=length,start_decision=start,
        repeated_button=repeated,matched_inputs="both frozen controllers see the same RGB, reward and historically executed action at every decision",
        interpretation="off-policy replay to diagnose the action choice; fresh games test actual progress"))
    score=Score(original["uncapped_exploration"])
    game=Emerald(*game_paths())
    counts=[Counter(),Counter()]
    matches=[0,0]
    first_diff=None
    try:
        game.load(game.boot)
        frame=game.boot_image
        reward=0.
        for n,button in enumerate(tape,1):
            choices=[agent.choose(frame,reward) for agent in agents]
            if choices[0]!=choices[1] and first_diff is None:
                assert agents[1].first_recovery is not None and agents[1].first_recovery<=n
                first_diff=dict(decision=n,control=choices[0],guard=choices[1],historical=button)
            for i,agent in enumerate(agents):
                matches[i]+=int(choices[i]==button)
                if start<=n<start+length:
                    counts[i][choices[i]]+=1
                agent.executed(button)
            frame,state=game.step(button)
            reward,done=score.read(game,state)
            if n%4096==0:
                write_json(out/"status.json",dict(buttons=n,first_difference=first_diff,recoveries=agents[1].recovery_count))
        assert score.events==original["events"] and score.first_maps==original["maps_first_reached"]
        write_json(out/"result.json",dict(original_sha256=sha,rom_sha1=ROM_SHA1,buttons=len(tape),first_difference=first_diff,
                   first_recovery=agents[1].first_recovery,guard_recoveries=agents[1].recovery_count,
                   matched_historical_actions=matches,original_repeat_length=length,
                   proposed_during_original_repeat=dict(control=dict(counts[0]),guard=dict(counts[1])),
                   input_tape_fixed=True,gameplay_changes_tested=False))
    finally:
        game.close()


if __name__=="__main__":
    run()
