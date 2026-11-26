"""Replay the common memory prefix once and inspect exactly where the two continuation policies spent time."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
from PIL import Image

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from emerald import Emerald, MAPS, PHASES, verify_rom
from pixel_env import Score,game_paths
from reward_branch_search import write_json


def run(arm,recovery=False):
    out=arm/("handoff-recovery" if recovery else "handoff-review")
    out.mkdir(exist_ok=True)
    if (out/"verified.json").exists():
        return
    cutoff=json.loads((arm/"handoff.json").read_text())["shared_buttons"]
    records=[]
    for i in range(2):
        audit,=(arm/f"games/env-{i}/step-0065536").glob("*.json")
        tape=audit.with_suffix(".bin").read_bytes()
        record=json.loads(audit.read_text())
        assert len(tape)==record["steps"]==65536
        records.append((record,tape))
    assert records[0][1][:cutoff]==records[1][1][:cutoff]
    core,rom=game_paths()
    verify_rom(rom)
    game=Emerald(str(core),str(rom))
    try:
        game.load(game.boot)
        score=Score(records[0][0]["uncapped_exploration"])
        for n,button in enumerate(records[0][1][:cutoff],1):
            _,key=game.step(button)
            score.read(game,key)
        snapshot=game.save()
        start=copy.deepcopy(score)
        Image.fromarray(game.screen).save(out/"last-shared-screen.png")
        source=dict(phase=PHASES[key[0]],map=MAPS[key[1]][key[2]])
        results={}
        for i,(audit,tape) in enumerate(records):
            label=("dreamer","uniform")[i]
            if i:
                game.load(snapshot)
                score=copy.deepcopy(start)
            phase_counts=Counter()
            maps=Counter()
            buttons=Counter()
            first_new={}
            was_battle=source["phase"]=="battle"
            fights=0
            previous_map=source["map"]
            whiteouts=[]
            pending=None
            callback_active=False
            for n,button in enumerate(tape[cutoff:],cutoff+1):
                _,key=game.step(button)
                score.read(game,key)
                if recovery:
                    callback=int.from_bytes(game.ram(0x030022C4,4),"little")&~1
                    if callback==0x08085f58 and not callback_active:
                        pending=dict(decision=n,previous_overworld_map=previous_map,
                                     battle_outcome=int.from_bytes(game.ram(0x0202433a,1),"little"),
                                     first_party_hp=int.from_bytes(game.ram(0x020244ec+86,2),"little"))
                        whiteouts.append(pending)
                    callback_active=callback==0x08085f58
                phase_counts[PHASES[key[0]]]+=1
                if key[0]==6 and not was_battle:
                    fights+=1
                was_battle=key[0]==6
                if key[0]==5 and game.verified_story(key):
                    name=MAPS[key[1]][key[2]]
                    maps[name]+=1
                    previous_map=name
                    if pending is not None:
                        pending.update(first_overworld_after=n,recovery_map=name,
                                       hp_after=int.from_bytes(game.ram(0x020244ec+86,2),"little"))
                        pending=None
                    if name not in start.first_maps:
                        first_new.setdefault(name,n)
                buttons[button]+=1
                if n%8192==0:
                    write_json(out/"status.json",dict(arm=label,through=n))
            assert score.events==audit["events"] and score.first_maps==audit["maps_first_reached"]
            assert len(score.cells)==audit["first_visit_cells"]
            Image.fromarray(game.screen).save(out/(label+"-last-screen.png"))
            results[label]=dict(phases=dict(phase_counts),overworld_maps=dict(maps),new_maps=first_new,
                                battles_started_after_handoff=fights,executed_buttons=dict(buttons),sha256=hashlib.sha256(tape).hexdigest())
            if recovery:
                results[label]["whiteouts"]=whiteouts
                write_json(out/(label+".json"),results[label])
        write_json(out/"verified.json",dict(shared_decisions=cutoff,starting_scene=source,
                   new_decisions_per_arm=65536-cutoff,diagnostic_shared_prefix_restores=1,original_games_used_restores=0,results=results))
    finally:
        game.close()

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm",type=Path)
    parser.add_argument("--battle-recovery",action="store_true")
    args=parser.parse_args()
    run(args.arm,args.battle_recovery)
