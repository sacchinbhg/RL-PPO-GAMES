"""Inspect trainer encounters and battle outcomes during a separate original-ROM replay."""
import hashlib
import json
from pathlib import Path
import sys
import time
from collections import Counter

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from emerald import Emerald,MAPS,ROM_SHA1,verify_rom
from pixel_env import Score,game_paths
from reward_branch_search import write_json


def run(audit,out):
    info=json.loads(audit.read_text())
    tape=audit.with_suffix(".bin").read_bytes()
    assert len(tape)==info["steps"]
    core,rom=game_paths()
    verify_rom(rom)
    game=Emerald(str(core),str(rom))
    score=Score(info["uncapped_exploration"])
    read=lambda addr,n:int.from_bytes(game.ram(addr,n),"little")
    seen=[]
    active=None
    last_map=None
    map_steps=Counter()
    trainer_frames=Counter()
    try:
        game.load(game.boot)
        for n,b in enumerate(tape,1):
            _,key=game.step(b)
            score.read(game,key)
            if key[0]==5 and game.verified_story(key):
                last_map=MAPS[key[1]][key[2]]
                map_steps[last_map]+=1
                if active:
                    value=read(0x0202433a,1)
                    if active["observed_zero_outcome"] and value and not active["first_nonzero_outcome"]:
                        active["first_nonzero_outcome"]=[n,value]
                    active["first_overworld_after"]=n
                    active=None
            if key[0]!=6:
                continue
            flags=read(0x02022fec,4)
            trainer=read(0x02038bca,2) if flags&8 else None
            if trainer is not None:
                trainer_frames[trainer]+=1
            if active is None:
                active=dict(first_battle_frame=n,previous_overworld_map=last_map,trainer_ids=[],
                            observed_zero_outcome=False,first_nonzero_outcome=None,first_overworld_after=None)
                seen.append(active)
            if trainer is not None and trainer not in active["trainer_ids"]:
                active["trainer_ids"].append(trainer)
            value=read(0x0202433a,1)
            if value==0:
                active["observed_zero_outcome"]=True
            elif active["observed_zero_outcome"] and not active["first_nonzero_outcome"]:
                active["first_nonzero_outcome"]=[n,value]
        out.parent.mkdir(parents=True,exist_ok=True)
        if score.events!=info["events"] or len(score.cells)!=info["first_visit_cells"]:
            mismatch=out.with_name(out.stem+f"-mismatch-{time.time_ns()}.json")
            write_json(mismatch,dict(input_sha256=hashlib.sha256(tape).hexdigest(),buttons=len(tape),
                       expected_events=info["events"],observed_events=score.events,
                       expected_cells=info["first_visit_cells"],observed_cells=len(score.cells),
                       expected_maps=info.get("maps_first_reached"),observed_maps=score.first_maps,
                       observed_battles=seen,not_accepted_as_matching_replay=True))
            raise ValueError(f"Cold replay disagreed with the recorded game. Details: {mismatch}")
        # The six IDs are the game's named Route 103 rival variants, checked against opponents.h.
        rival={520,523,526,529,532,535}
        write_json(out,dict(input_sha256=hashlib.sha256(tape).hexdigest(),buttons=len(tape),rom_sha1=ROM_SHA1,
            in_game_restores=0,maps=score.first_maps,overworld_decisions=dict(map_steps),recorded_battles=seen,
            trainer_battle_decisions=dict(trainer_frames),observed_route103_rival_ids=sorted(rival&trainer_frames.keys()),
            verified_events=score.events))
    finally:
        game.close()


if __name__=="__main__":
    run(Path(sys.argv[1]),Path(sys.argv[2]))
