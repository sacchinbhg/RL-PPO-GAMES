"""Compare an uninterrupted suffix with a fresh-state restore and the historic training restore."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from earned_starts import EarnedStarts,restore_score
from emerald import Emerald
from pixel_env import Score,game_paths
from train_earned_dreamer import write_json

root=project/"runs/pixels-only-01/official-wsl"
bankdir=root/"controls/dreamer-earned-resets/starts"
audit=root/"paired-self-earned-frontiers/bank-movement-ordinary-restart1/first-trainers/trainer-00535/game.json"
out=root/"controls/endpoint-button-sweep/replay-comparison"
read=lambda p:json.loads(p.read_text())


def summarize(game,score,image):
    return dict(rgb=hashlib.sha256(image.tobytes()).hexdigest(),descriptor=list(game.descriptor()),
                events=dict(score.events),cells=len(score.cells),maps=dict(score.first_maps))


def follow(game,score,tape):
    rows=[]
    for button in tape:
        image,key=game.step(button)
        reward,done=score.read(game,key)
        rows.append((hashlib.sha256(image).hexdigest(),key,round(100*reward),done))
    return rows,summarize(game,score,image)


def compare(a,b,start):
    diff=[start+n+1 for n,(x,y) in enumerate(zip(a,b)) if x!=y]
    return dict(different_observations=len(diff),first_difference=diff[0] if diff else None)


if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
    out.mkdir(parents=True,exist_ok=True)
    bank=EarnedStarts(bankdir)
    tape=audit.with_suffix(".bin").read_bytes()
    info=read(audit)
    points={r["prefix"]:i for i,r in enumerate(bank.records) if tape.startswith(bank.tape[:r["prefix"]])}
    start=max(points)
    game=Emerald(*game_paths())
    try:
        game.load(game.boot)
        score=Score(True)
        checks=[]
        for n,button in enumerate(tape[:start],1):
            image,key=game.step(button)
            score.read(game,key)
            if n in points:
                r=bank.records[points[n]]
                with np.load(bankdir/r["filename"],allow_pickle=False) as stored:
                    old=restore_score(json.loads(stored["score"].tobytes()))
                    checks.append(dict(prefix=n,same_rgb=bool(np.array_equal(image,stored["image"])),same_descriptor=list(key)==r["descriptor"],
                         same_events=score.events==old.events,same_cells=score.cells==old.cells,same_maps=score.first_maps==old.first_maps))
            if n in points or n%4096==0:
                write_json(out/"progress.json",dict(cold_decisions=n,expected=start,earned_prefix_checks=checks))
        new_state=game.save()
        saved_score=deepcopy(score)
        direct,direct_end=follow(game,score,tape[start:])
        game.load(new_state)
        fresh,fresh_end=follow(game,saved_score,tape[start:])
        _,old_score,_=bank.restore(game,points[start])
        historic,historic_end=follow(game,old_score,tape[start:])
        write_json(out/"result.json",dict(tape_sha256=hashlib.sha256(tape).hexdigest(),checkpoint=start,real_suffix_decisions=len(tape)-start,
             checks=checks,direct_vs_just_saved=compare(direct,fresh,start),direct_vs_historic=compare(direct,historic,start),
             direct=direct_end,just_saved=fresh_end,historic=historic_end,expected_cells=info["first_visit_cells"],expected_events=info["events"],
             purpose="human replay diagnostics only"))
    finally:
        game.close()
