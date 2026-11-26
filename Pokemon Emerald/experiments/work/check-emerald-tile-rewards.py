"""Verify fractional tile rewards, unchanged RGB/story signals and deduplication across a real rewind."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import numpy as np

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from emerald import Emerald
from pixel_env import Score,game_paths
from reward_branch_env import ForkEnv,TileVisitReward
from train_earned_dreamer import write_json
root=project/"runs/pixels-only-01/official-wsl"
out=root/"controls/individual-tile-reward"


if __name__=="__main__":
    marker=out/"verified.json"
    if marker.exists():
        sys.exit(0)
    unit=TileVisitReward()
    assert sum(unit.adjust(2 if i==0 else 0,(1,2,x,y),i==0) for i,(x,y) in enumerate(((0,0),(0,1),(1,0),(1,1))))==2
    assert unit.adjust(102,(1,2,0,0),True)==100
    source=next((root/"controls/outcome-collection-smoke/verified-training-tapes").glob("769d*/audit/*.bin"))
    tape=source.read_bytes()
    game=Emerald(*game_paths())
    actor=ForkEnv(shared_visit_reward=True,tile_visit_reward=True)
    visited=set()
    snapshots={}
    amounts=[]
    repeated=0
    import copy
    try:
        score=Score(True)
        game.load(game.boot)
        for pass_no,(lo,hi) in enumerate(((0,len(tape)),(1000,len(tape)))):
            if pass_no:
                game.load(snapshots["game"])
                score=copy.deepcopy(snapshots["score"])
                actor.command("restore",0)
            for i,b in enumerate(tape[lo:hi],lo+1):
                if i==1001 and not pass_no:
                    snapshots=dict(game=game.save(),score=copy.deepcopy(score))
                    actor.command("save",0)
                old=len(score.cells)
                img,key=game.step(b)
                original,done=score.read(game,key)
                credited=0.
                if key[0]==5 and game.verified_story(key) is not None:
                    sb=int.from_bytes(game.ram(0x03005d8c,4),"little")
                    x,y,g,m=struct.unpack("<hhBB",game.ram(sb,6))
                    assert (g,m,x//2,y//2)==key[1:5]
                    tile=(g,m,x,y)
                    if tile not in visited:
                        visited.add(tile)
                        credited=.5
                data,cents,term,used=actor.command("step",(b,1))
                expected=round(100*original)-2*(len(score.cells)>old)+credited
                assert used==1 and term==done and cents==expected and np.array_equal(img,data)
                if pass_no:
                    assert credited==0
                    repeated+=1
                else:
                    amounts.append(cents)
        out.mkdir(parents=True,exist_ok=True)
        assert len(visited)>len(score.cells) and any(a>=100 for a in amounts)
        write_json(marker,dict(source=hashlib.sha256(tape).hexdigest(),original_buttons=len(tape),rewound_buttons=repeated,
            original_cells=len(score.cells),individual_tiles=len(visited),tile_reward_per_visit=.005,images_and_story_agree=True,
            duplicate_visit_payments_after_rewind=0))
    finally:
        actor.close()
        game.close()
