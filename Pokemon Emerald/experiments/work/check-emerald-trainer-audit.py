"""Exercise the reporting pipe on a wild battle and repeated trainer IDs without feeding logs to the controller."""
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from reward_branch_env import _worker
from reward_branch_search import write_json

image=np.zeros((40,60,3),dtype=np.uint8)
class Game:
    boot=bytes(4)
    boot_image=image
    def __init__(self,*paths): self.at=0
    def load(self,_): pass
    def close(self): pass
    def step(self,action):
        self.at+=1
        return image,(6,0,0,0,0,0,0,0,0)
    def ram(self,address,size):
        if address==0x02022fec:
            return (0 if self.at==1 else 8).to_bytes(size,"little")
        assert address==0x02038bca
        return (520 if self.at<=3 else 523).to_bytes(size,"little")

class Score:
    def __init__(self,**kwargs):
        self.first_maps={}
        self.events={}
        self.cells=set()
    def read(self,game,descriptor): return 0.,False

class Pipe:
    def __init__(self):
        self.actions=iter([("step",(i,1)) for i in range(4)]+[("close",None)])
        self.sent=[]
    def recv(self): return next(self.actions)
    def send(self,response): self.sent.append(response)
    def close(self): pass

out=project/"runs/pixels-only-01/official-wsl/controls/trainer-contacts"
with tempfile.TemporaryDirectory() as tmp:
    folder=Path(tmp)
    pipe=Pipe()
    with patch("emerald.Emerald",Game),patch("pixel_env.game_paths",return_value=("","")),patch("pixel_env.Score",Score):
        _worker(pipe,trainer_audits=folder)
    found={p.parent.name:p.read_bytes().hex() for p in folder.glob("*/game.bin")}
    assert found=={"trainer-00520":"0001","trainer-00523":"00010203"}
    assert len(pipe.sent)==5 and np.array_equal(pipe.sent[0],image)
    assert all(np.array_equal(r[0],image) and r[1:]==(0,False,1) for r in pipe.sent[1:])
    write_json(out/"logger-fixture.json",dict(test="synthetic diagnostic fixture",stored_button_prefixes=found,
               one_record_per_opponent=True,wild_battle_skipped=True,controller_responses_unchanged=True))
