from pathlib import Path
import sys
import numpy as np

base=Path(__file__).resolve().parents[1]
proj=base/"outputs/EmeraldRL"
sys.path.insert(0,str(proj))
from scene_reservoir import SceneReservoir
from reward_branch_search import Point,visual_code,write_json
from option_explorer import DURATIONS

class FakeSlots:
    def __init__(self,pic):
        self.pic=pic
        self.slots={}
    def command(self,op,n):
        if op=="save":
            self.slots[n]=self.pic.copy()
        elif op=="restore":
            return self.slots[n]
        else:
            raise AssertionError(op)

root=proj/"runs/pixels-only-01/official-wsl"
data,=next((root/"paired-self-earned-frontiers/visual-slots-0/verified-training-tapes").iterdir()).glob("chunk-000000000.npz")
with np.load(data) as chunk:
    image=chunk["image"][0]
first=Point(visual_code(image),100,500,stage=1)
fake=FakeSlots(image)
table=SceneReservoir(fake,np.random.default_rng(17),1,True)
table.consider(first,image)
seen=set()
while len(seen)<45:
    if table.start() is None:
        continue
    choice=table.next_option(DURATIONS,32)
    assert choice not in seen and table.next_option(DURATIONS,32) is None
    seen.add(choice)
    table.consider(Point(first.screen,200,501,stage=1),image)
    assert table.points[0].length==500
assert table.distinct_trials==45
table.consider(Point(first.screen,300,502,stage=2),image)
assert not table.attempts and table.stage==2 and table.points[0].length==502
target=root/"controls/training-restore-distribution/systematic-options.json"
write_json(target,dict(buttons=9,durations=DURATIONS,distinct_before_repeat=len(seen),
           same_saved_state_across_trials=True,reset_on_new_reward_stage=True))
