"""Compare landmark eligibility on the same earned pixels and rewards, before opening any emulator."""
import json
from pathlib import Path
import sys
import numpy as np

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from scene_reservoir import SceneReservoir,appearance
from reward_branch_search import Point,visual_code,write_json

root=project/"runs/pixels-only-01/official-wsl"
source=root/"paired-self-earned-frontiers/visual-slots-0/verified-training-tapes"
sha=json.loads((source.parents[1]/"continued-reward-search/branch-origin.json").read_text())["parent_record"]["tape_sha256"]
data=source/sha

class Snapshots:
    def __init__(self):
        self.frames={}
    def command(self,op,slot):
        if op=="save":
            self.frames[slot]=self.current.copy()
        elif op=="restore":
            return self.frames[slot]
        else:
            raise ValueError(op)

def run():
    rows={}
    for label,reward_only,recent in (("periodic",False,0),("positive_scalar",True,0),("positive_recent",True,32)):
        emulator=Snapshots()
        table=SceneReservoir(emulator,np.random.default_rng(11234),128,True,reward_only,recent)
        step=stage=total=eligible=0
        for path in sorted(data.glob("chunk-*.npz")):
            with np.load(path) as chunk:
                image=chunk["image"]
                for index,r in enumerate(chunk["reward"]):
                    step+=1
                    cents=round(float(r)*100)
                    total+=cents
                    stage+=int(cents>=100)
                    accepts=table.accepts(step,1,cents)
                    assert accepts==(cents>0 if reward_only else step%16==0 or cents>=100)
                    if accepts:
                        eligible+=1
                        emulator.current=image[index+1]
                        table.consider(Point(visual_code(emulator.current),total,step,stage=stage),emulator.current)
        for slot,p in enumerate(table.points):
            assert p.stage==table.stage and visual_code(emulator.frames[1000+slot])==p.screen
            assert appearance(emulator.frames[1000+slot])==table.keys[slot]
        assert step==36058 and total==3686
        newest=list(table.seen)[-32:]
        retained=sum(key in table.ids for key in newest)
        if recent:
            assert [table.keys[s] for s in table.recent_slots]==newest and retained==recent
        rows[label]=dict(eligible=eligible,latest_stage=table.stage,
            retained=len(table.points),latest_stage_appearances=len(table.seen),first_accepted=min(p.length for p in table.points),
            last_accepted=max(p.length for p in table.points),newest_32_retained=retained)
    assert 0<rows["positive_scalar"]["eligible"]<rows["periodic"]["eligible"]
    report=dict(source_training_tape_sha256=sha,actions=step,entries=rows,all_retained_frames_match=True)
    write_json(root/"controls/training-restore-distribution/recent-reward-landmarks.json",report)
    print(json.dumps(report),flush=True)

if __name__=="__main__":
    run()
