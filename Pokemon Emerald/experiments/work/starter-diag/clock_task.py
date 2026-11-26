"""Replay real button tapes and count exact clock interaction opportunities on mGBA."""
import json, sys, struct
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
project=Path(__file__).resolve().parents[2]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from emerald import Emerald
from experiment import location

def one(arg):
    sub,i=arg
    folder=project/"runs"/sub
    actions=np.load(folder/"moments/buttons.npz")["actions"][:2048,i]
    cfg=json.loads((project/"local.json").read_text(encoding="utf-8-sig"))
    env=Emerald(cfg["core"],cfg["rom"])
    def rd(addr,n):
        return int.from_bytes(env.ram(addr,n),"little")
    dxy={1:(0,1),2:(0,-1),3:(-1,0),4:(1,0)}
    counts=Counter()
    clocks=[]
    try:
        for n,action in enumerate(actions,1):
            k=env.descriptor()
            name=location(k)
            cb=rd(0x030022c4,4)&~1
            if cb in (0x081349f4,0x08134c9c) and not k[6]:
                counts["clock_screen_decisions_before_set"]+=1
            if k[0]==5 and not k[6] and env.verified_story(k) is not None and name.endswith("House_2F"):
                obj=rd(0x02037590+5,1)
                if obj<16:
                    base=0x02037350+36*obj
                    x,y=struct.unpack("<hh",env.ram(base+0x10,4))
                    x-=7
                    y-=7
                    face=rd(base+0x18,1)&15
                    dx,dy=dxy.get(face,(0,0))
                    target=(5 if "Brendans" in name else 3,1)
                    counts["bedroom_decisions_before_set"]+=1
                    if abs(x-target[0])+abs(y-target[1])==1:
                        counts["adjacent"]+=1
                    if (x+dx,y+dy)==target:
                        counts["facing_clock"]+=1
                        counts["facing_clock_and_A"]+=int(action==1)
            _,k1=env.step(int(action))
            if k1[6] and not k[6] and env.verified_story(k1) is not None:
                clocks.append(n)
    finally:
        env.close()
    expected=json.loads((folder/"evaluation.json").read_text())["trials"][i]["verified_event_decision"].get("clock")
    assert (clocks[0] if clocks else None)==(expected if expected is not None and expected<=len(actions) else None)
    return dict(run=sub,trial=i,decisions=len(actions),counts=counts,clock_decision=clocks[:1])

if __name__=="__main__":
    tests=[(folder,i) for folder,ids in (("world-12h-20261124/evaluations/final",range(4)),
             ("native-night-20261124/transfer",range(4)),("native-night-20261124/pilot-transfer",[2]),
             ("world-12h-20261124/evaluations/step-6009856",[0])) for i in ids]
    with ProcessPoolExecutor(max_workers=4) as pool:
        data=list(pool.map(one,tests))
    (project/"starter-diagnostics/clock-task.json").write_text(json.dumps(data,indent=2)+"\n")
    print([(d["trial"],d["counts"],d["clock_decision"]) for d in data],flush=True)
