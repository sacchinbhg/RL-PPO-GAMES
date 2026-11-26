"""Test the final native dynamics checkpoint on saved, goal-free evaluation games."""
from collections import defaultdict
import json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
project=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project))
from experiment import progress
from native_emerald import NativeEmerald
from policy import tensors,N_FEATURES
from world_model import Dynamics,pack,PIXELS

torch.set_num_threads(1)
root=project/"runs/native-night-20261124/long"
cfg=json.loads((project/"local.json").read_text())
model=Dynamics().eval()
ck=torch.load(root/"train-01/checkpoint.pt",weights_only=False,map_location="cpu")
model.load_state_dict(ck["world"]["model"])
model_steps=ck["steps"]
del ck
env=NativeEmerald(cfg["native_core"],cfg["rom"])
records=defaultdict(list)
clock_steps=[]
def flush(buf):
    if not buf:return
    obs,btn,reward,event=zip(*buf)
    with torch.inference_mode():
        image,ram=tensors(obs,[None]*len(buf),"cpu")
        state=pack(image,ram)
        next_state,pred=model.step(state,F.one_hot(torch.tensor(btn),9).float(),ram[:,N_FEATURES:])
    for j,k in enumerate(event):
        records[k].append((float(reward[j]),float(pred[j]),float(next_state[j,PIXELS+13])))
    buf.clear()
try:
    for name in ("step-2707456","step-5426176","step-8210432","step-10986496","final"):
        folder=root/"evaluations"/name
        log=np.load(folder/"moments/buttons.npz")["actions"]
        ev=json.loads((folder/"evaluation.json").read_text())["trials"]
        for i,trial in enumerate(ev):
            env.load(env.boot)
            obs=env.boot_image,env.descriptor()
            seen={obs[1]}
            best=progress(obs[1])
            first=None
            verified_clock=None
            buf=[]
            for t,a in enumerate(log[:512,i],1):
                nxt=env.step(int(a))
                k=nxt[1]
                rank=progress(k)
                reward=.25*(k not in seen)+max(0,rank-best)
                best=max(rank,best)
                seen.add(k)
                tag="other"
                if first is None and k[6]:
                    first=t
                    tag="clock_onset"
                elif first is None and obs[1][0]==5 and obs[1][1]>=0:
                    tag="overworld_before_clock"
                if verified_clock is None and k[6] and env.verified_story(k) is not None:
                    verified_clock=t
                buf.append((obs,int(a),reward,tag))
                obs=nxt
                if len(buf)>=128:flush(buf)
            flush(buf)
            expected=trial["verified_event_decision"].get("clock")
            assert verified_clock==(expected if expected is not None and expected<=512 else None)
            if first:clock_steps.append(dict(snapshot=name,trial=i,first_clock_flag=first,first_verified_clock=verified_clock))
finally:
    env.close()
summary={}
for tag,rows in records.items():
    arr=np.array(rows)
    summary[tag]=dict(n=len(arr),actual_mean_reward=arr[:,0].mean(),predicted_mean_reward=arr[:,1].mean(),
                      actual_median_reward=np.median(arr[:,0]),predicted_median_reward=np.median(arr[:,1]),
                      mean_predicted_next_clock=arr[:,2].mean(),predicted_next_clock_ge_half=int((arr[:,2]>=.5).sum()))
out=dict(frozen_model_steps=model_steps,games=24,decisions_per_game=512,summary=summary,clock_steps=clock_steps)
(project/"starter-diagnostics/heldout-predictor.json").write_text(json.dumps(out,indent=2)+"\n")
print(summary,flush=True)
