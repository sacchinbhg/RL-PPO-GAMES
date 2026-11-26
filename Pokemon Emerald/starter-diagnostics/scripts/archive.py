from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import numpy as np
import torch

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project))
from emerald import Emerald
from native_emerald import NativeEmerald
from experiment import location, progress
from world_model import STATE, PIXELS
from policy import N_FEATURES

backend = sys.argv[1]
run = "world-12h-20261124" if backend == "mgba" else "native-night-20261124/long"
ck = torch.load(project / "runs" / run / "train-01/checkpoint.pt", weights_only=False, map_location="cpu")
rows = ck["world"]["replay"].numpy()
rewards = rows[:, -1]
outcomes = defaultdict(Counter)
for row in rows:
    outcomes[row[:STATE+1+N_FEATURES].tobytes()][float(row[-1])] += 1
conflicts = [c for c in outcomes.values() if len(c)>1]
replay = dict(rows=len(rows), positive=int((rewards>0).sum()), unique_inputs=len(outcomes),
              conflicting_inputs=len(conflicts), conflicting_rows=sum(c.total() for c in conflicts),
              conflicting_positive_rows=sum(c.total()-c[0] for c in conflicts),
              current_clock=int((rows[:,PIXELS+13]>.5).sum()),
              acquired_clock=int(((rows[:,PIXELS+13]<.5) & (rows[:,STATE+1+N_FEATURES+PIXELS+13]>.5)).sum()),
              acquired_party_feature=int(((rows[:,PIXELS+14]<.5) & (rows[:,STATE+1+N_FEATURES+PIXELS+14]>.5)).sum()))
cfg = json.loads((project/"local.json").read_text(encoding="utf-8-sig"))
env = NativeEmerald(cfg["native_core"],cfg["rom"]) if backend=="native" else Emerald(cfg["core"],cfg["rom"])
counts=Counter()
starts=Counter()
dist=defaultdict(list)
samples={}
rank_hist=Counter()
try:
    for key,cell in ck["archive"].items():
        env.load(cell["state"])
        k=env.descriptor()
        counts["all"]+=1
        counts["mismatched_descriptor"]+=int(k!=key)
        name=location(k)
        if not name or not 1<=k[5]<=6:
            continue
        if backend=="native":
            vals=env.inspect()
            gender=vals[12]
            route=vals[10]
            verified=(vals[9]>0,1<=vals[11]<=6)
        else:
            read=lambda addr,n:int.from_bytes(env.ram(addr,n),"little")
            sb=read(0x03005D8C,4)
            if env._header(*k[1:3])!=(read(0x02037318+4,4),read(0x02037318+18,2)):
                continue
            gender=read(read(0x03005D90,4)+8,1)
            route=read(sb+0x139c+2*0x60,2)
            verified=(read(sb+0x139c+2*0x50,2)>=1 and read(sb+0x139c+2*0x8d,2)>=3,
                      1<=read(0x020244e9,1)<=6)
        rival_house=("MaysHouse" if gender==0 else "BrendansHouse") in name
        facts=dict(valid=True, clock=bool(k[6]), sent_to_rival=k[5]>=6,
                   rival_house=rival_house, rival_upstairs=rival_house and name.endswith("_2F"),
                   met_rival=verified[0], town_after_rival=verified[0] and name=="LittlerootTown",
                   route_or_rescue=name=="Route101" or route>0, party=verified[1])
        for fact,hit in facts.items():
            if hit:
                counts[fact]+=1
                starts[fact]+=cell["starts"]
                dist[fact].append(len(cell["path"]))
                if fact not in samples or len(cell["path"])<len(samples[fact][1]):
                    samples[fact]=(k,cell["path"])
        rank_hist[str(progress(k))]+=1
finally:
    env.close()
keys=("all","mismatched_descriptor","valid","clock","sent_to_rival","rival_house","rival_upstairs","met_rival","town_after_rival","route_or_rescue","party")
result=dict(backend=backend,cells={k:counts[k] for k in keys},start_counts=dict(starts),
            shortest_saved_controller_paths={k:min(v) for k,v in dist.items()},
            verified_progress_rank_histogram=rank_hist,replay=replay)
out=project/"starter-diagnostics"/(backend+"-archive.json")
out.write_text(json.dumps(result,indent=2)+"\n")
for k in ("met_rival","town_after_rival"):
    if k in samples:
        (project/"starter-diagnostics"/(backend+"-"+k+".bin")).write_bytes(samples[k][1])
print(json.dumps(result),flush=True)
