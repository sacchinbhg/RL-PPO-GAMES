"""Replay the real rewards through the actual selector and measure final offline exposure."""
from pathlib import Path
import json
import sys
import numpy as np

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path[:0]=[str(project),"/opt/emerald-rl/source"]
from embodied.core.replay import Replay
from reward_replay import install_reward_sampling

root=project/"runs/pixels-only-01/official-wsl"
paths=list((root/"seed-71299-h15-reward-replay/train/replay").glob("*.npz"))
by={p.stem.split("-")[1]:p for p in paths}
succ={p.stem.split("-")[2] for p in paths}
roots=sorted(p for k,p in by.items() if k not in succ)
r=Replay(length=33,capacity=131072,online=False)
install_reward_sampling(r,1,.25,71299,retain=True)
for block in (roots[:4],roots[4:]):
    streams=[]
    for f in block:
        parts=[]
        while f:
            with np.load(f) as z:
                parts.append({k:z[k].copy() for k in ("reward","action","is_first","is_last")})
            f=by.get(f.stem.split("-")[2])
        streams.append({k:np.concatenate([d[k] for d in parts]) for k in parts[0]})
    sizes={len(s["reward"]) for s in streams}
    assert len(streams)==4 and len(sizes)==1
    for t in range(sizes.pop()):
        for worker,series in enumerate(streams):
            r.add({k:v[t] for k,v in series.items()},worker)
stats=r.stats()
assert stats["live_large_windows"]==0 and stats["reward_windows"]==564
count=8192
regular=r.sample(count,mode="report")["reward"][:,1:]
sample=r.sample(count)["reward"][:,1:]
assert regular.max()<1 and sample.max()>=4
record=dict(actual_transitions=262152,offline_draws=count,active_positive_windows=stats["reward_windows"],
            active_large_windows=stats["live_large_windows"],retained_windows=stats["retained_windows"],
            uniform_story_fraction=float((regular.max(1)>=1).mean()),
            retained_story_fraction=float((sample.max(1)>=1).mean()))
(root/"controls/retained-reward-exposure.json").write_text(json.dumps(record,indent=2))
print(record,flush=True)
