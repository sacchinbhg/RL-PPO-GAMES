"""Decompose the original mGBA progress score along the replayed diagnostic route."""
import json,sys
from pathlib import Path
project=Path(__file__).resolve().parents[2]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from emerald import Emerald
from experiment import location,progress
cfg=json.loads((project/"local.json").read_text())
out=project/"starter-diagnostics"
e=Emerald(cfg["core"],cfg["rom"])
try:
    key=e.descriptor()
    best=progress(key)
    seen={key}
    events={}
    ever={}
    ranks={}
    collisions=set()
    for n,button in enumerate((out/"guided-party.bin").read_bytes(),1):
        _,k=e.step(button)
        name=location(k)
        rank=progress(k)
        bonus=max(0,rank-best)
        novel=.25*(k not in seen)
        f=e.verified_story(k)
        facts=dict(clock=bool(k[6]),bag_signal=bool(k[7]))
        if f:
            facts.update(rival=f[0],route101=name=="Route101",party=f[1])
            if k in ever and ever[k]!=f[0]:collisions.add(k)
            ever[k]=f[0]
        for event,yes in facts.items():
            if yes and event not in events:
                events[event]=dict(decision=n,old_rank=progress(key),rank=rank,
                    progress_bonus=bonus,new_cell_bonus=novel,previous_best=best)
        best=max(best,rank)
        seen.add(k)
        key=k
    assert events["party"]["decision"]==683
    (out/"reward-trace.json").write_text(json.dumps(dict(events=events,exact_keys_seen_before_and_after_rival=len(collisions),
        scoring="one continuous no-goal tape; the real trainer truncates at 512 decisions"),indent=2)+"\n")
    print(events,flush=True)
finally:e.close()
