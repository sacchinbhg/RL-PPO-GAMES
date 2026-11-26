"""Paired frozen-policy interventions from the same eight emulator states."""
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

project=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project))
from emerald import BUTTONS
from experiment import location
from pool_gba import Pool
HORIZON=2048

def test(backend):
    import torch
    from policy import Policy,tensors
    cfg=json.loads((project/"local.json").read_text(encoding="utf-8-sig"))
    run=project/"runs"/("native-night-20261124/long" if backend=="native" else "world-12h-20261124")
    out=project/"starter-diagnostics"
    mode="bedroom" if backend=="native" else "after-rival"
    torch.set_num_threads(1)
    early=10986496 if backend=="native" else 6009856
    root=run/"train-01"
    paths={"earlier":root/"snapshots"/f"{early:010}"/"policy.pt","final":root/"policy.pt"}
    nets={}
    for label,path in paths.items():
        net=Policy().eval()
        net.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"))
        nets[label]=net
    pool=Pool(8,cfg["native_core" if backend=="native" else "core"],cfg["rom"],backend)
    try:
        if mode=="bedroom":
            log=np.load(run/"evaluations/final/moments/buttons.npz")["actions"]
            prefix=log[:1024]
        else:
            log=np.load(run/"evaluations/step-6009856/moments/buttons.npz")["actions"]
            prefix=np.tile(log[:785,0:1], (1,8))
        for acts in prefix:
            obs=pool.step(acts)
        starts=obs
        states=[pool.request(i,"save") for i in range(8)]
        start_maps=[location(k) for _,k in starts]
        Image.fromarray(pool.request(0,"frame")).save(out/(backend+"-branch-start.png"))
        if mode=="bedroom":
            assert all(k[6]==0 for _,k in starts)
        else:
            assert all(k[6]==1 for _,k in starts)
            audit=json.loads((run/"evaluations/step-6009856/evaluation.json").read_text())
            assert audit["trials"][0]["verified_event_decision"]["met_rival"]==len(prefix)
        uniforms=np.random.default_rng(825126).random((HORIZON,8))
        report=dict(backend=backend,starting_point=mode,prefix_decisions=len(prefix),budget=HORIZON,
                    earlier_steps=early,sampling_seed=825126,start_maps=start_maps,arms={})
        for arm in ("earlier","final","final_10pct_uniform","uniform"):
            model=nets["earlier" if arm=="earlier" else "final"]
            for i,(state,(_,key)) in enumerate(zip(states,starts)):
                assert pool.request(i,"load",state)==key
            obs=starts
            trials=[{} for _ in range(8)]
            seen=[set() for _ in range(8)]
            counts=Counter()
            dists=[]
            for step in range(HORIZON):
                with torch.inference_mode():
                    d,_=model(*tensors(obs,[None]*8,"cpu"))
                    probs=d.probs.numpy().astype(float)
                if arm=="final_10pct_uniform":
                    probs=.9*probs+.1/len(BUTTONS)
                elif arm=="uniform":
                    probs.fill(1/len(BUTTONS))
                dists.append(probs)
                buttons=(probs.cumsum(axis=1)<uniforms[step,:,None]).sum(axis=1).clip(max=len(BUTTONS)-1)
                counts.update(BUTTONS[a] for a in buttons)
                result=pool.step(buttons,verified=True)
                obs=[x[0] for x in result]
                for i,((img,k),facts) in enumerate(result):
                    if facts is None: continue
                    name=location(k)
                    seen[i].add(name)
                    for event,got in (("clock",k[6]),("rival",facts[0]),("route",name=="Route101"),("party",facts[1])):
                        if got:trials[i].setdefault(event,step+1)
            probs=np.asarray(dists)
            report["arms"][arm]=dict(trials=trials,maps=[sorted(m) for m in seen],buttons=dict(counts),
                                     mean_max_button_probability=float(probs.max(axis=2).mean()),
                                     mean_button_probability=dict(zip(BUTTONS,probs.mean(axis=(0,1)).tolist())))
            (out/(backend+"-branches.json")).write_text(json.dumps(report,indent=2)+"\n")
            print(backend,arm,{k:sum(k in t for t in trials) for k in ("clock","rival","route","party")},flush=True)
    finally:
        pool.close()

if __name__=="__main__":
    test(sys.argv[1])
