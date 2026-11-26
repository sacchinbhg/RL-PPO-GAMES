"""Test a short self-distillation intervention on a disposable final-policy copy."""
from copy import deepcopy
import json,sys
from pathlib import Path
import numpy as np
project=Path(__file__).resolve().parents[2]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from experiment import location
from pool_gba import Pool

def main():
    import torch
    from policy import Policy,tensors
    import torch.nn.functional as F
    torch.set_num_threads(2)
    torch.manual_seed(90427)
    cfg=json.loads((project/"local.json").read_text())
    run=project/"runs/native-night-20261124/long/train-01"
    out=project/"starter-diagnostics"
    nets={}
    for label,path in (("earlier",run/"snapshots/0010986496/policy.pt"),("final",run/"policy.pt")):
        net=Policy().eval()
        net.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"))
        nets[label]=net
    pool=Pool(8,cfg["native_core"],cfg["rom"],"native")
    def start():
        for i in range(8):pool.request(i,"load")
        return list(pool.initial)
    def act(probs,u):
        return (probs.astype(float).cumsum(axis=1)<u[:,None]).sum(axis=1).clip(max=8)
    try:
        rows=[]
        obs=start()
        rng=np.random.default_rng(110301)
        successes=set()
        for step in range(512):
            with torch.no_grad():
                image,ram=tensors(obs,[None]*8,"cpu")
                d,_=nets["earlier"](image,ram)
            rows.append((image,ram,d.probs))
            result=pool.step(act(d.probs.numpy(),rng.random(8)),verified=True)
            obs=[pair[0] for pair in result]
            for i,((_,k),flags) in enumerate(result):
                if flags is not None and k[6]: successes.add(i)
        assert successes
        xs,features,labels=[torch.cat([row[j][sorted(successes)] for row in rows]) for j in range(3)]
        student=deepcopy(nets["final"])
        opt=torch.optim.Adam(student.parameters(),lr=.00035)
        for update in range(240):
            ids=torch.randint(len(xs),(128,))
            d,_=student(xs[ids],features[ids])
            loss=F.kl_div(d.logits,labels[ids],reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(),1.)
            opt.step()
        nets["rehearsed"]=student
        result=dict(teacher_steps=10986496,student_base_steps=27060736,train_seed=110301,train_games=8,
                    clock_successful_train_games=len(successes),train_observations=len(xs),updates=240,
                    new_evaluation_seeds=[194307,194311],budget=1024,arms={})
        for name,net in nets.items():
            record=[]
            for seed in result["new_evaluation_seeds"]:
                obs=start()
                rng=np.random.default_rng(seed)
                events=[{} for _ in range(8)]
                for step in range(1,result["budget"]+1):
                    with torch.no_grad():
                        d,_=net(*tensors(obs,[None]*8,"cpu"))
                    pairs=pool.step(act(d.probs.numpy(),rng.random(8)),verified=True)
                    obs=[pair[0] for pair in pairs]
                    for i,((_,k),flags) in enumerate(pairs):
                        if flags is None:continue
                        for event,yes in (("clock",k[6]),("met_rival",flags[0]),("route101",location(k)=="Route101"),("party",flags[1])):
                            if yes:events[i].setdefault(event,step)
                record.extend(events)
            result["arms"][name]=dict(trials=record,counts={k:sum(k in t for t in record) for k in ("clock","met_rival","route101","party")})
            (out/"self-rehearsal.json").write_text(json.dumps(result,indent=2)+"\n")
            print(name,result["arms"][name]["counts"],flush=True)
        if result["arms"]["rehearsed"]["counts"]["clock"]>result["arms"]["final"]["counts"]["clock"]:
            torch.save(student.state_dict(),out/"rehearsed-policy.pt")
    finally:pool.close()
if __name__=="__main__":main()
