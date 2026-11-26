"""Fit an observation-conditioned controller to one autonomous exploration success, then evaluate fresh.

The episodic controller uses a lookup by screen, game variables, button history and consecutive
identical-screen count. It has no reference index or default next action from a tape.
"""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
import pickle
import struct
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from gym_explore import Game
from gym_data import RUN
from experiment import write_json


class History:
    def __init__(self,version=1):
        self.buttons=deque([9]*20,maxlen=20)
        self.previous=None
        self.dwell=0
        self.visits=Counter()
        self.version=version

    def input(self,game):
        pixels=game.env.small_screen()
        descriptor=game.env.descriptor()
        obs=hashlib.blake2b(pixels.tobytes()+struct.pack("<9i",*descriptor),digest_size=16).digest()
        self.dwell=self.dwell+1 if obs==self.previous else 0
        self.previous=obs
        key=obs+bytes(self.buttons)+struct.pack("<H",min(self.dwell,65535))
        coarse=np.asarray(descriptor,dtype=np.float32)/np.array([8,40,100,64,64,6,1,1,256],np.float32)
        vector=np.r_[coarse,np.log1p(self.dwell)/5,np.eye(10,dtype=np.float32)[list(self.buttons)].ravel()].astype(np.float32)
        if self.version>=2:
            visit=self.visits[obs]
            self.visits[obs]+=1
            key+=struct.pack("<I",visit)
            vector=np.r_[vector,np.float32(np.log1p(visit)/5)]
        return key,pixels,vector


class Clone(nn.Module):
    def __init__(self,nfeatures=210):
        super().__init__()
        self.vision=nn.Sequential(nn.Conv2d(3,16,3,2),nn.SiLU(),nn.Conv2d(16,32,3,2),nn.SiLU(),
                                  nn.Conv2d(32,48,3,2),nn.SiLU(),nn.Flatten())
        self.readout=nn.Sequential(nn.Linear(48*4*6+nfeatures,256),nn.LayerNorm(256),nn.SiLU(),
                                   nn.Linear(256,128),nn.SiLU(),nn.Linear(128,9))

    def forward(self,images,history):
        emb=self.vision(images.permute(0,3,1,2).float()/255.)
        return self.readout(torch.cat([emb,history],1))


def record(tape,version=1):
    game=Game()
    history=History(version)
    lookup=defaultdict(Counter)
    images,features=[],[]
    try:
        for button in tape:
            key,image,vec=history.input(game)
            lookup[key][button]+=1
            images.append(image)
            features.append(vec)
            history.buttons.append(button)
            game.step(button)
        st=game.state()
        return lookup, np.stack(images),np.stack(features),st
    finally: game.env.close()


def fit(images,features,tape,steps,refine=0):
    torch.manual_seed(7263)
    net=Clone(features.shape[1]).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=.001)
    x=torch.as_tensor(images,device="cuda")
    v=torch.as_tensor(features,device="cuda")
    y=torch.tensor(list(tape),device="cuda")
    for _ in range(steps):
        idx=torch.randint(len(y),(128,),device="cuda")
        loss=F.cross_entropy(net(x[idx],v[idx]),y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    if refine:
        for group in opt.param_groups:group["lr"]=.00025
        for iteration in range(refine):
            if iteration%40==0:
                with torch.no_grad():
                    logits=torch.cat([net(x[j:j+128],v[j:j+128]) for j in range(0,len(y),128)])
                    actual=logits.gather(1,y[:,None]).squeeze(1)
                    logits.scatter_(1,y[:,None],-torch.inf)
                    margin=actual-logits.max(1).values
                    if (margin>1.).all():break
                    difficult=torch.where(margin<3.)[0]
            idx=torch.cat((torch.randint(len(y),(64,),device="cuda"),difficult[torch.randint(len(difficult),(64,),device="cuda")]))
            opt.zero_grad()
            F.cross_entropy(net(x[idx],v[idx]),y[idx]).backward()
            opt.step()
    net.eval()
    with torch.inference_mode():
        hits=sum(int((net(x[start:start+128],v[start:start+128]).argmax(-1)==y[start:start+128]).sum())
                 for start in range(0,len(y),128))
    return net,hits/len(y)


@torch.inference_mode()
def evaluate(net,policy,folder,limit,delay,memory,stop,version=1):
    game=Game()
    history=History(version)
    first={}
    maps={}
    recalls=0
    try:
        for _ in range(delay):
            game.step(0)
        for n in range(1,limit+1):
            key,image,features=history.input(game)
            if memory and key in policy:
                a=policy[key]
                recalls+=1
            else:
                a=int(net(torch.as_tensor(image[None],device="cuda"),torch.as_tensor(features[None],device="cuda")).argmax())
            history.buttons.append(a)
            game.step(a)
            s=game.state()
            if s and s["phase"] in (5,6):
                maps.setdefault(s["name"],n)
                facts=dict(clock=s["clock"],rival=s["rival"],starter=s["party"]>0,lab=s["lab"]>=3,
                           rival_battle=s["lab"]>=4,dex=s["dex"],rustboro=s["name"]=="RustboroCity",
                           gym=s["name"]=="RustboroCity_Gym",
                           stone_badge=(s["badge"] and s["defeated"] and s["name"]=="RustboroCity_Gym"))
                for event,yes in facts.items():
                    if yes and event not in first:
                        first[event]=n
                        Image.fromarray(game.env.screen).save(folder/(event+".png"))
                if stop in first:break
        (folder/"buttons.bin").write_bytes(bytes(game.trace))
        result=dict(decisions=n,initial_waits=delay,resets=0,lookups=recalls,events=first,maps=maps,
                    final_state=game.state(),tape_sha256=hashlib.sha256(game.trace).hexdigest())
        write_json(folder/"evaluation.json",result)
        print(folder.name,result["events"],"lookups",recalls,"steps",n,flush=True)
        return result
    finally:game.env.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--steps",type=int,default=600)
    p.add_argument("--refine",type=int,default=0)
    p.add_argument("--features",type=int,choices=[1,2],default=2)
    p.add_argument("--stop",default="stone_badge")
    p.add_argument("--delay",type=int,default=0)
    p.add_argument("--pure",action="store_true")
    args=p.parse_args()
    folder=RUN/args.out
    folder.mkdir(parents=True,exist_ok=True)
    tape=(RUN/args.source).read_bytes()
    lookup,x,v,st=record(tape,args.features)
    print("Training route replayed",st,flush=True)
    net,acc=fit(x,v,tape,args.steps,args.refine)
    policy={key:c.most_common(1)[0][0] for key,c in lookup.items()}
    with (folder/"episodic.pkl").open("wb") as f:pickle.dump(policy,f)
    torch.save(net.state_dict(),folder/"clone.pt")
    write_json(folder/"fit.json",dict(source=args.source,training_inputs=len(tape),unique_keys=len(policy),
               conflicting_keys=sum(len(c)>1 for c in lookup.values()),clone_train_accuracy=acc,
               trained_from_own_search=True,feature_version=args.features,source_sha256=hashlib.sha256(tape).hexdigest(),replayed_end=st))
    for label,delay,memory in ([("fresh",0,True),("timing-shift",args.delay,True)] if args.delay else [("fresh",0,True)]):
        child=folder/label
        child.mkdir(exist_ok=True)
        evaluate(net,policy,child,len(tape)+512,delay,memory,args.stop,args.features)
    if args.pure:
        child=folder/"neural-only"
        child.mkdir(exist_ok=True)
        evaluate(net,policy,child,len(tape)+512,0,False,args.stop,args.features)


if __name__=="__main__":
    main()
