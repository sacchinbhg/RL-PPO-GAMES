"""Online latent dynamics, real-replay value learning and imagined REINFORCE for a pixel-only agent."""
import argparse
from collections import deque
from datetime import datetime,timezone
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
from gym_dream import Dream, MEMORY, WIDTH, logits_kl
from pixel_env import CLEAN, PixelPool

BINS=(0.,.02,.25,1.,2.,3.,4.,6.,10.,20.,32.)


class PixelDreamer(Dream):
    def __init__(self):
        super().__init__()
        del self.events
        self.reward=nn.Linear(WIDTH,len(BINS))
        self.continue_head=nn.Linear(WIDTH,1)
        self.register_buffer("bins",torch.tensor(BINS))
        # An untrained predictor has no evidence of high rewards.
        with torch.no_grad():
            self.reward.bias.fill_(-4.)
            self.reward.bias[0]=3.
            self.continue_head.bias.fill_(3.)

    def imagine_clean(self,start,horizon):
        state=start.detach()
        logp,entropy,values,rewards,cont=[],[],[],[],[]
        for _ in range(horizon):
            dist=Categorical(logits=self.actor(state))
            a=dist.sample()
            logp.append(dist.log_prob(a))
            entropy.append(dist.entropy())
            values.append(self.value(state).squeeze(-1))
            with torch.no_grad():
                h=self.recur(torch.cat([state[:,MEMORY:],F.one_hot(a,9).float()],-1),state[:,:MEMORY])
                state=torch.cat([h,self.latent(self.prior(h),True)],-1)
                rewards.append(self.reward(state).softmax(-1)@self.bins)
                cont.append(self.continue_head(state).squeeze(-1).sigmoid()*.995)
        tail=self.value(state).squeeze(-1).detach()
        targets=[]
        for t in reversed(range(horizon)):
            next_v=values[t+1].detach() if t+1<horizon else tail
            tail=rewards[t]+cont[t]*(.95*tail+.05*next_v)
            targets.append(tail)
        target=torch.stack(targets[::-1])
        v=torch.stack(values)
        adv=target-v.detach()
        return -(torch.stack(logp)*adv.detach()/adv.std().clamp_min(.5)).mean()-.012*torch.stack(entropy).mean()+F.smooth_l1_loss(v,target)


def small(images):
    return np.ascontiguousarray(images[...,::2,::2,:])


def window_segments(stream, dest, replay, segment_id):
    # Adjacent segments share a 16-step context. No sequence crosses a reset.
    x=np.asarray(stream[0],dtype=np.uint8)
    a=np.asarray(stream[1],dtype=np.uint8)
    r=np.asarray(stream[2],dtype=np.float32)
    term=np.asarray(stream[3],dtype=np.uint8)
    if len(a)<16:return
    entry=(x,a,r,term)
    replay.add(entry)
    np.savez_compressed(dest/f"{segment_id:09}.npz",x=x,a=a,r=r,term=term)


class Replay:
    def __init__(self):
        self.recent=deque(maxlen=160)
        self.rare=deque(maxlen=180)

    def add(self,entry):
        self.recent.append(entry)
        if entry[2].max()>=.99:self.rare.append(entry)

    def batch(self,batch,size,rng):
        chunks=[[],[],[],[]]
        for b in range(batch):
            rare=self.rare and b<batch//4
            buf=self.rare if rare else self.recent
            ep=buf[rng.integers(len(buf))]
            end=len(ep[1])-size
            if rare:
                hits=np.flatnonzero(ep[2]>=.99)
                pos=int(np.clip(rng.choice(hits)-rng.integers(1,size),0,end))
            else:pos=rng.integers(end+1)
            for bucket,array,extra in zip(chunks,ep,(1,0,0,0)):
                bucket.append(array[pos:pos+size+extra])
        x,a,r,t=[torch.as_tensor(np.stack(c),device="cuda") for c in chunks]
        return x.float()/255.,a.long(),r,t.float()


def update(net,opt,replay,rng,imag):
    x,a,r,t=replay.batch(32,16,rng)
    state,future,post,prior=net.observed(x,a,True)
    pred=future[:,1:]
    rec=F.mse_loss(net.decoder(state).sigmoid(),x[:,:,::2,::2].flatten(2))
    high=torch.searchsorted(net.bins,r.contiguous()).clamp(1,len(BINS)-1)
    low=high-1
    mix=((r-net.bins[low])/(net.bins[high]-net.bins[low])).clamp(0,1)
    logits=net.reward(pred).log_softmax(-1)
    rw=-(logits.gather(-1,low.unsqueeze(-1)).squeeze(-1)*(1-mix)+logits.gather(-1,high.unsqueeze(-1)).squeeze(-1)*mix).mean()
    cv=F.binary_cross_entropy_with_logits(net.continue_head(pred).squeeze(-1),1-t)
    val=net.value(state.detach()).squeeze(-1)
    tail=val[:,-1].detach()
    targets=[]
    for i in reversed(range(a.shape[1])):
        tail=r[:,i]+.995*(1-t[:,i])*(.95*tail+.05*val[:,i+1].detach())
        targets.append(tail)
    target=torch.stack(targets[::-1],1)
    real_value=F.smooth_l1_loss(val[:,:-1],target)
    dist=Categorical(logits=net.actor(state[:,:-1].detach()))
    adv=(target-val[:,:-1].detach()).clamp(-4,4)
    # Weak self-imitation of own actions with positive observed advantages.
    imitation=-(dist.log_prob(a)*adv.clamp_min(0)).mean()
    iloss=net.imagine_clean(state[:,::4].flatten(0,1),10) if imag else torch.zeros((),device="cuda")
    loss=8*rec+rw+.2*cv+.1*logits_kl(post.detach(),prior)+.025*logits_kl(post,prior.detach())+.4*real_value+.15*imitation+.4*iloss
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(),10)
    opt.step()
    return [float(v.detach()) for v in (loss,rec,rw,iloss)]


def atomic_json(path,value):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(value,indent=2),encoding="utf-8")
    os.replace(temp,path)


def train(args):
    root=CLEAN/args.name
    for d in (root,root/"replay",root/"audits",root/"checkpoints"):d.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    rng=np.random.default_rng(args.seed)
    model=PixelDreamer().cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    replay=Replay()
    n=args.envs
    tick=updates=seg=0
    best=0.
    latest=root/"checkpoints/latest.pt"
    if latest.exists():
        ck=torch.load(latest,map_location="cuda",weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        tick,updates,seg,best=ck["tick"],ck["updates"],ck["segment"],ck["best"]
        for f in sorted((root/"replay").glob("*.npz"))[-350:]:
            with np.load(f) as d:replay.add(tuple(d[k] for k in ("x","a","r","term")))
        # Never overwrite segments or episodes from an interrupted collection.
        seg=max([seg]+[int(p.stem) for p in (root/"replay").glob("*.npz")])
    session=f"{time.time_ns()}"
    auditdir=root/"audits"/session
    atomic_json(root/"protocol.json",dict(seed=args.seed,actions=9,backend="mGBA original ROM",acting_inputs=["current RGB","recurrent state from own action and RGB history"],
        training_feedback=["scalar game reward","true terminal"],initial_weights="random, or this clean run's own checkpoint on restart",
        fresh_boot_only=True,archive_restores=False,uses_demonstrations=False,uses_assisted_data=False,imagination=args.imagine,
        model="compact categorical recurrent state-space model, real-replay value learning, 10 imagined actions and REINFORCE"))
    pool=PixelPool(n)
    x=small(pool.initial)
    streams=[([x[i]],[],[],[]) for i in range(n)]
    epsteps=np.zeros(n,np.int64)
    returns=np.zeros(n)
    h=torch.zeros(n,MEMORY,device="cuda")
    last=np.zeros(n,np.int64)
    lastloss=None
    start=time.monotonic()
    starttick=tick
    def save():
        temp=latest.with_suffix(".tmp")
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),tick=tick,updates=updates,segment=seg,best=best),temp)
        os.replace(temp,latest)
    print("Clean training started",session,torch.cuda.get_device_name(),flush=True)
    try:
        while tick*n<args.budget:
            with torch.no_grad():
                obs=torch.as_tensor(x,device="cuda").float().flatten(1)/255.
                z=model.latent(model.posterior(torch.cat([h,model.encoder(obs)],-1)),True)
                dist=Categorical(logits=model.actor(torch.cat([h,z],-1)))
                a=dist.sample().cpu().numpy()
                explore=rng.random(n)<.18
                a[explore]=rng.integers(9,size=explore.sum())
                repeat=rng.random(n)<.12
                a[repeat]=last[repeat]
                h=model.recur(torch.cat([z,F.one_hot(torch.as_tensor(a,device="cuda"),9).float()],-1),h)
            frame,reward,ends=pool.step(a)
            x=small(frame)
            epsteps+=1
            returns+=reward
            tick+=1
            last=a
            for i in range(n):
                stream=streams[i]
                stream[0].append(x[i])
                for seq,v in zip(stream[1:],(a[i],reward[i],ends[i])):seq.append(v)
                stop=ends[i] or epsteps[i]>=args.episode
                if len(stream[1])>=192 or stop:
                    seg+=1
                    window_segments(stream,root/"replay",replay,seg)
                    streams[i]=(stream[0][-17:],stream[1][-16:],stream[2][-16:],stream[3][-16:])
                if stop:
                    pool.command(i,"audit",str(auditdir))
                    if returns[i]>best:
                        best=float(returns[i])
                        torch.save(model.state_dict(),root/"checkpoints/best-training-return.pt")
                    if ends[i]:
                        print("Real training episode reached true badge terminal",tick*n,flush=True)
                        save()
                        torch.save(model.state_dict(),root/"checkpoints/training-badge.pt")
                    x[i]=small(pool.command(i,"reset"))
                    streams[i]=([x[i]],[],[],[])
                    epsteps[i]=0
                    returns[i]=0
                    h[i]=0
                    last[i]=0
            if replay.recent and tick%8==0:
                lastloss=update(model,opt,replay,rng,args.imagine)
                updates+=1
            if tick%128==0:
                metrics=dict(real_decisions=tick*n,model_updates=updates,
                    imagined_transitions=updates*32*4*10 if args.imagine else 0,
                    decisions_per_second=round((tick-starttick)*n/max(1,time.monotonic()-start),1),
                    current_returns=returns.round(2).tolist(),best_completed_return=best,
                    losses=lastloss,updated_utc=datetime.now(timezone.utc).isoformat())
                atomic_json(root/"status.json",metrics)
                with (root/"metrics.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(metrics)+"\n")
                if tick%1024==0:print(metrics,flush=True)
            if tick%1024==0:save()
            if tick%8192==0:
                torch.save(model.state_dict(),root/"checkpoints"/f"step-{tick*n:010}.pt")
                # Bound disk growth. Keep recent real experience and archive milestone-rich chunks.
                files=sorted((root/"replay").glob("*.npz"))
                for f in files[:-600]:
                    if int(f.stem)%128:
                        with np.load(f) as d:keep=d["r"].max()>=2.99
                        if not keep:f.unlink()
        save()
    finally:pool.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--name",default="dream-seed-71299")
    parser.add_argument("--seed",type=int,default=71299)
    parser.add_argument("--envs",type=int,default=8)
    parser.add_argument("--episode",type=int,default=2048)
    parser.add_argument("--budget",type=int,default=500_000_000)
    parser.add_argument("--imagine",action=argparse.BooleanOptionalAction,default=True)
    train(parser.parse_args())


if __name__=="__main__":
    main()
