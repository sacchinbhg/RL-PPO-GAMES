"""Learn action-conditioned sustained screen changes from original-ROM self-play."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from emerald import BUTTONS, Emerald, REPEAT, ROM_SHA1
from fit_local_motion import fit, hashes, predict
from pixel_env import game_paths
from train_earned_dreamer import Trajectory, digest, write_json

read=lambda p:json.loads(p.read_text())


def fraction(a,b):
    return np.mean(np.max(np.abs(a.astype(np.int16)-b),axis=-1)>=24)


def examples(dataset, folder, skip, limit, stride):
    source=Trajectory(dataset)
    budget=min(limit,len(source.tape))
    folder.mkdir(parents=True,exist_ok=True)
    config=dict(tape=source.source["source_sha256"],skip=skip,budget=budget,stride=stride,
                required_quiet_decisions=4,quiet_pixel_fraction=.02,action_fork="one button followed by three waits",
                rom_sha1=ROM_SHA1,frames_per_decision=REPEAT,actions=list(BUTTONS))
    if (folder/"registration.json").exists() and read(folder/"registration.json")!=config:
        raise ValueError("Existing effect collection has different settings")
    write_json(folder/"registration.json",config)
    file=folder/"examples.npz"
    if (folder/"complete.json").exists():
        assert read(folder/"complete.json")["sha256"]==digest(file)
        return
    samples=[]
    labels=[]
    times=[]
    seen=set()
    game=Emerald(*game_paths())
    try:
        game.load(game.boot)
        frame=game.boot_image
        quiet=0
        next_at=skip
        matched=0
        for n,(recorded,button,_,_,_,_) in enumerate(source.episode()):
            if n>budget:
                break
            if n==0 or n%2048==0:
                write_json(folder/"status.json",dict(replayed=n,positions=len(samples),matched=matched,expected=budget))
            if n>=next_at and quiet>=4:
                key=hashlib.sha256(frame).digest()
                if key not in seen and np.array_equal(frame,recorded):
                    seen.add(key)
                    matched+=1
                    snapshot=game.save()
                    outcomes=[]
                    for candidate in range(len(BUTTONS)):
                        game.load(snapshot)
                        trace=[]
                        for action in (candidate,0,0,0):
                            image,_=game.step(action)
                            trace.append(image)
                        if candidate==0:
                            waiting=trace
                        outcomes.append(all(fraction(a,b)>=.02 for a,b in zip(trace[-2:],waiting[-2:])))
                    game.load(snapshot)
                    samples.append(frame.copy())
                    labels.append(outcomes)
                    times.append(n)
                    next_at=n+stride
            if n==budget:
                break
            before=frame
            frame,_=game.step(button)
            quiet=quiet+1 if fraction(frame,before)<.02 else 0
    finally:
        game.close()
    if not samples:
        raise ValueError("No recorded images exactly matched eligible original screens")
    temp=folder/"examples.tmp.npz"
    np.savez_compressed(temp,rgb=np.stack(samples),effects=np.array(labels),steps=np.array(times))
    temp.replace(file)
    write_json(folder/"complete.json",dict(**config,sha256=digest(file),positions=len(times),
               action_sensitive_positions=sum(any(y[1:]) and not all(y[1:]) for y in labels),
               alternative_decisions=len(times)*len(BUTTONS)*4,recorded_rgb_match_required=True))


def evaluate(out):
    data=[]
    for name in ("train","heldout"):
        folder=out/name
        meta=read(folder/"complete.json")
        assert digest(folder/"examples.npz")==meta["sha256"]
        with np.load(folder/"examples.npz",allow_pickle=False) as bundle:
            data.append((bundle["rgb"],bundle["effects"][:,1:].astype(np.float32),meta))
    (train,labels,a),(test,truth,b)=data
    assert a["tape"]!=b["tape"] and b["skip"]>=read(out/"registration.json")["common_prefix"]
    coords,loss=fit(train,labels)
    keep=~np.isin(hashes(test),hashes(train))
    sensitive=keep & (truth.min(1)!=truth.max(1))
    if not sensitive.any():
        raise ValueError("No held-out positions with different button effects")
    reports={}
    for name,scores in predict(train,labels,test,coords).items():
        tied=np.isclose(scores,scores.max(1,keepdims=True),atol=1e-7,rtol=0)
        measured=(truth*tied).sum(1)/tied.sum(1)
        reports[name]=dict(effect_probability=float(measured[sensitive].mean()),
                           mean_squared_error=float(((scores[keep]-truth[keep])**2).mean()))
    chance=float(truth[sensitive].mean())
    qualifies=int(sensitive.sum())>=25 and reports["local_screen"]["effect_probability"]>max(
        chance,reports["whole_screen"]["effect_probability"],reports["button_frequency"]["effect_probability"])
    dest=out/"model.npz"
    np.savez_compressed(dest,train=train,outcomes=labels,patches=coords,buttons=np.arange(1,len(BUTTONS)))
    write_json(out/"result.json",dict(reports=reports,uniform=chance,novel_positions=int(keep.sum()),sensitive=int(sensitive.sum()),
               model_sha256=digest(dest),selected_patches_using_training_only=True,passes_comparison=bool(qualifies),
               heldout_stalled_policy_footage_used=False))


def compare(training, heldout, output, stride=160):
    if stride<1:
        raise ValueError("Sampling stride must be positive")
    a,b=Trajectory(training),Trajectory(heldout)
    if a.source["source_sha256"]==b.source["source_sha256"]:
        raise ValueError("Expected separate training and held-out games")
    common=next((i for i,(x,y) in enumerate(zip(a.tape,b.tape)) if x!=y),min(len(a.tape),len(b.tape)))
    if common==len(b.tape):
        raise ValueError("The held-out game has no distinct suffix")
    output.mkdir(parents=True,exist_ok=True)
    plan=dict(train=a.source["source_sha256"],heldout=b.source["source_sha256"],common_prefix=common,
              effect="minimum 2% RGB pixel change relative to waiting after both 3 and 4 decisions",
              validation="exclude shared action prefix and exact training screens; compare uniform, training button frequency and whole-screen retrieval",
              deployment="at least 25 unseen action-sensitive positions and strictly better than all three baselines",
              observations="current RGB and candidate button only",frozen_policy_examples_used=False)
    registration=output/"registration.json"
    if registration.exists() and read(registration)!=plan:
        raise ValueError("Another effect experiment owns this output directory")
    write_json(registration,plan)
    examples(training,output/"train",0,len(a.tape),stride)
    examples(heldout,output/"heldout",common+1,len(b.tape),stride)
    evaluate(output)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("training","heldout","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--stride",type=int,default=160)
    args=parser.parse_args()
    compare(args.training,args.heldout,args.output,args.stride)
