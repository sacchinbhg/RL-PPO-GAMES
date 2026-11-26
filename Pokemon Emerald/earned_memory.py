"""A frozen self-imitation controller that checks predicted screens against the live emulator."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from train_earned_dreamer import Trajectory, digest, write_json


def encode(frame):
    exact=int.from_bytes(hashlib.blake2b(frame.tobytes(),digest_size=8).digest(),"little")
    small=(frame.reshape(8,5,10,6,3).sum(axis=(1,3),dtype=np.uint16)//30).astype(np.uint8)
    return exact,small.ravel()


def fit(datasets, output):
    games=[Trajectory(d) for d in datasets]
    tapes=[g.source["source_sha256"] for g in games]
    marker=output/"fit.json"
    output.mkdir(parents=True,exist_ok=True)
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved["self_earned_tapes"]!=tapes or saved["model_sha256"]!=digest(output/"model.npz"):
            raise ValueError("Another memory model occupies this directory")
        return
    arrays={}
    for j,game in enumerate(games):
        keys,features,buttons,progress,rewards=[],[],[],[],[]
        phase=0
        for pic,button,incoming,first,last,terminal in game.episode():
            phase+=int(incoming>=.25)
            key,feature=encode(pic)
            keys.append(key)
            features.append(feature)
            buttons.append(button)
            progress.append(phase)
            rewards.append(incoming)
        # Value of each observed continuation. The last state has no outgoing recorded action.
        returns=np.zeros(len(rewards),np.float32)
        for t in range(len(rewards)-2,-1,-1):
            returns[t]=rewards[t+1]+.9995*returns[t+1]
        arrays.update({f"{j}_keys":np.array(keys,np.uint64),f"{j}_features":np.stack(features),
                       f"{j}_buttons":np.array(buttons,np.uint8),f"{j}_phase":np.array(progress,np.uint16),
                       f"{j}_value":returns,f"{j}_total":np.array(sum(rewards),np.float32)})
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        path=Path(tmp)/"model.npz"
        np.savez_compressed(path,**arrays)
        path.replace(output/"model.npz")
    write_json(marker,dict(model_sha256=digest(output/"model.npz"),self_earned_tapes=tapes,
        training_decisions=[len(t.tape) for t in games],self_imitation=True,frozen_evaluation_data_used=False,
        inputs="RGB, incoming scalar reward and previously executed action",
        method="match observed screens with learned action-conditioned transitions and select a known continuation"))


class Memory:
    def __init__(self, directory, seed):
        self.seed=seed
        self.rng=np.random.default_rng(seed)
        self.identity=json.loads((directory/"fit.json").read_text())
        if self.identity["model_sha256"]!=digest(directory/"model.npz"):
            raise ValueError("Memory model checksum mismatch")
        self.traces=[]
        self.index=defaultdict(list)
        with np.load(directory/"model.npz",allow_pickle=False) as data:
            for j,_ in enumerate(self.identity["self_earned_tapes"]):
                trace={k:data[f"{j}_{k}"] for k in ("keys","features","buttons","phase","value","total")}
                self.traces.append(trace)
                for t in range(len(trace["buttons"])-1):
                    self.index[(int(trace["phase"][t]),int(trace["keys"][t]))].append((j,t))
        self.current=None
        self.expected=None
        self.phase=0
        self.usage=Counter()

    def choose(self, frame, incoming):
        self.phase+=int(incoming>=.25)
        exact,feature=encode(frame)
        def usable(j,t):
            trace=self.traces[j]
            return 0<=t<len(trace["buttons"])-1 and trace["phase"][t]==self.phase
        location=None
        if self.expected is not None:
            j,t=self.expected
            trace=self.traces[j]
            if usable(j,t) and trace["keys"][t]==exact:
                location=(j,t)
                mode="confirmed_next_screen"
            else:
                nearby=[k for k in range(t-4,t+7) if usable(j,k)]
                matches=[k for k in nearby if trace["keys"][k]==exact]
                if matches:
                    location=(j,min(matches,key=lambda k:abs(k-t)))
                    mode="corrected_with_local_screen"
        if location is None:
            matches=self.index.get((self.phase,exact),())
            if matches:
                def rank(sample):
                    j,t=sample
                    trace=self.traces[j]
                    if self.expected is None:
                        return (t==0,float(trace["total"]),float(trace["value"][t]),-t)
                    prior,pos=self.expected
                    return (j==prior,-abs(t-pos) if j==prior else 0,float(trace["value"][t]),-t)
                location=max(matches,key=rank)
                mode="retrieved_matching_screen"
        if location is None and self.expected is not None:
            j,t=self.expected
            trace=self.traces[j]
            nearby=[k for k in range(t-4,t+7) if usable(j,k)]
            if nearby:
                residual=np.abs(trace["features"][nearby].astype(np.int16)-feature).mean(axis=1)
                cost=residual+.1*np.abs(np.array(nearby)-t)
                best=int(np.argmin(cost))
                if residual[best]<=1.5:
                    location=(j,nearby[best])
                    mode="close_visual_prediction"
        self.current=location
        if location is None:
            self.usage["unknown_screen"]+=1
            return int(self.rng.integers(9))
        self.usage[mode]+=1
        j,t=location
        return int(self.traces[j]["buttons"][t])

    def executed(self, button):
        if self.current is not None:
            j,t=self.current
            # Advance only across a transition whose actual action agrees with the learned model.
            self.expected=(j,t+int(button==self.traces[j]["buttons"][t]))


def play(directory, output, seed, limit, delay):
    from pixel_env import PixelPool
    import subprocess
    import sys
    if limit<1 or delay<0:
        raise ValueError("Expected positive evaluation length and nonnegative perturbation interval")
    output.mkdir(parents=True,exist_ok=True)
    agent=Memory(directory,seed)
    config=dict(model_sha256=agent.identity["model_sha256"],seed=seed,button_budget=limit,
                external_wait_interval=delay,training_updates=0,in_game_restores=0,
                controller="frozen screen-conditioned self-imitation memory")
    setup=output/"provenance.json"
    if setup.exists():
        if json.loads(setup.read_text())!=config:
            raise ValueError("Evaluation directory belongs to another policy")
        if (output/"result.json").exists():
            return
        raise FileExistsError("Choose a new directory for a fresh attempt")
    write_json(setup,config)
    env=PixelPool(1,map_audits=output/"first-maps")
    try:
        frame=env.command(0,"reset")
        reward=0.
        interventions=0
        for n in range(1,limit+1):
            proposed=agent.choose(frame,reward)
            actual=0 if delay and n%delay==0 else proposed
            interventions+=int(actual!=proposed)
            agent.executed(actual)
            frame,reward,done=env.command(0,"act",actual)
            if n%8192==0 or reward>=1 or n==limit or done:
                dest=output/f"step-{n:07d}"
                env.command(0,"audit",str(dest))
            if n%1024==0 or reward>=1 or n==limit or done:
                status=dict(decisions=n,action_sources=dict(agent.usage),changed_actions=interventions)
                write_json(output/"status.json",status)
            if done:
                audit,=dest.glob("*.json")
                subprocess.run([sys.executable,str(Path(__file__).with_name("pixel_replay.py")),str(audit),
                                str(output/"badge-replay"),"--event","stone_badge","--video"],check=True)
                break
        write_json(output/"result.json",dict(**config,**status,verified_badge=(output/"badge-replay/verification.json").exists()))
    finally:
        env.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",action="append",required=True,type=Path)
    parser.add_argument("--model",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--seed",type=int,default=197461)
    parser.add_argument("--limit",type=int,default=65536)
    parser.add_argument("--wait-every",type=int,default=0,help="Insert generic wait actions in a separate robustness trial")
    args=parser.parse_args()
    fit(args.dataset,args.model)
    play(args.model,args.output,args.seed,args.limit,args.wait_every)
