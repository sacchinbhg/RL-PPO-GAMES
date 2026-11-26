"""Keep every fixed-length training attempt from a self-earned prefix, including unsuccessful ones."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from reward_branch_env import ForkEnv
from reward_branch_search import visual_code,write_json
from screen_explorer import ScreenExplorer
from train_earned_dreamer import Trajectory

def read(file):
    return json.loads(file.read_text())

def collect(data,folder,seed,branches,steps,inherited):
    source=Trajectory(data)
    manifest=source.source
    if not source.audit["uncapped_exploration"] or manifest.get("source_training_forks",0)<1:
        raise ValueError("Expected a self-earned, independently replayed training discovery")
    minimum=manifest["search_buttons_at_discovery"]
    inherited=minimum if inherited is None else inherited
    if min(branches,steps)<1 or inherited<minimum:
        raise ValueError("Invalid branch or inherited training budget")
    folder.mkdir(parents=True,exist_ok=True)
    settings=dict(seed=seed,training_forks=True,receives_saves_from_other_runs=False,
        observation="RGB, scalar reward, episode end",source_sha256=manifest["source_sha256"],
        source_decisions=len(source.tape),inherited_training_buttons=inherited,
        branch_count=branches,buttons_per_branch=steps,selection="every attempt, in seed order",
        behavior="online screen-action transition counts and scalar reward; independent seed per branch")
    if (folder/"settings.json").exists() and read(folder/"settings.json")!=settings:
        raise ValueError("Different collection already owns this directory")
    write_json(folder/"settings.json",settings)
    if (folder/"completed.json").exists():
        return
    runs=folder/"discoveries"
    runs.mkdir(exist_ok=True)
    with np.load(sorted(data.glob("chunk-*.npz"))[-1]) as final:
        expected=final["image"][-1].copy()
    env=ForkEnv(folder/"first-maps",log_whiteouts=True)
    try:
        total=0
        for i,button in enumerate(source.tape,1):
            image,cents,terminal,_=env.command("step",(button,1))
            total+=cents
            if terminal:
                raise ValueError("Cannot branch past the completed game")
            if i%4096==0:
                write_json(folder/"initialization.json",dict(cold_replay_buttons=i,expected=len(source.tape)))
        if total!=manifest["return_cents"] or not np.array_equal(image,expected):
            raise ValueError("Prefix reward or screen disagrees with the verified original game")
        env.command("save",0)
        write_json(folder/"initialization.json",dict(cold_replay_buttons=len(source.tape),matched_image_and_reward=True))
        records=[]
        best=None
        for index in range(branches):
            target=runs/f"branch-{index:03d}"
            if target.exists():
                record=read(target/"record.json")
                if record["index"]!=index or hashlib.sha256((target/"game.bin").read_bytes()).hexdigest()!=record["tape_sha256"]:
                    raise ValueError("Previously saved branch changed")
            else:
                image=env.command("restore",0)
                if not np.array_equal(image,expected):
                    raise ValueError("Training restore changed the starting observation")
                model=ScreenExplorer(np.random.SeedSequence([seed,index]))
                duration_rng=np.random.default_rng([seed,index,144])
                state=visual_code(image)
                used=earned=0
                terminal=False
                while used<steps and not terminal:
                    button=model.choose(state)
                    duration=min(steps-used,int(duration_rng.choice((1,1,2,2,4,8))))
                    image,cents,terminal,n=env.command("step",(button,duration))
                    following=visual_code(image)
                    model.observe(state,button,following,cents/100)
                    state=following
                    used+=n
                    earned+=cents
                with tempfile.TemporaryDirectory(dir=runs) as temp:
                    draft=Path(temp)/"sealed"
                    env.command("audit",str(draft))
                    audit=read(draft/"game.json")
                    tape=(draft/"game.bin").read_bytes()
                    if (not tape.startswith(source.tape) or len(tape)!=len(source.tape)+used
                            or audit["steps"]!=len(tape) or audit["return_cents"]!=total+earned):
                        raise ValueError("Training game, scalar return and recorded inputs disagree")
                    record=dict(index=index,work=inherited+sum(r["branch_buttons"] for r in records)+used,
                        branches=manifest["source_training_forks"]+index+1,branch_buttons=used,
                        length=len(tape),return_cents=total+earned,screen=state.hex(),audit=str(target.relative_to(folder)),
                        tape_sha256=hashlib.sha256(tape).hexdigest(),training_only=True,terminal=terminal)
                    write_json(draft/"record.json",record)
                    draft.rename(target)
            records.append(record)
            if best is None or record["return_cents"]>best["return_cents"]:
                best=record
                write_json(folder/"best.json",best)
            write_json(folder/"status.json",dict(work=record["work"],finished_attempts=len(records),
                       highest_return_cents=best["return_cents"],fixed_new_buttons=branches*steps))
            if record["terminal"]:
                break
        write_json(folder/"completed.json",dict(settings=settings,records=records,best=best,
                   cold_replay_buttons=len(source.tape),new_emulator_buttons=sum(r["branch_buttons"] for r in records)))
    finally:
        env.close()

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--seed",required=True,type=int)
    p.add_argument("--branches",default=8,type=int)
    p.add_argument("--steps-per-branch",default=8192,type=int)
    p.add_argument("--inherited-search-buttons",type=int)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/"job.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        collect(a.dataset.resolve(),a.output.resolve(),a.seed,a.branches,a.steps_per_branch,a.inherited_search_buttons)
