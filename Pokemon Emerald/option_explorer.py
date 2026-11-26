"""Learn the consequences of repeating each button, using the explorer's own pixels and rewards."""
import argparse
from collections import Counter, deque
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from reward_branch_env import ForkEnv
from reward_branch_search import Archive, Point, write_json


DURATIONS=(1,2,4,8,16)


def initialize(source, folder, seed):
    """Use the source run's own highest recorded scalar-return discovery."""
    if folder.exists():
        record=json.loads((folder/"branch-origin.json").read_text())
        if record.get("source_run")!=str(source) or record["seed"]!=seed:
            raise ValueError("The existing branch has a different origin")
        return
    settings=json.loads((source/"settings.json").read_text())
    best=json.loads((source/"best.json").read_text())
    audit=(source/best["audit"]).resolve()
    if (not settings["training_forks"] or settings["receives_saves_from_other_runs"] or not best["training_only"]
            or not audit.is_relative_to((source/"discoveries").resolve())
            or hashlib.sha256((audit/"game.bin").read_bytes()).hexdigest()!=best["tape_sha256"]):
        raise ValueError("Expected an authentic self-generated training discovery")
    live=json.loads((source/"status.json").read_text()) if (source/"status.json").exists() else {}
    inherited=max(best["work"],live.get("work",0))
    relative=f"discoveries/work-{inherited:010d}"
    start=dict(best,work=inherited,branches=max(best["branches"],live.get("branches",0)),audit=relative)
    folder.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder.parent) as tmp:
        stage=Path(tmp)/"arm"
        shutil.copytree(audit,stage/relative)
        write_json(stage/relative/"record.json",start)
        write_json(stage/"best.json",start)
        write_json(stage/"branch-origin.json",dict(parent_record=best,source_run=str(source),seed=seed,
                   inherited_training_buttons=inherited,cold_replay_buttons=len((audit/"game.bin").read_bytes())))
        stage.rename(folder)


class Options:
    def __init__(self, rng):
        self.rng=rng
        self.visits=Counter()
        self.context={}
        self.edges={}
        self.prior={}
        self.chosen=Counter()

    def screen(self, pic):
        grid=(pic.reshape(4,10,6,10,3).sum(axis=(1,3),dtype=np.uint16)//6400).astype(np.uint8)
        key=hashlib.blake2b(grid.tobytes(),digest_size=12).digest()
        if key not in self.context:
            coarse=grid.reshape(2,2,3,2,3).sum(axis=(1,3))//4
            self.context[key]=coarse.astype(np.uint8).tobytes()
        return key

    def record(self, state, button, duration, following, cents):
        option=(button,duration)
        entries=self.edges.setdefault(state,{})
        if option not in entries:
            entries[option]=[Counter(),0.]
        outcomes, earned=entries[option]
        outcomes[following]+=1
        value=math.tanh(cents/100)/math.sqrt(duration)
        entries[option][1]=earned+value
        prior=self.prior.setdefault((self.context[state],option),[0,0.,0.])
        prior[0]+=1
        prior[1]+=value
        prior[2]+=int(following!=state)

    def choose(self, state, remaining):
        self.visits[state]+=1
        entries=self.edges.get(state,{})
        scores={}
        for button in range(9):
            for duration in DURATIONS:
                if duration>remaining:
                    continue
                option=(button,duration)
                samples,reward,changed=self.prior.get((self.context[state],option),(0,0.,0.))
                prior=(reward + .2*changed + 1.)/(samples+2.)
                outcomes,earned=entries.get(option,({},0.))
                count=sum(outcomes.values())
                novelty=sum(n/math.sqrt(1+self.visits[k]) for k,n in outcomes.items())
                scores[option]=(earned+novelty+2*prior)/(count+2)+.3/math.sqrt(count+1)
        best=max(scores.values())
        ties=[a for a,s in scores.items() if s==best]
        chosen=ties[int(self.rng.integers(len(ties)))]
        self.chosen[chosen[1]]+=1
        return chosen


def run(folder, budget, motion_probe=None, stage_reservoir=0, proposal=None, systematic=False):
    origin=json.loads((folder/"branch-origin.json").read_text())
    settings=dict(seed=origin["seed"],training_forks=True,receives_saves_from_other_runs=False,
                  observation="RGB, scalar reward, episode end", root="self-earned cold-boot action history",
                  durations=list(DURATIONS),behavior="empirical button-duration transitions with visual backoff")
    if motion_probe:
        settings["visual_motion_source"]=json.loads(motion_probe.read_text())["training_tape_sha256"]
    if stage_reservoir:
        settings["stage_reservoir"]=stage_reservoir
    if systematic:
        if not stage_reservoir:
            raise ValueError("Systematic option trials require at least one visual landmark")
        settings["systematic_landmark_trials"]=True
    if proposal:
        settings["proposal"]=proposal["kind"]
        settings["proposal_probability_per_option"]=.5
        if proposal["kind"]=="dreamer":
            from train_earned_dreamer import digest
            settings["proposal_weight_sha256"]=digest(proposal["checkpoint"]/"agent.pkl")
    setup=folder/"settings.json"
    if setup.exists() and json.loads(setup.read_text())!=settings:
        raise ValueError("A different experiment owns this directory")
    write_json(setup,settings)
    old=json.loads((folder/"best.json").read_text())
    tape=(folder/old["audit"]/"game.bin").read_bytes()
    if not old["training_only"] or hashlib.sha256(tape).hexdigest()!=old["tape_sha256"]:
        raise ValueError("Unverified training ancestry")
    previous=json.loads((folder/"status.json").read_text()) if (folder/"status.json").exists() else {}
    work=max(old["work"],previous.get("work",0))
    branches=max(old["branches"],previous.get("branches",0))
    if work>=budget or old["terminal"]:
        return
    initial_work=work
    rng=np.random.default_rng([origin["seed"],work])
    model=Options(rng)
    guide=None
    if motion_probe:
        from visual_motion import MotionGuide
        guide=MotionGuide(motion_probe,rng)
        if guide.source!=origin["parent_record"]["tape_sha256"]:
            raise ValueError("Motion model must come from the same self-earned branch")
    actor=None
    if proposal and proposal["kind"]=="dreamer":
        from play_frozen_dreamer import load_actor
        actor=load_actor(proposal["checkpoint"],proposal["parent"],origin["seed"],proposal["upstream"],
                         batch_size=1,with_forks=True)
    env=ForkEnv(folder/"first-maps")
    proposal_choices=0
    if proposal:
        from fork_policy import ProposalEnv
        gate=np.random.default_rng([origin["seed"],work,919])
        env=ProposalEnv(env,actor,np.random.default_rng([origin["seed"],work,519]))
    try:
        warm_start=time.monotonic()
        archive=Archive(env,rng)
        landmarks=None
        if stage_reservoir:
            from scene_reservoir import SceneReservoir
            landmarks=SceneReservoir(env,np.random.default_rng([origin["seed"],work,817]),stage_reservoir,systematic)
        landmark_stride=16 if systematic else 64
        state=model.screen(env.initial)
        if guide:
            guide.restore(env.initial,state)
        total=stage=0
        recent=deque(maxlen=max(DURATIONS))
        for i,button in enumerate(tape,1):
            pic,cents,done,_=env.command("step",(button,1))
            following=model.screen(pic)
            if guide:
                guide.observe(button,pic,following)
            total+=cents
            model.visits[state]+=1
            if recent and recent[-1][1]!=button:
                recent.clear()
            recent.append((state,button,cents))
            history=list(recent)
            for duration in DURATIONS:
                if duration<=len(history):
                    window=history[-duration:]
                    model.record(window[0][0],button,duration,following,sum(p[2] for p in window))
            state=following
            stage+=int(cents>=100)
            point=Point(state,total,i,pose=guide.pose if guide else None,stage=stage)
            if cents:
                archive.offer(point)
            if landmarks and (i%landmark_stride==0 or cents>=100):
                landmarks.consider(point,pic)
            if i%4096==0:
                write_json(folder/"initialization.json",dict(cold_replay_buttons=i,expected=len(tape)))
        if done or total!=old["return_cents"] or state.hex()!=old["screen"]:
            raise ValueError("Failed to reproduce the inherited training history")
        started=time.monotonic()
        write_json(folder/"initialization.json",dict(cold_replay_buttons=len(tape),elapsed_seconds=started-warm_start,
                    matched_total_and_image=True,source_sha256=old["tape_sha256"]))
        best=last_saved=total
        while work<budget:
            pickup=landmarks.start() if landmarks else None
            if pickup:
                point,restored=pickup
            else:
                point=archive.start()
                restored=archive.restored_image
            state,total,length=point.screen,point.total,point.length
            stage=point.stage
            if guide:
                guide.restore(restored,state,point.pose)
            lengths=[16,32,64] if systematic else [128,256,512,1024,2048]
            stop=min(work+int(rng.choice(lengths)),budget)
            branches+=1
            while work<stop:
                start_state=state
                trial=landmarks.next_option(DURATIONS,stop-work) if systematic else None
                if trial:
                    guided=trial
                elif proposal and gate.random()<.5:
                    guided=env.propose(),1
                    proposal_choices+=1
                else:
                    guided=guide.choose() if guide else None
                button,duration=guided or model.choose(state,stop-work)
                if guided:
                    model.visits[state]+=1
                    model.chosen[duration]+=1
                remaining=duration
                earned=0
                while remaining:
                    pic,cents,terminal,used=env.command("step",(button,1 if guide or proposal else remaining))
                    state=model.screen(pic)
                    if guide:
                        guide.observe(button,pic,state)
                    remaining-=used
                    earned+=cents
                    work,total,length=work+used,total+cents,length+used
                    stage+=int(cents>=100)
                    point=Point(state,total,length,pose=guide.pose if guide else None,stage=stage)
                    if cents:
                        archive.offer(point)
                    if landmarks and (length//landmark_stride!=(length-used)//landmark_stride or cents>=100):
                        landmarks.consider(point,pic)
                    best=max(best,total)
                    if (total==best and total>=last_saved+50) or terminal:
                        relative=f"discoveries/work-{work:010d}"
                        dest=folder/relative
                        env.command("audit",str(dest))
                        sha=hashlib.sha256((dest/"game.bin").read_bytes()).hexdigest()
                        saved=dict(work=work,branches=branches,return_cents=total,length=length,screen=state.hex(),
                                   audit=relative,tape_sha256=sha,training_only=True,terminal=terminal)
                        write_json(dest/"record.json",saved)
                        write_json(folder/"best.json",saved)
                        last_saved=total
                    if terminal:
                        return
                model.record(start_state,button,duration,state,earned)
            archive.offer(point,recent=True)
            status=dict(work=work,branches=branches,best_return_cents=best,
                       latest_path_length=length,slots=len(archive.points),screen_codes=len(model.context),
                       selected_durations=dict(model.chosen),steps_per_second=(work-initial_work)/(time.monotonic()-started))
            if guide:
                status.update(motion_decisions=dict(guide.decisions),confirmed_motion=dict(guide.moved),
                              estimated_places=len(guide.visits))
            if landmarks:
                status.update(landmark_stage=landmarks.stage,landmark_slots=len(landmarks.points),
                              landmark_appearances=len(landmarks.seen),landmark_starts=landmarks.chosen,
                              landmark_start_length_bins=dict(landmarks.starts))
                if systematic:
                    status["untried_state_option_pairs_tested"]=landmarks.distinct_trials
            if proposal:
                status.update(proposals_executed=proposal_choices,neural_observations=env.observed,
                              verified_training_restores=env.restores)
            write_json(folder/"status.json",status)
    finally:
        env.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    parser.add_argument("--from-search",type=Path,help="Initialize a new branch from an agent's own saved best discovery")
    parser.add_argument("--seed",type=int,default=465721)
    budgets=parser.add_mutually_exclusive_group(required=True)
    budgets.add_argument("--steps",type=int,help="Total search decisions including inherited work")
    budgets.add_argument("--additional-steps",type=int,help="Search decisions after the inherited work")
    parser.add_argument("--motion-probe",type=Path)
    parser.add_argument("--stage-reservoir",type=int,default=0,
                        help="Keep this many visual landmarks at the highest number of large scalar rewards")
    parser.add_argument("--systematic-options",action="store_true",
                        help="Run short training branches that test different options from the same saved state")
    predictors=parser.add_mutually_exclusive_group()
    predictors.add_argument("--dreamer-proposal",type=Path,help="Use a frozen Dreamer checkpoint for half of option choices")
    predictors.add_argument("--uniform-proposal",action="store_true",help="Use uniform buttons at the same choice rate")
    parser.add_argument("--proposal-parent",type=Path)
    parser.add_argument("--upstream",type=Path)
    args=parser.parse_args()
    if not 0<=args.stage_reservoir<=256:
        parser.error("The number of visual landmarks must be between 0 and 256")
    folder=args.folder.resolve()
    if args.from_search:
        initialize(args.from_search.resolve(),folder,args.seed)
    budget=args.steps
    if args.additional_steps is not None:
        if args.additional_steps<=0:
            parser.error("Additional search decisions must be positive")
        budget=json.loads((folder/"branch-origin.json").read_text())["inherited_training_buttons"]+args.additional_steps
    proposal=None
    if args.dreamer_proposal:
        if not args.proposal_parent or not args.upstream:
            parser.error("Provide the Dreamer parent configuration and upstream checkout")
        proposal=dict(kind="dreamer",checkpoint=args.dreamer_proposal,parent=args.proposal_parent,upstream=args.upstream)
    elif args.uniform_proposal:
        proposal=dict(kind="uniform")
    run(folder,budget,args.motion_probe,args.stage_reservoir,proposal,args.systematic_options)
