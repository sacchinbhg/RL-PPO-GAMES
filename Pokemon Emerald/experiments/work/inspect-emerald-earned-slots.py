"""Rebuild the training archive from an exact self-earned trajectory and measure where it can restart."""
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from reward_branch_search import Archive, Point, visual_code, write_json
from scene_reservoir import SceneReservoir
from train_earned_dreamer import Trajectory

def analyze(dataset, output):
    game=Trajectory(dataset)
    with np.load(next(iter(sorted(dataset.glob("chunk-*.npz"))))) as chunk:
        boot=chunk["image"][0]
    class SnapshotTracker:
        initial=boot
        current=boot
        def __init__(self):
            self.images={}
        def command(self,command,slot):
            if command=="save":
                self.images[slot]=self.current
            elif command=="restore":
                return self.images[slot]
            else:
                raise AssertionError(command)
    env=SnapshotTracker()
    slots=Archive(env,np.random.default_rng(465721))
    landmarks=SceneReservoir(env,np.random.default_rng([465721,321586,817]),64)
    count=stage=0
    first_times={}
    for i,(pic,act,reward,first,last,term) in enumerate(game.episode(uncapped=True)):
        cents=round(100*reward)
        count+=cents
        stage+=int(cents>=100)
        env.current=pic
        point=Point(visual_code(pic),count,i,stage=stage)
        if cents:
            slots.offer(point)
            if cents>=100:
                first_times[i]=cents
        if i and (i%64==0 or cents>=100):
            landmarks.consider(point,pic)
    elite=sorted([[p.length,p.total] for s,p in slots.points.items() if s])
    ordered=sorted(slots.points,key=lambda s:(-slots.points[s].total,slots.points[s].length))
    prob=Counter()
    for mass,group in ((.55,ordered[:6]),(.30,ordered[:24]),(.15,ordered)):
        for sid in group:
            prob[slots.points[sid].length]+=mass/len(group)
    output.parent.mkdir(parents=True,exist_ok=True)
    for _ in range(200):
        landmarks.start()
    assert landmarks.chosen>50 and all(p.stage==stage for p in landmarks.points)
    extra=[p.length for p in landmarks.points]
    for i in extra:
        assert i>=max(first_times)
    write_json(output,dict(source_sha256=game.source["source_sha256"],training_decisions=len(game.tape),
               return_cents=count,cold_start_elite=elite,earliest_elite=elite[0][0],latest_elite=elite[-1][0],
               probability_at_last_quarter=sum(p for i,p in prob.items() if i>=.75*len(game.tape)),
               probability_at_last_tenth=sum(p for i,p in prob.items() if i>=.9*len(game.tape)),
               initial_sampling_probability_by_button=prob,scalar_rewards_at_least_one=first_times,
               visual_landmark_lengths=sorted(extra),visual_landmark_stage=stage,distinct_appearances=len(landmarks.seen),
               matched_sampling_last_tenth=.5*sum(p for i,p in prob.items() if i>=.9*len(game.tape))+
                                           .5*sum(i>=.9*len(game.tape) for i in extra)/len(extra)))


if __name__=="__main__":
    analyze(Path(sys.argv[1]),Path(sys.argv[2]))
