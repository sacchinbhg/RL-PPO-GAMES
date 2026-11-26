"""Explore using a translation model estimated from self-generated RGB and action pairs."""
from collections import Counter
import json
import math

import numpy as np


class MotionGuide:
    def __init__(self, probe, rng):
        fit=json.loads(probe.read_text())
        if not fit["informative"] or fit["position_or_map_input"]:
            raise ValueError("Expected a validated pixel-only displacement model")
        self.source=fit["training_tape_sha256"]
        self.shifts={int(b):tuple(row["most_common"]) for b,row in fit["learned_actions"].items()}
        self.rng=rng
        self.visits=Counter()
        self.attempts=Counter()
        self.successes=Counter()
        self.decisions=Counter()
        self.moved=Counter()
        self.gray=self.pose=None

    def restore(self, image, code, pose=None):
        self.gray=image.astype(np.float32).mean(axis=2)
        self.pose=pose if pose is not None else (code,0,0,0,-1)

    def observe(self, button, image, code):
        now=image.astype(np.float32).mean(axis=2)
        prior=self.gray
        anchor,x,y,ttl,last=self.pose
        place=(anchor,x,y)
        edge=place,button
        self.attempts[edge]+=1
        change=float(np.abs(now-prior).mean())
        moved=False
        if button in self.shifts and change>=2.5:
            dx,dy=self.shifts[button]
            h,w=now.shape
            top,bottom=max(0,dy)+1,min(h,h+dy)-1
            left,right=max(0,dx)+1,min(w,w+dx)-1
            target=now[top:bottom,left:right]
            residual=np.abs(target-prior[top-dy:bottom-dy,left-dx:right-dx]).mean()
            unchanged=np.abs(target-prior[top:bottom,left:right]).mean()
            moved=unchanged>=2.5 and residual<=.55*unchanged
            if moved:
                x,y=x-dx,y-dy
                self.successes[edge]+=1
                self.moved[button]+=1
                ttl,last=14,button
        if not moved:
            ttl=max(0,ttl-1)
            if change>=28:
                anchor,x,y,ttl,last=code,0,0,0,-1
        self.visits[anchor,x,y]+=1
        self.gray,self.pose=now,(anchor,x,y,ttl,last)

    def choose(self):
        anchor,x,y,ttl,last=self.pose
        place=anchor,x,y
        if self.rng.random()>(.73 if ttl else .08):
            self.decisions["ordinary_options"]+=1
            return None
        blocked=any(self.attempts[place,b]-self.successes[place,b]>=2 for b in self.shifts)
        other=[b for b in range(9) if b not in self.shifts]
        if blocked and other and self.rng.random()<.22:
            fewest=min(self.attempts[place,b] for b in other)
            choices=[b for b in other if self.attempts[place,b]==fewest]
            self.decisions["test_other_action"]+=1
        else:
            scores={}
            for b,(dx,dy) in self.shifts.items():
                n=self.attempts[place,b]
                p=(self.successes[place,b]+1)/(n+2)
                neighbor=self.visits[anchor,x-dx,y-dy]
                scores[b]=p/math.sqrt(1+neighbor)+.18/math.sqrt(1+n)+(.10 if b==last and ttl else 0)
            highest=max(scores.values())
            choices=[b for b,s in scores.items() if s==highest]
            self.decisions["predicted_motion"]+=1
        return choices[int(self.rng.integers(len(choices)))],1
