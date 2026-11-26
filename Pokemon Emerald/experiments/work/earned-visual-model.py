"""Plan in a tabular visual dynamics model learned from the explorer's own verified gameplay."""
import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np

project = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL"
sys.path.insert(0, str(project))
from train_earned_dreamer import Trajectory, digest, write_json


def visual(frame):
    bins = frame.reshape(10,4,15,4,3).sum(axis=(1,3),dtype=np.uint16) // 256
    return hashlib.blake2b(bins.astype(np.uint8).tobytes(),digest_size=8).digest()


class History:
    def __init__(self):
        self.past = deque(maxlen=3)
        self.screen = None
        self.dwell = self.progress = 0

    def observe(self, frame, incoming, last_button):
        now = visual(frame)
        if self.screen is not None:
            self.past.append((self.screen, last_button))
        self.dwell = min(31, self.dwell+1) if now==self.screen else 0
        self.screen = now
        self.progress += int(incoming >= .25)
        # Reward count and dwell are computed from observed scalar rewards and RGB.
        # The model has no global game clock, map information or episode index.
        return [(self.progress, self.dwell, *self.past, now), (self.progress, now), (now,)]


class TransitionModel:
    def __init__(self):
        self.states = {}
        self.edges = {}

    def index(self, key):
        return self.states.setdefault(key, len(self.states))

    def record(self, a, button, b, reward):
        edge = self.index(a), button, self.index(b)
        count, cents = self.edges.get(edge,(0,0))
        self.edges[edge] = count+1, cents+round(100*reward)

    def solve(self, gamma=.997):
        edge = np.array(list(self.edges), dtype=np.int32)
        start, button, end = edge.T
        statistics = np.array(list(self.edges.values()))
        count, cents = statistics.T
        pair, which = np.unique(start*9+button,return_inverse=True)
        total = np.bincount(which,weights=count)
        mean_reward = np.bincount(which,weights=cents) / (100*total)
        probability = count/total[which]
        origin = pair//9
        value = np.zeros(len(self.states))
        for sweep in range(2048):
            q = mean_reward + gamma*np.bincount(which,weights=probability*value[end])
            updated = np.zeros_like(value)
            np.maximum.at(updated,origin,q)
            change = np.max(np.abs(value-updated))
            value = updated
            if change<1e-7:
                break
        q = mean_reward + gamma*np.bincount(which,weights=probability*value[end])
        # Keep the highest-value observed actions and their counts for reproducible tie breaks.
        # A missing state backs off to a less detailed visual model at inference time.
        table = {}
        for state,key in enumerate(self.states):
            lo, hi = np.searchsorted(origin,[state,state+1])
            if hi<=lo:
                continue
            score = q[lo:hi]
            best = score >= np.max(score)-1e-9
            table[key] = (tuple(int(x) for x in pair[lo:hi][best]%9),
                          tuple(int(x) for x in total[lo:hi][best]))
        return table, dict(states=len(self.states), observed_state_actions=len(pair), distinct_outcomes=len(edge),
                           bellman_sweeps=sweep+1, last_change=float(change), discount=gamma)


def fit(datasets, out):
    out.mkdir(parents=True,exist_ok=True)
    games = [Trajectory(d) for d in datasets]
    tapes = [g.source["source_sha256"] for g in games]
    if (out/"fit.json").exists():
        result = json.loads((out/"fit.json").read_text())
        assert result["training_tapes"]==tapes and result["model_sha256"]==digest(out/"policy.pkl")
        return
    models = [TransitionModel() for _ in range(3)]
    for game in games:
        hist = History()
        prev = last_button = None
        for frame,button,reward,first,last,term in game.episode():
            codes = hist.observe(frame,reward,last_button)
            if prev is not None:
                for model,src,dst in zip(models,prev,codes):
                    model.record(src,last_button,dst,reward)
            prev, last_button = codes, button
    tables, stats = zip(*(model.solve() for model in models))
    with (out/"policy.tmp").open("wb") as f:
        pickle.dump(tables,f,protocol=4)
    (out/"policy.tmp").replace(out/"policy.pkl")
    write_json(out/"fit.json",dict(training_tapes=tapes,training_decisions=sum(len(g.tape) for g in games),
        model_sha256=digest(out/"policy.pkl"),tables=stats, frozen_evaluation_data_used=False,
        observation="RGB, preceding actions and scalar rewards only",
        method="empirical action-conditioned transitions, discounted value iteration and visual-history backoff"))


def play(out, seed, limit):
    from pixel_env import PixelPool
    target = out / f"fresh-seed-{seed}-limit-{limit}"
    target.mkdir(exist_ok=True)
    model_sha = digest(out/"policy.pkl")
    if json.loads((out/"fit.json").read_text())["model_sha256"] != model_sha:
        raise ValueError("Frozen visual model hash mismatch")
    if (target/"result.json").exists():
        return
    with (out/"policy.pkl").open("rb") as f:
        tables = pickle.load(f)
    hist, usage, rng = History(), Counter(), np.random.default_rng(seed)
    env = PixelPool(1)
    try:
        frame = env.command(0,"reset")
        reward, last_button = 0., None
        for n in range(1,limit+1):
            keys = hist.observe(frame,reward,last_button)
            for level,(table,key) in enumerate(zip(tables,keys)):
                options = table.get(key)
                if options:
                    buttons,counts = options
                    button = buttons[0] if len(buttons)==1 else int(rng.choice(buttons,p=np.asarray(counts)/sum(counts)))
                    break
            else:
                level, button = 3, int(rng.integers(9))
            usage[level] += 1
            frame,reward,done = env.command(0,"act",button)
            last_button = button
            if n % 8192 == 0 or reward>=1 or done or n==limit:
                dest = target/f"step-{n:06}"
                env.command(0,"audit",str(dest))
                write_json(target/"status.json",dict(steps=n, actions_by_lookup_depth=dict(usage)))
            if done:
                audit, = dest.glob("*.json")
                subprocess.run([sys.executable,str(project/"pixel_replay.py"),str(audit),str(target/"badge-replay"),
                                "--event","stone_badge","--video"],check=True)
                break
        write_json(target/"result.json",dict(steps=n, seed=seed, model_sha256=model_sha,
            actions_by_lookup_depth=dict(usage), fresh_boot=True, ingame_restores=0, weights_updated=0,
            verified_stone_badge=(target/"badge-replay/verification.json").exists()))
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--dataset",type=Path,action="append",required=True)
    parser.add_argument("--seed",type=int,default=24917)
    parser.add_argument("--limit",type=int,default=65536)
    args = parser.parse_args()
    output = args.out.resolve()
    fit([p.resolve() for p in args.dataset],output)
    play(output,args.seed,args.limit)
