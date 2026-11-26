"""Use measured action effects to recover when self-imitation stalls on quiet images."""
import json
import numpy as np

from fit_local_motion import descriptors, distances, nearest
from train_earned_dreamer import digest


class EffectPlanner:
    def __init__(self, directory, mode, seed):
        if mode not in ("frequency","local"):
            raise ValueError("Unknown effect-ranking method")
        report=json.loads((directory/"result.json").read_text())
        path=directory/"model.npz"
        if not report["passes_comparison"] or report["model_sha256"]!=digest(path):
            raise ValueError("Expected the verified model selected by the held-out comparison")
        self.sha=report["model_sha256"]
        self.mode=mode
        self.rng=np.random.default_rng([seed,173753])
        with np.load(path,allow_pickle=False) as model:
            self.buttons=model["buttons"]
            self.labels=model["outcomes"]
            self.patches=model["patches"]
            self.features=[descriptors(model["train"],top,left) for top,left in self.patches]
        self.prior=self.labels.mean(0)
        self.previous=None
        self.quiet=0
        self.calls=self.disagreements=0
        self.first_disagreement=None

    def observe(self, frame):
        if self.previous is not None:
            changed=np.abs(frame.astype(np.int16)-self.previous).max(2)>=24
            self.quiet=self.quiet+1 if changed.mean()<.02 else 0
        self.previous=frame

    def predictions(self, frame):
        scores=[]
        for j,(top,left) in enumerate(self.patches):
            patch=descriptors(frame[None],top,left)
            scores.append(nearest(distances(patch,self.features[j]),self.labels[:,j:j+1])[0,0])
        return np.asarray(scores)

    def select(self, frame, fallback, recovery_started, decision):
        if not recovery_started or self.quiet<4:
            return fallback
        self.calls+=1
        draw=self.rng.random()
        proposals=[]
        for scores in (self.prior,self.predictions(frame)):
            choices=self.buttons[np.isclose(scores,scores.max(),atol=1e-7,rtol=0)]
            proposals.append(int(choices[int(draw*len(choices))]))
        if proposals[0]!=proposals[1]:
            self.disagreements+=1
            if self.first_disagreement is None:
                self.first_disagreement=dict(decision=decision,frequency=proposals[0],local=proposals[1])
        return proposals[int(self.mode=="local")]
