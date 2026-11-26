"""Keep the official recurrent policy aligned with the real buttons in an explorer's training forks."""
import numpy as np


class ProposalEnv:
    def __init__(self, env, actor, rng):
        self.env=env
        self.actor=actor
        self.rng=rng
        self.initial=self.image=env.initial
        self.incoming=0.
        self.previous=None
        self.pending=None
        self.saved={}
        self.observed=self.restores=0

    def propose(self):
        if self.pending is None:
            if self.actor is None:
                self.pending=int(self.rng.integers(9))
            else:
                decide,_,_=self.actor
                actual=None if self.previous is None else np.array([self.previous],np.int32)
                button=decide(self.image[None],np.array([self.incoming],np.float32),actual is None,previous_buttons=actual)
                self.pending=int(button[0])
                self.observed+=1
        return self.pending

    def command(self, name, arg):
        if name=="step":
            button,duration=arg
            if duration!=1:
                raise ValueError("A recurrent proposal must observe every executed action")
            if self.actor is not None:
                self.propose()
            result=self.env.command(name,arg)
            self.image,cents,_,used=result
            if used!=1:
                raise RuntimeError("Proposal skipped an environment observation")
            self.incoming,self.previous,self.pending=cents/100.,button,None
            return result
        if name=="save":
            if self.pending is not None:
                raise RuntimeError("Save the state before choosing its next action")
            result=self.env.command(name,arg)
            carry=self.actor[1]() if self.actor is not None else None
            self.saved[arg]=(carry,self.previous,self.incoming)
            return result
        if name=="restore":
            self.image=self.env.command(name,arg)
            carry,self.previous,self.incoming=self.saved[arg]
            if self.actor is not None:
                self.actor[2](carry)
            self.pending=None
            self.restores+=1
            return self.image
        return self.env.command(name,arg)

    def close(self):
        self.env.close()
