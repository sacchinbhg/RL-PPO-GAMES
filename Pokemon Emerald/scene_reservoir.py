"""Sample visually distinct training states since the latest large scalar rewards."""
from collections import Counter, deque
import numpy as np
from reward_branch_search import visual_code


def appearance(image):
    small=image[::2,::2].reshape(2,10,2,15,3).astype(np.float32)
    means=(small.mean(axis=(1,3))//32).astype(np.uint8)
    texture=(small.std(axis=(1,3))//24).astype(np.uint8)
    return means.tobytes()+texture.tobytes()


class SceneReservoir:
    def __init__(self,env,rng,capacity,systematic=False,reward_landmarks=False,recent_landmarks=0):
        if not 0<=recent_landmarks<=capacity:
            raise ValueError("The recent landmark count must fit inside the reservoir")
        self.env,self.rng,self.capacity=env,rng,capacity
        self.systematic=systematic
        self.reward_landmarks=reward_landmarks
        self.recent_limit=recent_landmarks
        self.recent_slots=deque()
        self.recent_starts=0
        self.attempts={}
        self.last_slot=None
        self.distinct_trials=0
        self.stage=-1
        self.starts=Counter()
        self.seen=Counter()
        self.ids={}
        self.points=[]
        self.keys=[]
        self.chosen=0

    def accepts(self,length,used,cents):
        if self.reward_landmarks:
            return cents>0
        stride=16 if self.systematic else 64
        return cents>=100 or length//stride!=(length-used)//stride

    def new_slot(self):
        if len(self.keys)<self.capacity:
            slot=len(self.keys)
            if self.recent_limit:
                if len(self.recent_slots)==self.recent_limit:
                    self.recent_slots.popleft()
                self.recent_slots.append(slot)
            return slot
        if not self.recent_limit:
            slot=int(self.rng.integers(len(self.seen)))
            return slot if slot<self.capacity else None
        # The oldest recent item becomes an ordinary reservoir candidate. Keep the
        # exact emulator save in its current slot when it wins a historical place.
        historical=sorted(set(range(self.capacity)).difference(self.recent_slots))
        oldest=self.recent_slots.popleft()
        draw=int(self.rng.integers(len(self.seen)-self.recent_limit))
        slot=historical[draw] if draw<len(historical) else oldest
        self.recent_slots.append(slot)
        return slot

    def consider(self,point,image):
        if point.stage<self.stage:
            return
        if point.stage>self.stage:
            self.stage=point.stage
            self.seen.clear()
            self.ids.clear()
            self.keys.clear()
            self.points.clear()
            self.attempts.clear()
            self.recent_slots.clear()
        key=appearance(image)
        self.seen[key]+=1
        if key in self.ids:
            slot=self.ids[key]
            if self.systematic and self.attempts.get(slot):
                # Preserve the exact emulator state while testing its different outgoing options.
                return
            if self.rng.integers(self.seen[key])!=0:
                return
        elif self.seen[key]>1:
            return
        else:
            slot=self.new_slot()
            if slot is None:
                return
            if slot==len(self.keys):
                self.keys.append(key)
                self.points.append(point)
            else:
                del self.ids[self.keys[slot]]
        self.ids[key]=slot
        self.keys[slot]=key
        self.points[slot]=point
        self.attempts.pop(slot,None)
        self.env.command("save",1000+slot)

    def start(self):
        self.last_slot=None
        if not self.points or self.rng.random()>=(.9 if self.systematic else .5):
            return None
        slot=int(self.rng.integers(len(self.points)))
        point=self.points[slot]
        image=self.env.command("restore",1000+slot)
        if visual_code(image)!=point.screen:
            raise RuntimeError("Stored visual landmark disagrees with its original image")
        self.chosen+=1
        if self.recent_limit and slot in self.recent_slots:
            self.recent_starts+=1
        self.last_slot=slot
        self.starts[point.length//1024]+=1
        return point,image

    def next_option(self,durations,remaining):
        if self.last_slot is None:
            return None
        attempted=self.attempts.setdefault(self.last_slot,Counter())
        options=[(b,n) for b in range(9) for n in durations if n<=remaining]
        fewest=min(attempted[op] for op in options)
        choices=[op for op in options if attempted[op]==fewest]
        chosen=choices[int(self.rng.integers(len(choices)))]
        self.distinct_trials+=int(attempted[chosen]==0)
        attempted[chosen]+=1
        self.last_slot=None
        return chosen
