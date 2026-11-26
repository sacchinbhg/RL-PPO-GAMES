"""Sample visually distinct training states since the latest large scalar rewards."""
from collections import Counter
import numpy as np
from reward_branch_search import visual_code


def appearance(image):
    small=image[::2,::2].reshape(2,10,2,15,3).astype(np.float32)
    means=(small.mean(axis=(1,3))//32).astype(np.uint8)
    texture=(small.std(axis=(1,3))//24).astype(np.uint8)
    return means.tobytes()+texture.tobytes()


class SceneReservoir:
    def __init__(self,env,rng,capacity):
        self.env,self.rng,self.capacity=env,rng,capacity
        self.stage=-1
        self.starts=Counter()
        self.seen=Counter()
        self.ids={}
        self.points=[]
        self.keys=[]
        self.chosen=0

    def consider(self,point,image):
        if point.stage<self.stage:
            return
        if point.stage>self.stage:
            self.stage=point.stage
            self.seen.clear()
            self.ids.clear()
            self.keys.clear()
            self.points.clear()
        key=appearance(image)
        self.seen[key]+=1
        if key in self.ids:
            if self.rng.integers(self.seen[key])!=0:
                return
            slot=self.ids[key]
        elif self.seen[key]>1:
            return
        elif len(self.keys)<self.capacity:
            slot=len(self.keys)
            self.keys.append(key)
            self.points.append(point)
        else:
            slot=int(self.rng.integers(len(self.seen)))
            if slot>=self.capacity:
                return
            del self.ids[self.keys[slot]]
        self.ids[key]=slot
        self.keys[slot]=key
        self.points[slot]=point
        self.env.command("save",1000+slot)

    def start(self):
        if not self.points or self.rng.random()>=.5:
            return None
        slot=int(self.rng.integers(len(self.points)))
        point=self.points[slot]
        image=self.env.command("restore",1000+slot)
        if visual_code(image)!=point.screen:
            raise RuntimeError("Stored visual landmark disagrees with its original image")
        self.chosen+=1
        self.starts[point.length//1024]+=1
        return point,image
