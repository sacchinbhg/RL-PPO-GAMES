"""Recover a route using a fixed model of screen transitions observed in the agent's own games."""
from collections import Counter,defaultdict
import hashlib
import heapq
import numpy as np
from earned_memory import encode

def appearance(feature):
    return hashlib.blake2b((feature//16).astype(np.uint8).tobytes(),digest_size=10).digest()

class VisualRouteRecovery:
    def __init__(self,memory):
        edges=defaultdict(Counter)
        first={}
        traces=[]
        for trace in memory.traces:
            states=[appearance(f) for f in trace["features"]]
            traces.append(states)
            last=max(1,len(states)-1)
            for t,(state,following) in enumerate(zip(states,states[1:])):
                first[state]=min(first.get(state,1.),t/last)
                edges[state,int(trace["buttons"][t])][following]+=1
            first.setdefault(states[-1],1.)
        best=max(range(len(traces)),key=lambda j:float(memory.traces[j]["total"]))
        route=traces[best]
        goals={s for s in route[3*len(route)//4:] if first[s]>=.75}
        reverse=defaultdict(list)
        for (state,button),counts in edges.items():
            total=sum(counts.values())
            for following,n in counts.items():
                if following!=state:
                    # Frequently reproduced transitions cost fewer attempts.
                    reverse[following].append((state,total/n))
        distance={s:0. for s in goals}
        queue=[(0.,s) for s in goals]
        heapq.heapify(queue)
        while queue:
            cost,state=heapq.heappop(queue)
            if cost!=distance[state]:
                continue
            for earlier,edge_cost in reverse[state]:
                candidate=cost+edge_cost
                if candidate<distance.get(earlier,float("inf")):
                    distance[earlier]=candidate
                    heapq.heappush(queue,(candidate,earlier))
        choices=defaultdict(list)
        for (state,button),counts in edges.items():
            if state not in distance:
                continue
            total=sum(counts.values())
            advancing=[distance[s]+total/n for s,n in counts.items() if distance.get(s,float("inf"))<distance[state]]
            if advancing:
                choices[state].append((min(advancing),button))
        self.choices=choices
        self.distance=distance
        self.early={s for s in choices if first[s]<.65}
        self.fit=dict(screen_states=len(first),observed_state_actions=len(edges),known_late_screens=len(goals),
                      states_with_return_path=len(choices),early_screens_with_return_path=len(self.early))
        self.quiet=self.cooldown=self.burst=0
        self.previous=None
        self.failed=Counter()
        self.stats=Counter()

    def choose(self,frame,reward):
        _,feature=encode(frame)
        state=appearance(feature)
        self.quiet=0 if reward>=.25 else self.quiet+1
        if self.previous is not None:
            prior,button=self.previous
            if self.distance.get(state,float("inf"))<self.distance[prior]:
                self.stats["steps_toward_known_frontier"]+=1
                self.failed[prior,button]=0
            else:
                self.failed[prior,button]+=1
            self.previous=None
        if self.burst and (state not in self.choices or self.burst>=512):
            if self.distance.get(state)==0:
                self.stats["arrived_at_known_late_screen"]+=1
            self.burst=0
            self.cooldown=64
        if self.cooldown:
            self.cooldown-=1
            return None
        if not self.burst and (self.quiet<256 or state not in self.early):
            return None
        allowed=[(cost+4*self.failed[state,b],b) for cost,b in self.choices.get(state,())
                 if self.failed[state,b]<5]
        if not allowed:
            self.burst=0
            self.cooldown=64
            return None
        if not self.burst:
            self.stats["route_recoveries_started"]+=1
        self.burst+=1
        _,button=min(allowed)
        self.previous=(state,button)
        self.stats["buttons_chosen"]+=1
        return button
