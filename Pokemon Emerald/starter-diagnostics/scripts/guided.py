"""Reachability control. Uses game-map waypoints, restores, controller buttons and no RAM writes."""
import heapq, itertools, json, struct, sys
from pathlib import Path
from PIL import Image
project=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project))
from emerald import Emerald
from experiment import location

cfg=json.loads((project/"local.json").read_text())
env=Emerald(cfg["core"],cfg["rom"])
out=project/"starter-diagnostics"
prefix=(project/"starter-diagnostics/mgba-town_after_rival.bin").read_bytes()
def rd(a,n):return int.from_bytes(env.ram(a,n),"little")
def actor_xy():
    obj=rd(0x02037595,1)
    if obj>=16:return 0,0,0
    base=0x02037350+obj*36
    x,y=struct.unpack("<hh",env.ram(base+16,4))
    return x-7,y-7,rd(base+24,1)&15
def get():
    k=env.descriptor()
    sb=rd(0x03005d8c,4)
    name=location(k)
    if not name or env._header(*k[1:3])!=(rd(0x0203731c,4),rd(0x0203732a,2)):return None
    town=rd(sb+0x139c+2*0x50,2)
    route=rd(sb+0x139c+2*0x60,2)
    party=rd(0x020244e9,1)
    return dict(map=name,town=town,route=route,party=party,bag=bool(k[7]),phase=k[0])
def guide():
    for a in prefix:env.step(a)
    story=get()
    assert story["town"]>=1 and story["map"]=="LittlerootTown" and not story["party"]
    rootstory=story
    trace=b""
    for _ in range(16):env.step(2);trace+=bytes([2])
    q=[]
    serial=itertools.count()
    seen=set()
    events={}
    count=16
    def enter(path,story):
        if not story or story["map"] not in ("LittlerootTown","Route101"):return
        on_route=story["map"]=="Route101"
        tier=3+int(story["route"]>=2)+int(story["bag"]) if on_route else min(2,story["town"])
        x,y,face=actor_xy()
        target=(7,14) if tier>=4 else ((10,19) if on_route else (11,-1 if tier==2 else 1))
        distance=abs(x-target[0])+abs(y-target[1])
        lock=rd(0x03000f2c,1)
        cb=rd(0x030022c4,4)
        script=env.ram(0x03000e40,12) if lock else b""
        sig=(tier,x,y,face,lock,cb,script,story["route"])
        if sig in seen:return
        seen.add(sig)
        score=-tier*1000+distance*6+len(path)*.18
        heapq.heappush(q,(score,next(serial),env.save(),path))
    enter(trace,get())
    while q and count<8500 and "party" not in events:
        _,_,saved,base=heapq.heappop(q)
        for seq in ((4,),(5,),(6,),(7,),(1,)*6,(2,)*4,(1,)*32):
            env.load(saved)
            path=base
            for a in seq:
                env.step(a)
                path+=bytes((a,))
                count+=1
                s=get()
                if s and s["map"]=="Route101":
                    for event,ok in (("route101",s["phase"] in (5,6)),("bag",s["bag"]),("party",1<=s["party"]<=6 and s["phase"] in (5,6))):
                        if ok and event not in events:
                            events[event]=dict(search_decisions=count,from_boot=len(prefix)+len(path),story=s)
                            (out/("guided-"+event+".bin")).write_bytes(prefix+path)
                            Image.fromarray(env.screen).save(out/("guided-"+event+".png"))
                            print(event,events[event],flush=True)
                if "party" in events:break
            if "party" in events:break
            enter(path,get())
    proof=None
    best=next((e for e in ("party","bag","route101") if e in events),None)
    if best:
        env.load(env.boot)
        for a in (out/("guided-"+best+".bin")).read_bytes():env.step(a)
        proof=get()
        assert proof==events[best]["story"]
    (out/"guided.json").write_text(json.dumps(dict(start=rootstory,prefix=len(prefix),nodes=len(seen),
        search_decisions=count,events=events,verified_from_boot=bool(best),final_replayed=proof),indent=2)+"\n")
try:
    guide()
finally:env.close()
