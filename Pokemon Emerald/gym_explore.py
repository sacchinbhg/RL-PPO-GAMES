"""Learn successful sequences of interaction options in the original ROM using save-state branching.

The explorer observes the current loaded map, local objects, active coordinate events and game
milestones. It has no list of target coordinates, route to the gym, or demonstration inputs.
"""
import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import heapq
import itertools
import json
import random
import struct
import time
from pathlib import Path

import numpy as np
from PIL import Image
from emerald import Emerald, MAPS
from experiment import location, write_json
from gym_data import ROOT, RUN

VECTORS = {4:(0,-1), 5:(0,1), 6:(-1,0), 7:(1,0)}
DIRECTIONS = {1:5, 2:4, 3:6, 4:7}
MAP_PROGRESS = {"Route101":1, "OldaleTown":2, "Route103":3, "Route102":4,
                "PetalburgCity":5, "PetalburgCity_Gym":6, "Route104":7,
                "PetalburgWoods":8, "RustboroCity":9, "RustboroCity_Gym":10}


class StepLimit(Exception):
    pass


class Game:
    def __init__(self):
        cfg = json.loads((ROOT / "local.json").read_text(encoding="utf-8-sig"))
        self.env = Emerald(cfg["core"], cfg["rom"])
        self.count = 0
        self.trace = bytearray()
        self.rng = random.Random(15472)
        self.move_values = defaultdict(lambda: [0.,0])
        self.pending_attack = None

    def read(self, addr, size):
        if 0x08000000 <= addr < 0x0A000000:
            return self.env.rom_data[addr-0x08000000:addr-0x08000000+size]
        return self.env.ram(addr, size)

    def num(self, addr, n=1):
        return int.from_bytes(self.read(addr,n), "little")

    def step(self, action):
        if getattr(self,"limit",None) is not None and self.count>=self.limit:
            raise StepLimit()
        self.env.step(action)
        self.count += 1
        self.trace.append(action)
        if getattr(self,"capture_battles",None):
            cb=self.num(0x03005d60,4)&~1
            if cb==0x08057bfc and (self.num(0x030022c4,4)&~1)==0x08038420:
                enemy=self.num(0x02024084+0x58,2)
                ally=self.num(0x02024084,2)
                level=self.num(0x02024084+0x2a)
                trainer=self.num(0x02038bca,2) if self.num(0x02022fec,4)&8 else 0
                st=self.state()
                if st and st["phase"]==6 and st["name"]=="RustboroCity_Gym":
                    sig=(trainer,ally,enemy,level)
                    if sig not in self.battle_saved:
                        self.battle_saved.add(sig)
                        path=self.capture_battles/("-".join(map(str,sig)))
                        path.mkdir(parents=True,exist_ok=True)
                        tape=bytes(self.trace)
                        (path/"state.snap").write_bytes(self.env.save())
                        (path/"buttons.bin").write_bytes(tape)
                        write_json(path/"source.json",dict(trainer=trainer,party_species=ally,enemy_species=enemy,
                                   level=level,decisions=len(tape),own_hp=self.num(0x02024084+0x28,2),
                                   own_maxhp=self.num(0x02024084+0x2c,2),search_decisions=self.count))
                        print("BATTLE",sig,len(tape),flush=True)

    def state(self):
        e = self.env
        k = e.descriptor()
        name = location(k)
        sb = self.num(0x03005D8C,4)
        if not name or not (0x02025a00<=sb<=0x02025a7c) or e._header(*k[1:3]) != (self.num(0x0203731c,4),self.num(0x0203732a,2)):
            return None
        var = lambda i:self.num(sb+0x139c+2*i,2)
        flag = lambda i:(self.num(sb+0x1270+i//8)>>(i%8))&1
        oid = self.num(0x02037595)
        if oid>=16:
            return None
        base = 0x02037350+36*oid
        x,y = struct.unpack("<hh",self.read(base+16,4))
        party = self.num(0x020244e9)
        if party>6:
            return None
        return dict(name=name, mapid=k[1:3], pos=(x-7,y-7), face=self.num(base+24)&15, phase=k[0],
                    clock=k[6], rival=int(var(0x50)>=1 and var(0x8d)>=3),
                    town=var(0x50), route=var(0x60), lab=var(0x84), party=party,
                    level=self.num(0x020244ec+84) if party else 0, hp=self.num(0x020244ec+86,2) if party else 0,
                    maxhp=self.num(0x020244ec+88,2) if party else 0,
                    petal=var(0x57), woods=var(0x98), dex=flag(0x861), defeated=flag(0x4f0),
                    badge=flag(0x867), lock=self.num(0x03000f2c), sb=sb)

    def geometry(self):
        layout = self.num(0x02037318,4)
        w,h = struct.unpack("<ii",self.read(layout,8))
        gw,gh,grid = struct.unpack("<iiI",self.read(0x03005dc0,12))
        if not (0<w<512 and 0<h<512 and w<gw<550 and h<gh<550 and 0x02000000<=grid<0x02040000):
            return None
        tiles = np.frombuffer(self.read(grid,gw*gh*2),dtype="<u2").reshape(gh,gw)
        return w,h,tiles

    def objects(self):
        actors = {}
        raw = self.read(0x02037350,36*16)
        player = self.num(0x02037595)
        for i in range(16):
            off=i*36
            if i!=player and raw[off]&1:
                x,y=struct.unpack_from("<hh",raw,off+16)
                actors[raw[off+8]]=(x-7,y-7)
        return actors

    def options(self, state):
        geometry=self.geometry()
        if geometry is None:
            return {}
        w,h,t=geometry
        opts={}
        for local, pos in self.objects().items():
            opts[f"object/{local}"]=("object",pos)
        ptr=self.num(0x0203731c,4)
        counts=self.read(ptr,4)
        templates=self.num(ptr+4,4)
        for i in range(counts[0]):
            rec=self.read(templates+24*i,24)
            hidden=struct.unpack_from("<H",rec,20)[0]
            if hidden and (self.num(state["sb"]+0x1270+hidden//8)>>(hidden%8))&1:
                continue
            opts.setdefault(f"object/{rec[0]}",("object",struct.unpack_from("<hh",rec,4)))
        for kind,num,stride,off in (("warp",counts[1],8,8),("trigger",counts[2],16,12),("sign",counts[3],12,16)):
            addr=self.num(ptr+off,4)
            for i in range(num):
                rec=self.read(addr+i*stride,stride)
                pos=struct.unpack_from("<hh",rec)
                if kind=="trigger":
                    var,index=struct.unpack_from("<HH",rec,6)
                    if 0x4000<=var<0x4100 and self.num(state["sb"]+0x139c+2*(var-0x4000),2)!=index:
                        continue
                opts[f"{kind}/{i}"]=(kind,pos)
        conn=self.num(0x02037324,4)
        if conn:
            num,addr=struct.unpack("<iI",self.read(conn,8))
            for i in range(num):
                dr=self.num(addr+i*12)
                if dr not in DIRECTIONS:
                    continue
                a=DIRECTIONS[dr]
                dx,dy=VECTORS[a]
                spans=[]
                for z in range(w if dx==0 else h):
                    x,y=(z,0 if dy<0 else h-1) if dx==0 else (0 if dx<0 else w-1,z)
                    if all((val&0x0c00)==0 and (val&0x3ff)!=0x3ff for val in (int(t[y+7,x+7]),int(t[y+7+dy,x+7+dx]))):
                        if not spans or z!=spans[-1][-1]+1:spans.append([])
                        spans[-1].append(z)
                for j,span in enumerate(spans):
                    z=span[len(span)//2]
                    x,y=(z,0 if dy<0 else h-1) if dx==0 else (0 if dx<0 else w-1,z)
                    opts[f"edge/{dr}/{j}"]=("edge",(x+dx,y+dy))
        return opts

    def path(self, start, goal, kind, blocked):
        w,h,tiles=self.geometry()
        occupied=set(self.objects().values())
        layout=self.num(0x02037318,4)
        attrs=[self.num(self.num(layout+offset,4)+16,4) for offset in (16,20)]
        q=deque([(start,())])
        seen={start}
        while q:
            pos,actions=q.popleft()
            for a,(dx,dy) in VECTORS.items():
                to=(pos[0]+dx,pos[1]+dy)
                if (pos,a) in blocked:
                    continue
                if to==goal:
                    return actions+(a,)
                x,y=to
                if to in seen or to in occupied or not (0<=x<w and 0<=y<h):
                    continue
                tile=int(tiles[y+7,x+7])
                if kind=="object" and (x+dx,y+dy)==goal:
                    metatile=tile&0x3ff
                    if self.num(attrs[metatile//512]+2*(metatile%512))==0x80:
                        return actions+(a,)
                if tile&0x0c00 or (tile&0x3ff)==0x3ff or tile>>12==1:
                    continue
                seen.add(to)
                q.append((to,actions+(a,)))
        return ()

    def destination(self,label):
        fields=label.split("/")
        if fields[0]=="warp":
            addr=self.num(self.num(0x0203731c,4)+8,4)+8*int(fields[1])
            num,group=self.read(addr+6,2)
        elif fields[0]=="edge":
            count,addr=struct.unpack("<iI",self.read(self.num(0x02037324,4),8))
            for i in range(count):
                if self.num(addr+12*i)==int(fields[1]):
                    group,num=self.read(addr+12*i+8,2)
                    break
            else:return None
        else:return None
        return MAPS[group][num] if group<len(MAPS) and num<len(MAPS[group]) else None

    def settle(self, start_map, cap=900):
        stable=0
        for i in range(cap):
            st=self.state()
            if st and st["phase"]==5 and not st["lock"]:
                stable+=1
                if stable>=3:return st
                self.step(0)
                continue
            else:stable=0
            # Move selection explores the available attacks. Selection is fixed per turn.
            cb=self.num(0x03005d60,4)&~1
            if st and st["phase"]==6 and cb==0x08057bfc:
                if not getattr(self,"selecting",False):
                    pp=self.read(0x02024084+0x24,4)
                    choices=[j for j,v in enumerate(pp) if v]
                    moves=struct.unpack("<4H",self.read(0x02024084+12,8))
                    foe=self.num(0x02024084+0x58,2)
                    hp=self.num(0x02024084+0x58+0x28,2)
                    if self.pending_attack:
                        oldkey,oldhp=self.pending_attack
                        if oldkey[1]==foe and oldhp>=hp:
                            stats=self.move_values[oldkey]
                            stats[0]+=min(30,oldhp-hp)
                            stats[1]+=1
                    def quality(j):
                        power=self.num(0x0831c898+12*moves[j]+1)
                        total,visits=self.move_values[(moves[j],foe)]
                        return (min(16,power/6)+total)/(1+visits)+self.rng.uniform(0,3)
                    self.move=max(choices,key=quality) if choices else 0
                    self.pending_attack=((moves[self.move],foe),hp)
                self.selecting=True
                cur=self.num(0x020244b0)
                if (cur^self.move)&1:button=7 if self.move&1 else 6
                elif (cur^self.move)&2:button=5 if self.move&2 else 4
                else:button=1
            else:
                self.selecting=False
                button=2 if i%17==12 else 1
            self.step(button)
        return self.state()

    def interact(self, label, option):
        before=self.state()
        if before is None:return None
        if before["lock"] or before["phase"]!=5:
            return self.settle(before["name"])
        kind,goal=option
        blocked=set()
        for _ in range(100):
            st=self.state()
            if not st or st["name"]!=before["name"] or st["phase"]!=5 or st["lock"]:
                return self.settle(before["name"])
            if kind=="object":
                goal=self.objects().get(int(label.split("/")[1]),goal)
            pos=st["pos"]
            if pos==goal and kind not in ("object","sign"):
                if kind=="warp":self.step(4)
                return self.settle(before["name"],100)
            path=self.path(pos,goal,kind,blocked)
            if not path:return st
            a=path[0]
            if len(path)==1 and kind in ("object","sign"):
                if st["face"]!=next(d for d,b in DIRECTIONS.items() if b==a):
                    self.step(a)
                self.step(1)
                self.step(1)
                return self.settle(before["name"])
            if len(path)==1 and kind=="warp":
                self.step(a)
                self.step(a)
                return self.settle(before["name"])
            for attempt in range(3):
                self.step(a)
                now=self.state()
                if not now or now["pos"]!=pos or now["name"]!=st["name"] or now["phase"]!=5 or now["lock"]:break
            else:blocked.add((pos,a))
        return self.state()


def tier(s):
    if s["badge"] and s["defeated"]:return 100
    if s["dex"]:
        return 35 + 3*max(0,MAP_PROGRESS.get(s["name"],0)-3)+min(3,s["woods"])+min(2,s["petal"])
    if s["lab"]>=4:
        return 32+int(s["name"] in ("Route101","LittlerootTown","LittlerootTown_ProfessorBirchsLab"))+int(s["name"] in ("LittlerootTown","LittlerootTown_ProfessorBirchsLab"))
    if s["lab"]>=3:return 20+min(3,MAP_PROGRESS.get(s["name"],0))
    if s["party"]:return 14
    if s["route"]>=2:return 11
    if s["name"]=="Route101":return 10
    if s["town"]>=2:return 9
    return 8 if s["rival"] else int(s["clock"])*3


def signature(s):
    return (s["name"],s["pos"],s["rival"],s["town"],s["route"],s["lab"],s["party"],
            min(30,s["level"]),s["hp"]//3,s["petal"],s["woods"],s["dex"],s["badge"])


@dataclass
class Cell:
    snap: bytes
    tape: bytes
    plan: tuple
    state: dict


def explore(args):
    root=RUN/args.name
    root.mkdir(parents=True,exist_ok=True)
    game=Game()
    game.rng.seed(args.seed)
    prefix=(ROOT/"starter-diagnostics/mgba-town_after_rival.bin").read_bytes()
    resume = root/"best.bin"
    tape = resume.read_bytes() if resume.exists() else prefix
    if args.parent and not resume.exists():
        tape=(RUN/args.parent).read_bytes()
    for b in tape:game.step(b)
    if args.capture_battles:
        game.capture_battles=root/"battles"
        game.battle_saved=set()
    old = json.loads((root/"best.json").read_text()) if resume.exists() else {}
    game.trace=bytearray(tape)
    queue=[]
    serial=itertools.count()
    seen=set()
    visits=defaultdict(int)
    best=-1
    bestlevel=0
    first_party=None
    maps=set()
    start=time.monotonic()
    def register(plan):
        nonlocal best,bestlevel,first_party
        st=game.state()
        if not st or st["phase"]!=5 or st["lock"]:return
        key=signature(st)
        if key in seen:return
        seen.add(key)
        rank=tier(st)
        opts=game.options(st)
        if args.omit_triggers:
            opts={name:option for name,option in opts.items() if option[0]!="trigger"}
        if st["party"] and first_party is None:
            first_party=game.count-len(tape)
        if args.conditioning and rank<100:
            hp=st["hp"]/max(1,st["maxhp"])
            if st["name"].startswith("RustboroCity"):
                rank=55+min(13,max(0,st["level"]-12))+int(hp*5)
                if st["name"]=="RustboroCity_Gym":rank+=4 if hp>.5 else 0
                if "PokemonCenter_1F" in st["name"]:rank+=3 if hp<.7 else 1
        if st["name"]=="Route104" and st["woods"]>=1:
            for label,(kind,pos) in opts.items():
                if game.destination(label)=="RustboroCity" and game.path(st["pos"],pos,kind,set()):
                    rank=max(rank,54)
                    break
        level=st["level"]
        maps.add(st["name"])
        if rank>best or rank==best and level>bestlevel:
            best,bestlevel=rank,level
            raw=bytes(game.trace)
            label=f"tier-{rank:03}-level-{level:02}"
            (root/(label+".bin")).write_bytes(raw)
            (root/"best.bin").write_bytes(raw)
            meta=dict(tier=rank,decisions_from_boot=len(raw),real_decisions=game.count,state=st,plan=plan,source="self-discovered prefix and current-map exploration options")
            write_json(root/"best.json",meta)
            Image.fromarray(game.env.screen).save(root/(label+".png"))
            print("NEW",label,st["name"],len(raw),game.count,flush=True)
        cell=Cell(game.env.save(),bytes(game.trace),plan,st)
        health = st["hp"]/max(1,st["maxhp"])
        for name,option in opts.items():
            kind,pos=option
            count=visits[(st["name"],st["lab"],name)]
            dr=abs(st["pos"][0]-pos[0])+abs(st["pos"][1]-pos[1])
            preference={"edge":14,"trigger":13,"object":9,"warp":8,"sign":0}[kind]
            destination=game.destination(name)
            if destination and destination not in maps:
                preference+=10
            if destination in MAP_PROGRESS and st["dex"]:
                preference+=4*max(-2,min(3,MAP_PROGRESS[destination]-MAP_PROGRESS.get(st["name"],0)))
            if destination and st["hp"]*2<st["maxhp"] and "PokemonCenter_1F" in destination:
                preference+=15
            if args.conditioning and st["name"].startswith("RustboroCity"):
                if destination and "PokemonCenter_1F" in destination and health<.8:preference+=80
                if "PokemonCenter" in st["name"] and destination=="RustboroCity" and health>.8:preference+=70
                if health>.7 and destination=="RustboroCity_Gym":preference+=80
            priority=rank*15 + preference + min(18,st["level"])*1.4+health*3 - 12*count - .19*dr - .14*len(plan) + game.rng.uniform(0,8)
            heapq.heappush(queue,(-priority,next(serial),cell,name,option))
    try:
        if not old and not args.parent:assert game.state()["rival"] and game.state()["name"]=="LittlerootTown"
        if args.opening_test:
            game.limit=len(tape)+args.budget
        game.settle(game.state()["name"])
        register(tuple(old.get("plan",())))
        expansions=0
        while queue and game.count < (game.limit if args.opening_test else args.budget) and best<100:
            if args.opening_test and first_party is not None:break
            _,_,cell,name,option=heapq.heappop(queue)
            visit=(cell.state["name"],cell.state["lab"],name)
            if visits[visit]>8 and game.rng.random()<.95:continue
            visits[visit]+=1
            game.env.load(cell.snap)
            game.trace=bytearray(cell.tape)
            game.interact(name,option)
            register(cell.plan+((cell.state["name"],name),))
            expansions+=1
            if expansions%40==0:
                write_json(root/"status.json",dict(expansions=expansions,decisions=game.count,best=best,level=bestlevel,
                      cells=len(seen),queued=len(queue),maps=sorted(maps),elapsed_s=round(time.monotonic()-start,1)))
        print("END",game.count,best,len(queue),flush=True)
    except StepLimit:
        pass
    finally:
        if args.opening_test:
            write_json(root/"opening-test.json",dict(seed=args.seed,trigger_targets=not args.omit_triggers,
                budget_after_restore=args.budget,real_decisions_after_restore=game.count-len(tape),
                first_confirmed_party_after_option=first_party,archive_cells=len(seen)))
        game.env.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--name",default="explore-a")
    parser.add_argument("--budget",type=int,default=500000)
    parser.add_argument("--seed",type=int,default=726)
    parser.add_argument("--parent",help="A tape previously discovered by this exploration job")
    parser.add_argument("--conditioning",action="store_true",help="Give health and experience training credit around a discovered gym")
    parser.add_argument("--omit-triggers",action="store_true")
    parser.add_argument("--opening-test",action="store_true")
    parser.add_argument("--capture-battles",action="store_true")
    explore(parser.parse_args())
