"""Original-ROM environment for the clean experiment. Only RGB, scalar reward and done cross the pipe."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import traceback
import numpy as np
from emerald import Emerald

ROOT=Path(__file__).resolve().parent
CLEAN=ROOT/"runs"/"pixels-only-01"


def game_paths():
    core = os.environ.get("EMERALD_CORE")
    rom = os.environ.get("EMERALD_ROM")
    if not core or not rom:
        config = json.loads((ROOT / "local.json").read_text(encoding="utf-8-sig"))
        core = core or config["core"]
        rom = rom or config["rom"]
    return Path(core), Path(rom)


class Score:
    def __init__(self):
        self.paid=set()
        self.cells=set()
        self.high_level=5
        self.events={}
        self.decisions=0

    def read(self,env,key):
        self.decisions+=1
        verified=env.verified_story(key)
        if verified is None:return 0.,False
        rival,party=verified
        def read(addr,n):return int.from_bytes(env.ram(addr,n),"little")
        sb=read(0x03005d8c,4)
        var=lambda i:read(sb+0x139c+2*i,2)
        flag=lambda i:bool(read(sb+0x1270+i//8,1)&(1<<(i%8)))
        route=var(0x60)
        lab=var(0x84)
        badge=bool(party and flag(0x867) and flag(0x4f0))
        rewards=(("bedroom",key[5]>=4,1),("clock",bool(key[6]),3),
                 ("sent_to_meet_rival",key[5]>=6 and bool(key[6]),1),("met_rival",rival,4),
                 ("route101",rival and route>=1,4),("heard_birch",rival and route>=2,2),
                 ("starter",party,10),("lab_handoff",party and lab>=3,4),
                 ("rival_battle",party and lab>=4,6),("pokedex",party and flag(0x861),5),
                 ("wally",party and var(0x57)>=3,5),("woods",party and var(0x98)>=1,5),
                 ("stone_badge",badge,20))
        reward=0.
        for name,done,value in rewards:
            if done and name not in self.paid:
                self.paid.add(name)
                self.events[name]=self.decisions
                reward+=value
        if key[0]==5 and len(self.cells)<50 and key[1:5] not in self.cells:
            self.cells.add(key[1:5])
            reward+=.02
        if party:
            level=read(0x020244ec+84,1)
            if self.high_level<level<=100:
                reward+=.25*(min(level,20)-min(20,self.high_level))
                self.high_level=level
        return reward,badge


def worker(conn,n):
    env=None
    try:
        env=Emerald(*game_paths())
        episode=0
        score=Score()
        tape=bytearray()
        conn.send(env.boot_image)
        while True:
            cmd,arg=conn.recv()
            if cmd=="close":break
            if cmd=="reset":
                env.load(env.boot)
                episode+=1
                score=Score()
                tape=bytearray()
                conn.send(env.boot_image)
            elif cmd=="act":
                pic,key=env.step(arg)
                tape.append(arg)
                reward,done=score.read(env,key)
                conn.send((pic,reward,done))
            elif cmd=="audit":
                # Only writes reports for humans. Never returns game information to the learner.
                target=Path(arg)
                target.mkdir(parents=True,exist_ok=True)
                label=f"worker-{n:02}-episode-{episode:05}"
                (target/(label+".bin")).write_bytes(tape)
                (target/(label+".json")).write_text(json.dumps(dict(events=score.events,
                    steps=score.decisions,first_visit_cells=len(score.cells)),indent=2))
                conn.send(True)
            else:raise ValueError(cmd)
    except BaseException:
        try:conn.send({"worker_error":traceback.format_exc()})
        except (BrokenPipeError,EOFError):pass
    finally:
        if env:env.close()
        conn.close()


class PixelPool:
    def __init__(self,count):
        self.pipes=[]
        self.processes=[]
        ctx=mp.get_context("spawn")
        for i in range(count):
            front,back=ctx.Pipe()
            proc=ctx.Process(target=worker,args=(back,i),daemon=True)
            proc.start()
            back.close()
            self.pipes.append(front)
            self.processes.append(proc)
        self.initial=np.stack([self.recv(i) for i in range(count)])

    def recv(self,i):
        pipe=self.pipes[i]
        if not pipe.poll(90): raise TimeoutError(f"Pixel worker {i}")
        value=pipe.recv()
        if isinstance(value,dict):raise RuntimeError(value)
        return value

    def command(self,i,cmd,arg=None):
        self.pipes[i].send((cmd,arg))
        return self.recv(i)

    def step(self,actions):
        for pipe,a in zip(self.pipes,actions):pipe.send(("act",int(a)))
        images,rewards,ends=zip(*(self.recv(i) for i in range(len(actions))))
        return np.stack(images),np.asarray(rewards,dtype=np.float32),np.asarray(ends,dtype=bool)

    def close(self):
        for pipe in self.pipes:
            try:pipe.send(("close",None))
            except (OSError,EOFError):pass
            pipe.close()
        for process in self.processes:
            process.join(2)
            if process.is_alive():process.terminate()
