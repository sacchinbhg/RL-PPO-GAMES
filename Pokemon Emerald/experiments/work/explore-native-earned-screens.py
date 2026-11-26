"""Collect independent native-port discoveries from RGB and scalar rewards, with original-ROM transfer tested separately."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from emerald import MAPS
from native_emerald import NativeEmerald
from reward_branch_search import Archive,Point,visual_code
from screen_explorer import ScreenExplorer
from train_earned_dreamer import write_json
root=project/"runs/pixels-only-01/official-wsl"
read=lambda p:json.loads(p.read_text(encoding="utf-8-sig"))


class NativeGame:
    def __init__(self,output,training=True,fast_restore=False):
        cfg=read(project/"local.json")
        self.game=NativeEmerald(cfg["native_core"],cfg["rom"])
        self.initial=self.game.boot_image
        self.slots={}
        self.output=output
        self.training=training
        self.fast_restore=fast_restore
        self.render_fallbacks=0
        self.paid=set()
        self.cells=set()
        self.events={}
        self.maps={}
        self.ever=set()
        self.total=0
        self.cold_buttons=0
        self.input_hash=hashlib.sha256(Path(cfg["native_core"]).read_bytes()).hexdigest()

    def record(self,dest):
        dest.mkdir(parents=True,exist_ok=True)
        tape=bytes(self.game.actions)
        (dest/"game.bin").write_bytes(tape)
        info=dict(backend="native-port",from_blank=True,training_only=self.training,steps=len(tape),events=self.events,
                  maps_first_reached=self.maps,first_visit_cells=len(self.cells),return_cents=self.total,
                  tape_sha256=hashlib.sha256(tape).hexdigest())
        write_json(dest/"game.json",info)
        return info

    def step(self,button):
        pic,key=self.game.step(button)
        amount=0
        valid=self.game.verified_story(key)
        name=None
        if valid is not None:
            rival,party=valid
            route=self.game.inspect()[10]
            milestones=(("bedroom",key[5]>=4,100),("clock",bool(key[6]),300),("sent_to_meet_rival",key[5]>=6 and bool(key[6]),100),
                        ("met_rival",rival,400),("route101",rival and route>=1,400),("heard_birch",rival and route>=2,200),
                        ("starter",party,1000),("stone_badge",party and bool(key[8]&1),2000))
            for name_,condition,value in milestones:
                if condition and name_ not in self.paid:
                    self.paid.add(name_)
                    self.events[name_]=len(self.game.actions)
                    amount+=value
            if key[0]==5:
                name=MAPS[key[1]][key[2]]
                self.maps.setdefault(name,len(self.game.actions))
                if key[1:5] not in self.cells:
                    self.cells.add(key[1:5])
                    amount+=2
        self.total+=amount
        if name is not None and name not in self.ever:
            self.ever.add(name)
            target=self.output/"first-maps"/name
            if not (target/"game.json").exists():
                self.record(target)
        return pic,amount,"stone_badge" in self.paid

    def command(self,name,slot):
        if name=="save":
            self.slots[slot]=(self.game.save(),self.game.inspect(),hashlib.sha256(self.game.small_screen()).digest(),
                              copy.deepcopy((self.paid,self.cells,self.events,self.maps,self.total)))
            return
        if name!="restore":
            raise ValueError(name)
        blob,state,image,score=self.slots[slot]
        self.game.load(blob,render_intermediate=not self.fast_restore)
        current=self.game.small_screen()
        if self.game.inspect()!=state or hashlib.sha256(current).digest()!=image:
            diagnostic=dict(slot=int(slot),buttons=len(self.game.actions),expected_state=state,
                            expected_image=image.hex(),first_state=self.game.inspect(),
                            first_image=hashlib.sha256(current).hexdigest(),fast=self.fast_restore)
            if self.fast_restore:
                self.game.load(blob,render_intermediate=True)
                current=self.game.small_screen()
                diagnostic.update(stock_state=self.game.inspect(),stock_image=hashlib.sha256(current).hexdigest())
                self.render_fallbacks+=1
            exact=self.game.inspect()==state and hashlib.sha256(current).digest()==image
            diagnostic["recovered_exactly"]=exact
            dest=self.output/"restore-checks"/f"{hashlib.sha256(blob).hexdigest()}.json"
            dest.parent.mkdir(parents=True,exist_ok=True)
            write_json(dest,diagnostic)
            if not exact:
                raise RuntimeError(f"Native replay did not reproduce a stored state; diagnostics in {dest}")
            self.fast_restore=False
        self.paid,self.cells,self.events,self.maps,self.total=copy.deepcopy(score)
        self.cold_buttons+=len(self.game.actions)
        return current


def run(limit,seed,output,native_source=None,recovery_attempt=0):
    output.mkdir(parents=True,exist_ok=True)
    if (output/"completed.json").exists():
        return
    if native_source:
        native_source=native_source.resolve()
        family=(root/"native-earned-openings").resolve()
        if not native_source.is_relative_to(family) or native_source.name!="game.json":
            raise ValueError("Expected a native self-play training episode")
        record=read(native_source)
        assert record["backend"]=="native-port" and record["from_blank"] and record["training_only"]
        source=native_source.with_suffix(".bin")
    else:
        plan=read(root/"controls/reward-frontier-expansion/plan.json")
        record=plan["selected"]
        source=root/Path(plan["source"]).name/record["audit"]/"game.bin"
    original=source.read_bytes()
    assert hashlib.sha256(original).hexdigest()==record["tape_sha256"] and record["training_only"]
    cfg=dict(seed=seed,new_native_decisions=limit,source_training_tape=record["tape_sha256"],frames_per_decision=24,
             observation="RGB, scalar reward, terminal; screen transition counts",backend="native game port",
             reward="ordinary early story milestones, 0.02 per first native cell and first badge; no lab, later story, or level bonus",
             distinct_from_original_rom_evaluation=True)
    if native_source:
        cfg["source_backend"]="native-port"
        proof=read(root/"controls/native-restore-rendering/result.json")
        core=Path(read(project/"local.json")["native_core"])
        cfg["render_only_restored_frame"]=bool(proof["eligible_for_native_training"] and
            proof["native_engine_sha256"]==hashlib.sha256(core.read_bytes()).hexdigest())
    manifest=output/"registration.json"
    if manifest.exists() and read(manifest)!=cfg:
        raise ValueError("Conflicting experiment settings")
    write_json(manifest,cfg)
    old=read(output/"best.json") if (output/"best.json").exists() else None
    tape=(output/old["relative"]/"game.bin").read_bytes() if old else original
    if old and hashlib.sha256(tape).hexdigest()!=old["tape_sha256"]:
        raise ValueError("Native restart tape changed")
    work=old["work"] if old else 0
    branches=old["branches"] if old else 0
    if recovery_attempt:
        previous=[p for p in [output/"status.json",*output.glob("engine-errors/*.json")] if p.is_file()]
        for path in previous:
            counters=read(path)
            work=max(work,counters["work"])
            branches=max(branches,counters["branches"])
    extra=[recovery_attempt] if recovery_attempt else []
    model=ScreenExplorer(np.random.SeedSequence([seed,work,0,*extra]))
    rng=np.random.default_rng([seed,work,1,*extra])
    env=NativeGame(output,fast_restore=cfg.get("render_only_restored_frame",False) and not recovery_attempt)
    try:
        archive=Archive(env,rng,48)
        state=visual_code(env.initial)
        total=0
        best=-1
        for i,b in enumerate(tape,1):
            pic,cents,done=env.step(b)
            following=visual_code(pic)
            model.visits[state]+=1
            model.transitions.setdefault(state,{})
            model.observe(state,b,following,cents/100)
            state=following
            total+=cents
            if cents>0:
                archive.offer(Point(state,total,i))
                best=max(best,total)
            if i%1024==0:
                archive.offer_visual(Point(state,total,i),model.visits,best)
        if old:
            evidence=read(output/old["relative"]/"game.json")
            matches=(total==old["return_cents"] and old["screen"]==state.hex() and env.events==evidence["events"]
                     and env.maps==evidence["maps_first_reached"] and len(env.cells)==evidence["first_visit_cells"])
            dest=output/"resume-checks"/f"work-{work:09d}.json"
            dest.parent.mkdir(parents=True,exist_ok=True)
            write_json(dest,dict(reproduced=matches,buttons=i,actual_return_cents=total,actual_events=env.events,
                       actual_maps=env.maps,actual_screen=state.hex(),actual_cells=len(env.cells),source=old["tape_sha256"],
                       recovery_attempt=recovery_attempt,charged_work=work,fast_restore=env.fast_restore))
            if not matches:
                raise RuntimeError(f"Native cold replay disagrees with the saved training history; diagnostics in {dest}")
        if native_source and old is None and (total!=record["return_cents"] or env.events!=record["events"] or env.maps!=record["maps_first_reached"]):
            raise RuntimeError("Native boot replay disagrees with its own training discovery")
        since=time.monotonic()
        initial=work
        if not old:
            write_json(output/"initialization.json",dict(buttons=i,warm_return_cents=total,events=dict(env.events),native_engine_sha256=env.input_hash))
        while work<limit:
            start=archive.start()
            state=start.screen
            total=start.total
            depth=start.length
            branches+=1
            branch_budget=int(rng.choice([1024,2048,4096,8192]))
            elapsed=0
            while elapsed<branch_budget and work<limit:
                button=model.choose(state)
                duration=min(int(rng.choice([1,1,1,2,4,8])),branch_budget-elapsed,limit-work)
                for tick in range(duration):
                    if tick:
                        model.visits[state]+=1
                        model.transitions.setdefault(state,{})
                    pic,cents,done=env.step(button)
                    new=visual_code(pic)
                    model.observe(state,button,new,cents/100)
                    state=new
                    total+=cents
                    work+=1
                    depth+=1
                    elapsed+=1
                    point=Point(state,total,depth)
                    if cents:
                        archive.offer(point)
                    if total>best and (not old or total>=old["return_cents"]+25 or cents>=100 or done):
                        folder=f"discoveries/work-{work:09d}"
                        info=env.record(output/folder)
                        assert info["return_cents"]==total and info["steps"]==depth
                        old=dict(relative=folder,return_cents=total,screen=state.hex(),work=work,branches=branches,tape_sha256=info["tape_sha256"])
                        write_json(output/"best.json",old)
                    best=max(best,total)
                    if work%1024==0:
                        archive.offer_visual(point,model.visits,best)
                    if work%4096==0 or done:
                        write_json(output/"status.json",dict(work=work,branches=branches,best=best,current_length=depth,
                            full_history_buttons_replayed_for_restores=env.cold_buttons,distinct_screens=len(model.visits),
                            new_buttons_per_second=(work-initial)/(time.monotonic()-since)))
                    if done:
                        write_json(output/"completed.json",dict(work=work,native_badge=True,requires_original_rom_verification=True))
                        return
                    if cents:
                        break
            archive.offer(point,recent=True)
        write_json(output/"completed.json",dict(work=work,best=best,native_badge=False))
    except OSError as error:
        path=output/"engine-errors"/f"work-{work:09d}-attempt-{recovery_attempt}.json"
        path.parent.mkdir(parents=True,exist_ok=True)
        write_json(path,dict(work=work,branches=branches,engine_error=str(error),recovery_attempt=recovery_attempt,
                            fast_restore=env.fast_restore,completed_actions=len(env.game.actions)))
        raise
    finally:
        env.game.close()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--steps",type=int,default=1048576)
    p.add_argument("--seed",type=int,default=173743)
    p.add_argument("--earned-native",type=Path)
    p.add_argument("--recovery-attempt",type=int,default=0)
    a=p.parse_args()
    run(a.steps,a.seed,root/"native-screen-exploration"/f"seed-{a.seed}",a.earned_native,a.recovery_attempt)
