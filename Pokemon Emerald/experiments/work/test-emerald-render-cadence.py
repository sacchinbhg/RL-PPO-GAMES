"""Measure stock mGBA rendering every 24th frame while still executing all 24 hardware frames per action."""
import argparse
import copy
import ctypes as c
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from emerald import Emerald,REPEAT,verify_rom
from pixel_env import Score,game_paths
from reward_branch_search import write_json

root=project/"runs/pixels-only-01/official-wsl"
source=root/"paired-self-earned-frontiers/visual-slots-0/discoveries/work-0000139234/game.bin"
group=os.environ.get("EMERALD_RENDER_TRAINING_DATA")
if group:
    dataset=Path(group)
    manifest=json.loads((dataset/"manifest.json").read_text())
    source,=(dataset/"audit").glob("*.bin")
    if manifest["frozen_evaluation_data_used"] or manifest["source_sha256"]!=hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError("Expected a verified training trajectory")

class Variable(c.Structure):
    _fields_=[("key",c.c_char_p),("value",c.c_void_p)]

class Candidate(Emerald):
    value=c.create_string_buffer(b"23")
    requests=0
    def environment(self,cmd,data):
        if cmd&0xffff==15 and data:
            var=Variable.from_address(data)
            if var.key==b"mgba_frameskip":
                var.value=c.addressof(self.value)
                self.requests+=1
                return True
        return super().environment(cmd,data)

def worker(variant,steps,path):
    core,rom=game_paths()
    verify_rom(rom)
    buttons=source.read_bytes()[:steps]
    assert len(buttons)==steps and REPEAT==24
    game=(Emerald if variant=="stock" else Candidate)(str(core),str(rom))
    observation=[]
    state=[]
    rewards=[]
    def record(b):
        image,key=game.step(b)
        reward,done=score.read(game,key)
        observation.append(hashlib.blake2b(image,digest_size=16).hexdigest()+hashlib.blake2b(game.screen,digest_size=16).hexdigest())
        state.append(key)
        rewards.append((round(100*reward),int(done)))
    try:
        game.load(game.boot)
        score=Score(True)
        fork=steps-512
        clock=time.perf_counter()
        cpu=time.process_time()
        for n,b in enumerate(buttons,1):
            record(b)
            if n==fork:
                saved=game.save()
                old=copy.deepcopy(score)
        game.load(saved)
        score=copy.deepcopy(old)
        for b in buttons[fork:]:
            record(b)
        cpu=time.process_time()-cpu
        clock=time.perf_counter()-clock
        assert variant=="stock" or game.requests>=1
        np.savez(path,obs=observation,state=state,rewards=rewards)
        write_json(path.with_suffix(".json"),dict(cpu=cpu,wall=clock,actual_frames=len(observation)*REPEAT,
                   actual_decisions=len(observation),frameskip_option_queries=getattr(game,"requests",0),
                   own_restore_images_match=observation[fork:steps]==observation[steps:],
                   own_restore_game_state_matches=state[fork:steps]==state[steps:],
                   own_restore_rewards_match=rewards[fork:steps]==rewards[steps:]))
    finally:
        game.close()

def compare(steps,order):
    out=root/"controls/last-frame-render"
    if group:
        out=out/manifest["source_sha256"][:12]
    out.mkdir(parents=True,exist_ok=True)
    marker=out/f"comparison-{steps}.json"
    if marker.exists():
        print(marker.read_text(),flush=True)
        return
    scratch_root=base/"work/emerald-render-traces"/f"frames-{steps}-{len(order)}"
    if group:
        scratch_root=scratch_root/manifest["source_sha256"][:12]
    scratch_root.mkdir(parents=True,exist_ok=True)
    tape_digest=hashlib.sha256(source.read_bytes()).hexdigest()
    core,rom=game_paths()
    config=dict(steps=steps,order=order,training_tape_sha256=tape_digest,
                rom_sha256=hashlib.sha256(rom.read_bytes()).hexdigest(),core_sha256=hashlib.sha256(core.read_bytes()).hexdigest())
    with (scratch_root/"job.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        setup=scratch_root/"settings.json"
        if setup.exists() and json.loads(setup.read_text())!=config:
            raise ValueError("Stored renderer traces have a different input or emulator")
        write_json(setup,config)
        results={}
        reference=None
        for index,variant in enumerate(order):
            directory=scratch_root/f"run-{index}-{variant}"
            scratch=directory/"trace.npz"
            if not directory.exists():
                with tempfile.TemporaryDirectory(dir=scratch_root) as temp:
                    draft=Path(temp)/"measurement"
                    draft.mkdir()
                    subprocess.run([sys.executable,"-u",__file__,"--worker",variant,"--steps",str(steps),
                                    "--output",str(draft/"trace.npz")],check=True)
                    draft.rename(directory)
            trace=np.load(scratch,allow_pickle=False)
            if reference is None:
                reference={k:trace[k] for k in trace.files}
            row=json.loads(scratch.with_suffix(".json").read_text())
            for k,old in reference.items():
                current=trace[k]
                equal=(current==old).reshape((len(old),-1)).all(axis=1)
                differences=np.flatnonzero(~equal)
                row[k+"_differences"]=len(differences)
                row[k+"_first_difference"]=int(differences[0]+1) if len(differences) else None
            trace.close()
            results.setdefault(variant,[]).append(row)
            write_json(out/f"progress-{steps}.json",dict(complete=index+1,order=order,measurements=results))
        speed=float(np.median([r["cpu"] for r in results["stock"]])/np.median([r["cpu"] for r in results["cadence"]]))
        accepted=all(not any(row[k+"_differences"] for k in ("obs","state","rewards"))
                     and all(row[k] for k in ("own_restore_images_match","own_restore_game_state_matches","own_restore_rewards_match"))
                     for group in results.values() for row in group)
        write_json(marker,dict(measurements=results,cpu_speedup=speed,all_images_states_and_rewards_match=accepted,
                   images="128-bit BLAKE2b digests for both 40x60 policy RGB and native 240x160 RGB at each decision",
                   buttons_held_frames=14,hardware_frames_per_decision=24,source_training_tape=hashlib.sha256(source.read_bytes()).hexdigest()))
        print(marker.read_text(),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--worker",choices=["stock","cadence"])
    p.add_argument("--steps",type=int,default=2048)
    p.add_argument("--output",type=Path)
    p.add_argument("--long",action="store_true")
    a=p.parse_args()
    if a.worker:
        worker(a.worker,a.steps,a.output)
    else:
        compare(a.steps,["stock","cadence"] if a.long else ["stock","cadence","cadence","stock"])
