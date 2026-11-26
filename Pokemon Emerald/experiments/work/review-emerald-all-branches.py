"""Review frequent training approaches using shared-prefix caches. All coordinates are audit-only."""
from collections import OrderedDict
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from earned_starts import EarnedStarts
from emerald import Emerald,MAPS
from pixel_env import game_paths
from train_earned_dreamer import write_json
spec=importlib.util.spec_from_file_location("distance_review",base/"work/monitor-emerald-frontier-approaches.py")
review=importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
root=review.root
out=review.out/"all-branches"
bankdir=root/"controls/dreamer-earned-resets/starts"
read=lambda p:json.loads(p.read_text())


def run():
    bank=EarnedStarts(bankdir)
    original=bank.tape
    measured=read(root/"controls/trainer-contacts/highest-return-review/measured.json")
    best=read(review.out/"closest.json")
    assert measured["input_sha256"]==bank.info["source_sha256"] and measured["nearest_template_manhattan_tiles"]>best["distance"]
    obj,=[o for o in read(base/"work/pokeemerald/data/maps/Route103/map.json")["object_events"] if o.get("local_id")=="LOCALID_ROUTE103_RIVAL"]
    template=(obj["x"]+7,obj["y"]+7)
    game=Emerald(*game_paths())
    checked=read(out/"index.json") if (out/"index.json").exists() else {}
    shared,cache={},OrderedDict()
    total=0
    try:
        shared[0]=(game.boot,game.descriptor(),(999999,None,False,None,None))
        for i,entry in enumerate(bank.records):
            bank.restore(game,i)
            key=game.descriptor()
            route=key[1]>=0 and MAPS[key[1]][key[2]]=="Route103"
            shared[entry["prefix"]]=(game.save(),key,(999999,None,route,None,None))
        write_json(out/"ready.json",dict(original_bank=bank.info["source_sha256"],shared_starts=sorted(shared),
            rule="the complete shared original prefix was independently checked and cannot improve the existing three-tile record",
            actor_receives_audit=False))
        if (out/"closest.json").exists():
            best=min([best,read(out/"closest.json")],key=lambda r:r["distance"])
        while True:
            if best["distance"]<=1:
                write_json(out/"adjacency-complete.json",best)
                return
            candidates=[]
            for arm in review.family.iterdir():
                settings=arm/"settings.json"
                if not settings.is_file() or read(settings).get("source_sha256")!=bank.info["source_sha256"]:
                    continue
                for file in arm.glob("discoveries/work-*/record.json"):
                    row=read(file)
                    if row["tape_sha256"] not in checked:
                        candidates.append((file.stat().st_mtime_ns,file,row))
            for _,file,row in sorted(candidates,key=lambda item:item[0],reverse=True):
                sha=row["tape_sha256"]
                if sha in checked:
                    continue
                audit=file.with_name("game.json")
                data=read(audit)
                if "maps_first_reached" in data and "Route103" not in data["maps_first_reached"]:
                    checked[sha]=dict(route=False)
                    continue
                tape=file.with_name("game.bin").read_bytes()
                assert hashlib.sha256(tape).hexdigest()==sha and len(tape)==row["length"]
                common=next((j for j,(a,b) in enumerate(zip(original,tape)) if a!=b),min(len(original),len(tape)))
                start=max(n for n in shared if n<=common)
                snapshot=shared[start]
                prefixes={}
                hash_=hashlib.sha256()
                for n in range(512,len(tape)+1,512):
                    hash_.update(tape[n-512:n])
                    prefix=hash_.digest()
                    prefixes[n]=prefix
                    if n>start and prefix in cache:
                        start,snapshot=n,cache[prefix]
                        cache.move_to_end(prefix)
                blob,key,state=snapshot
                game.load(blob)
                assert game.descriptor()==key
                for n,button in enumerate(tape[start:],start+1):
                    state=review.observe(game,key,state,n-1,template)
                    _,key=game.step(button)
                    if n in prefixes:
                        cache[prefixes[n]]=(game.save(),key,state)
                        if len(cache)>384:
                            cache.popitem(last=False)
                total+=len(tape)-start
                checked[sha]=dict(route=True,source=str(audit),actual_suffix=len(tape)-start,
                                  candidate_distance=None if state[1] is None else state[0])
                if state[0]<best["distance"]:
                    trial=out/sha
                    trial.mkdir(exist_ok=True)
                    proof=trial/"cold-distance.json"
                    with (trial/"cold-check.log").open("a") as log:
                        proc=subprocess.run([sys.executable,str(base/"work/diagnose-emerald-rival-approach.py"),str(audit),str(proof)],
                                            stdout=log,stderr=subprocess.STDOUT)
                    if proc.returncode==0:
                        cold=read(proof)
                        if cold["in_game_restores"]==0 and cold["input_sha256"]==sha and cold["nearest_template_manhattan_tiles"]<best["distance"]:
                            best=dict(distance=cold["nearest_template_manhattan_tiles"],decision=cold["first_button_at_nearest"],
                                      audit=str(audit),verification=str(proof))
                            write_json(out/"closest.json",best)
                            if best["distance"]<=1:
                                write_json(out/"adjacency-complete.json",best)
                                return
                write_json(out/"status.json",dict(checked=len(checked),latest=sha,suffix_steps=total,private_cached_prefixes=len(cache),
                           best_verified_distance=best["distance"],updated=time.time()))
                write_json(out/"index.json",checked)
            write_json(out/"index.json",checked)
            time.sleep(30)
    finally:
        game.close()


if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
    out.mkdir(parents=True,exist_ok=True)
    with (out/"job.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
