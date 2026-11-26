"""Review sealed training tapes efficiently. Confirm any closer approach independently from boot."""
import bisect
import fcntl
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from emerald import Emerald,MAPS
from pixel_env import game_paths
from reward_branch_search import write_json
root=project/"runs/pixels-only-01/official-wsl"
family=root/"paired-self-earned-frontiers"
out=root/"controls/trainer-contacts/frontier-reviews"
read=lambda p:json.loads(p.read_text())

def observe(game,key,memory,n,template,position=True):
    nearest,at,on_route,battle,trainer=memory
    if key[0]==5 and key[1]>=0 and game.verified_story(key) is not None:
        on_route=MAPS[key[1]][key[2]]=="Route103"
        if on_route and position:
            index=game.ram(0x02037590+5,1)[0]
            if index<16:
                obj=game.ram(0x02037350+36*index,36)
                if obj[0]&1 and obj[2]&1:
                    x,y=struct.unpack_from("<hh",obj,16)
                    distance=abs(x-template[0])+abs(y-template[1])
                    if distance<nearest:
                        nearest,at=distance,n+1
    if key[0]==6 and on_route and battle is None:
        if int.from_bytes(game.ram(0x02022fec,4),"little")&8:
            found=int.from_bytes(game.ram(0x02038bca,2),"little")
            if found in (520,523,526,529,532,535):
                battle,trainer=n,found
    return nearest,at,on_route,battle,trainer

def main():
    plan=read(root/"controls/reward-frontier-expansion/plan.json")
    source=Path(plan["source"])/plan["selected"]["audit"]/"game.bin"
    original=source.read_bytes()
    assert hashlib.sha256(original).hexdigest()==plan["selected"]["tape_sha256"]
    reviewed=read(root/"controls/trainer-contacts/highest-return-review/measured.json")
    assert reviewed["input_sha256"]==plan["selected"]["tape_sha256"]
    rival,=[o for o in read(base/"work/pokeemerald/data/maps/Route103/map.json")["object_events"] if o.get("local_id")=="LOCALID_ROUTE103_RIVAL"]
    template=(rival["x"]+7,rival["y"]+7)
    game=Emerald(*game_paths())
    try:
        game.load(game.boot)
        key=game.descriptor()
        memory=(999999,None,False,None,None)
        saved={0:(game.boot,key,memory)}
        for n,b in enumerate(original,1):
            memory=observe(game,key,memory,n-1,template)
            _,key=game.step(b)
            if n%256==0 or n==len(original):
                saved[n]=(game.save(),key,memory)
            if n%4096==0:
                write_json(out/"status.json",dict(warming=n,expected=len(original)))
        assert (memory[0],memory[1])==(reviewed["nearest_template_manhattan_tiles"],reviewed["first_button_at_nearest"])
        write_json(out/"cached-source.json",dict(source_sha256=reviewed["input_sha256"],matched_closest=memory[0],
                   matched_decision=memory[1],checkpoints=len(saved),buttons=len(original)))
        indices=sorted(saved)
        closest=read(out/"closest.json")["distance"] if (out/"closest.json").exists() else memory[0]
        checked=set()
        suffix_steps=0
        while True:
            for arm in family.glob("systematic-reward-frontiers*"):
                for record in sorted(arm.glob("discoveries/*/record.json")):
                    if record in checked:
                        continue
                    row=read(record)
                    audit=record.with_name("game.json")
                    tape=audit.with_suffix(".bin").read_bytes()
                    sha=hashlib.sha256(tape).hexdigest()
                    if sha!=row["tape_sha256"] or len(tape)!=row["length"]:
                        continue
                    common=next((i for i,(a,b) in enumerate(zip(original,tape)) if a!=b),min(len(original),len(tape)))
                    index=indices[bisect.bisect_right(indices,common)-1]
                    blob,key,state=saved[index]
                    game.load(blob)
                    assert game.descriptor()==key
                    for n,b in enumerate(tape[index:],index+1):
                        state=observe(game,key,state,n-1,template)
                        _,key=game.step(b)
                    state=observe(game,key,state,len(tape),template,position=False)
                    suffix_steps+=len(tape)-index
                    if state[0]<closest or (state[3] is not None and not (out/"first-rival-encounter.json").exists()):
                        trial=out/sha
                        trial.mkdir(exist_ok=True)
                        write_json(trial/"candidate.json",dict(audit=str(audit),source_common_buttons=common,
                                  distance=state[0],closest_decision=state[1],potential_battle=state[3]))
                        if state[0]<closest:
                            replay=trial/"cold-distance.json"
                            if not replay.exists():
                                subprocess.run([sys.executable,str(base/"work/diagnose-emerald-rival-approach.py"),str(audit),str(replay)],check=True)
                            cold=read(replay)
                            assert (cold["nearest_template_manhattan_tiles"],cold["first_button_at_nearest"])==state[:2]
                            assert cold["in_game_restores"]==0 and cold["input_sha256"]==sha
                            closest=state[0]
                            write_json(out/"closest.json",dict(distance=closest,decision=state[1],audit=str(audit),verification=str(replay)))
                        if state[3] is not None and not (out/"first-rival-encounter.json").exists():
                            replay=trial/"cold-battles.json"
                            subprocess.run([sys.executable,str(base/"work/diagnose-emerald-battles.py"),str(audit),str(replay)],check=True)
                            cold=read(replay)
                            assert cold["in_game_restores"]==0 and state[4] in cold["observed_route103_rival_ids"]
                            write_json(out/"first-rival-encounter.json",dict(decision=state[3],opponent=state[4],audit=str(audit),verification=str(replay)))
                    checked.add(record)
            write_json(out/"status.json",dict(time=time.time(),checked_sealed_tapes=len(checked),suffix_buttons_replayed=suffix_steps,
                       best_verified_distance=closest,reporting_only=True))
            time.sleep(45)
    finally:
        game.close()

if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),
        EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    out.mkdir(parents=True,exist_ok=True)
    with (out/"job.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
