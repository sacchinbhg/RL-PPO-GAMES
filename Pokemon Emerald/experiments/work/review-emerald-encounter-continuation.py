"""Replay an already-recorded training continuation from its verified ancestor to inspect the battle."""
import hashlib
import json
import os
from pathlib import Path
import sys
from PIL import Image

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from emerald import Emerald
from earned_starts import EarnedStarts
from pixel_env import game_paths
from reward_branch_search import visual_code,write_json

root=project/"runs/pixels-only-01/official-wsl"
arm=root/"paired-self-earned-frontiers/every-sealed-endpoint-01"
out=root/"controls/endpoint-button-sweep/first-encounter-followup"
read=lambda p:json.loads(p.read_text())

if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
    observed=min((arm/"first-trainers").glob("*/game.bin"),key=lambda p:p.stat().st_mtime_ns)
    tape0=observed.read_bytes()
    target=int(observed.parent.name.split("-")[-1])
    found=[]
    for file in arm.glob("discoveries/*/record.json"):
        row=read(file)
        if row["length"]<=len(tape0):
            continue
        tape=file.with_name("game.bin").read_bytes()
        if tape.startswith(tape0):
            found.append((len(tape),file,row,tape))
    _,file,row,tape=max(found,key=lambda item:item[0])
    assert hashlib.sha256(tape).hexdigest()==row["tape_sha256"]
    bank=EarnedStarts(root/"controls/dreamer-earned-resets/starts")
    start,index=max((e["prefix"],i) for i,e in enumerate(bank.records) if e["prefix"]<len(tape0) and tape.startswith(bank.tape[:e["prefix"]]))
    out.mkdir(parents=True,exist_ok=True)
    game=Emerald(*game_paths())
    try:
        image,score,_=bank.restore(game,index)
        total=bank.records[index]["scalar_return_cents"]
        first=exit_at=outcome=party_entry=None
        health_history=[]
        unresolved=False
        for n,b in enumerate(tape[start:],start+1):
            image,key=game.step(b)
            rew,_=score.read(game,key)
            total+=round(100*rew)
            if n==len(tape0)-1:
                party_entry=dict(level=game.ram(0x020244ec+84,1)[0],hp=int.from_bytes(game.ram(0x020244ec+86,2),"little"),
                                 maximum=int.from_bytes(game.ram(0x020244ec+88,2),"little"))
            if n<len(tape0):
                continue
            if key[0]==6 and int.from_bytes(game.ram(0x02022fec,4),"little")&8 and int.from_bytes(game.ram(0x02038bca,2),"little")==target:
                first=first or n
            if first and not exit_at:
                value=game.ram(0x0202433a,1)[0]
                if key[0]==6:
                    combat=[]
                    for battler in (0,1):
                        mon=0x02024084+0x58*battler
                        combat.append([int.from_bytes(game.ram(mon+0x28,2),"little"),int.from_bytes(game.ram(mon+0x2c,2),"little")])
                    if all(0<=hp<=maximum<=999 and maximum>0 for hp,maximum in combat):
                        if not health_history or health_history[-1]["hp"]!=combat:
                            health_history.append(dict(decision=n,hp=combat))
                if value==0:
                    unresolved=True
                elif unresolved and outcome is None:
                    outcome=dict(decision=n,value=value)
                    Image.fromarray(game.screen).save(out/"outcome.png")
                if key[0]==5 and game.verified_story(key):
                    exit_at=n
            if first and n in (first+50,first+120):
                Image.fromarray(game.screen).save(out/f"battle-{n-first}.png")
        assert total==row["return_cents"] and visual_code(image).hex()==row["screen"]
        assert first==len(tape0)
        Image.fromarray(game.screen).save(out/"last.png")
        write_json(out/"measured.json",dict(steps=len(tape),first_trainer_battle=first,trainer_id=target,
            first_resolved_outcome=outcome,first_overworld_after=exit_at,source_sha256=row["tape_sha256"],source=str(file.with_name("game.json")),
            confirmed_final_image_and_return=True,replayed_from_verified_training_prefix=start,independent_cold_replay=False))
        write_json(out/"battle-health.json",dict(entry_party=party_entry,hp_changes=health_history,source_sha256=row["tape_sha256"],
                   human_review_only=True,order=["player","opponent"]))
    finally:
        game.close()
