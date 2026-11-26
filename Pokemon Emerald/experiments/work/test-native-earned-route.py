"""Check whether the current highest-return original game can initialize fast native training."""
import hashlib
import json
from pathlib import Path
import sys
import time

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from emerald import MAPS
from native_emerald import NativeEmerald
from train_earned_dreamer import write_json
root=project/"runs/pixels-only-01/official-wsl"
read=lambda p:json.loads(p.read_text(encoding="utf-8-sig"))
out=root/"controls/native-earned-route"

if __name__=="__main__":
    plan=read(root/"controls/reward-frontier-expansion/plan.json")
    record=plan["selected"]
    source=root/Path(plan["source"]).name/record["audit"]
    tape=(source/"game.bin").read_bytes()
    assert record["training_only"] and hashlib.sha256(tape).hexdigest()==record["tape_sha256"]
    actual=read(source/"game.json")
    measured=read(root/"controls/trainer-contacts/highest-return-review/measured.json")
    assert measured["input_sha256"]==record["tape_sha256"]
    cfg=read(project/"local.json")
    out.mkdir(exist_ok=True)
    game=NativeEmerald(cfg["native_core"],cfg["rom"])
    events={}
    maps={}
    started=time.monotonic()
    try:
        for i,b in enumerate(tape,1):
            _,key=game.step(b)
            valid=game.verified_story(key)
            if valid is not None:
                rival,party=valid
                if key[0]==5:
                    maps.setdefault(MAPS[key[1]][key[2]],i)
                for name,test in (("bedroom",key[5]>=4),("clock",bool(key[6])),
                        ("met_rival",rival),("starter",party)):
                    if test:
                        events.setdefault(name,i)
            if i%8192==0:
                write_json(out/"status.json",dict(buttons=i,expected=len(tape),native_maps=maps,native_events=events))
    finally:
        game.close()
    write_json(out/"result.json",dict(input_sha256=record["tape_sha256"],buttons=len(tape),source_only_self_earned=True,
        original_events=actual["events"],original_maps=measured["first_maps"],native_events=events,native_maps=maps,
        native_seconds=time.monotonic()-started,engine_sha256=hashlib.sha256(Path(cfg["native_core"]).read_bytes()).hexdigest(),
        confirms_original_gameplay=False))
