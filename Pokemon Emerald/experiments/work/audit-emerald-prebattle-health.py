"""Find where a verified training history lost hit points before its first recorded trainer battle."""
import json
import os
from pathlib import Path
import sys
base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from emerald import Emerald
from pixel_env import game_paths
from earned_starts import EarnedStarts
from train_earned_dreamer import write_json
root=project/"runs/pixels-only-01/official-wsl"
out=root/"controls/endpoint-button-sweep/first-encounter-followup"
read=lambda p:json.loads(p.read_text())

if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
    mark=read(out/"measured.json")
    tape=Path(mark["source"]).with_suffix(".bin").read_bytes()
    cut=mark["first_trainer_battle"]-1
    bank=EarnedStarts(root/"controls/dreamer-earned-resets/starts")
    usable=[(s["prefix"],i) for i,s in enumerate(bank.records) if s["prefix"]<cut and tape.startswith(bank.tape[:s["prefix"]])]
    begin,index=usable[-2]
    game=Emerald(*game_paths())
    try:
        bank.restore(game,index)
        events=[]
        def health():
            return [int.from_bytes(game.ram(0x020244ec+off,2),"little") for off in (86,88)]
        prior=health()
        start=prior
        for n,b in enumerate(tape[begin:cut],begin+1):
            _,key=game.step(b)
            hp=health()
            if hp!=prior:
                events.append(dict(step=n,previous=prior,current=hp,frames_to_trainer=(cut+1-n)*24,battle_callback=key[0]==6))
                prior=hp
        assert prior==[read(out/"battle-health.json")["entry_party"][k] for k in ("hp","maximum")]
        write_json(out/"approach-health.json",dict(from_verified_prefix=begin,to_decision=cut,initial=start,final=prior,
                   changes=events,source_sha256=mark["source_sha256"],human_review_only=True))
    finally:
        game.close()
