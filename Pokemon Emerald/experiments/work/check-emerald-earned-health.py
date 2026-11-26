"""Human audit of remaining hit points in the explorer's previously earned training positions."""
import os
from pathlib import Path
import sys
base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from earned_starts import EarnedStarts
from emerald import Emerald
from pixel_env import game_paths
from train_earned_dreamer import write_json

if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
    root=project/"runs/pixels-only-01/official-wsl"
    bank=EarnedStarts(root/"controls/dreamer-earned-resets/starts")
    game=Emerald(*game_paths())
    try:
        rows=[]
        for i,r in enumerate(bank.records):
            bank.restore(game,i)
            rows.append(dict(decisions=r["prefix"],level=game.ram(0x020244ec+84,1)[0],hp=int.from_bytes(game.ram(0x020244ec+86,2),"little"),
                             maximum=int.from_bytes(game.ram(0x020244ec+88,2),"little")))
        write_json(root/"controls/endpoint-button-sweep/first-encounter-followup/earned-health.json",dict(source=bank.info["source_sha256"],observations=rows,actor_receives_this=False))
    finally:
        game.close()
