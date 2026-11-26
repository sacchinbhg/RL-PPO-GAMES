"""Illustrate recorded endpoint trials after their actions have been executed."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from earned_starts import EarnedStarts
from emerald import Emerald
from pixel_env import game_paths
from reward_branch_search import visual_code,write_json

root=project/"runs/pixels-only-01/official-wsl"
arm=root/"paired-self-earned-frontiers/every-sealed-endpoint-01"
out=root/"controls/endpoint-button-sweep/observed-interactions"
read=lambda p:json.loads(p.read_text())

if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),
                      EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
    out.mkdir(parents=True,exist_ok=True)
    nearest=read(root/"controls/trainer-contacts/frontier-reviews/closest.json")
    proof=read(Path(nearest["verification"]))
    near=proof["input_sha256"]
    bank=EarnedStarts(root/"controls/dreamer-earned-resets/starts")
    game=Emerald(*game_paths())
    report=[]
    try:
        for path in sorted(arm.glob("discoveries/work-*/record.json")):
            row=read(path)
            if row["source_history"]!=near or row["first_button"] not in (0,1):
                continue
            anchor=row["source_prefix_length"]
            if anchor<proof["first_button_at_nearest"]-1 or row["length"]-anchor!=20:
                continue
            tape=path.with_name("game.bin").read_bytes()
            assert hashlib.sha256(tape).hexdigest()==row["tape_sha256"]
            prefix,index=max((s["prefix"],i) for i,s in enumerate(bank.records)
                             if s["prefix"]<=anchor and tape.startswith(bank.tape[:s["prefix"]]))
            _,score,_=bank.restore(game,index)
            total=bank.records[index]["scalar_return_cents"]
            for n,button in enumerate(tape[prefix:],prefix+1):
                image,state=game.step(button)
                rew,_=score.read(game,state)
                total+=round(100*rew)
                if n-anchor in (0,1,4,12,20):
                    Image.fromarray(game.screen).save(out/f"{anchor}-{row['first_button']}-{n-anchor}.png")
            assert visual_code(image).hex()==row["screen"] and total==row["return_cents"]
            report.append(dict(training_record=row["tape_sha256"],anchor=anchor,first_button=row["first_button"],
                               checked_final_image_and_return=True))
    finally:
        game.close()
    write_json(out/"verified.json",dict(reconstructed_training_tapes=report,images_for_human_review_only=True,
               prior_cold_replay_sha256=near,source="original ROM restored from verified self-earned training prefix"))
