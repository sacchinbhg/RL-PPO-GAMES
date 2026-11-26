"""Try every real button from the same held-out stuck frame. Results are diagnostic only."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from emerald import BUTTONS,Emerald,REPEAT,ROM_SHA1
from pixel_env import game_paths
from train_earned_dreamer import write_json

root=project/"runs/pixels-only-01/official-wsl"
control=root/"controls/retrieval-loops"
out=control/"heldout-button-effects"


def run():
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),
                      EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"registration.json",dict(position=50000,arms=list(BUTTONS),initial_choice=1,following_waits=3,
        comparison="fraction of policy pixels differing from the waiting counterfactual by at least 24 in any channel",
        persistent="at least 2% of pixels differ at both the third and fourth decision",
        purpose="diagnosis of held-out policy behavior only; exclude these trials from any fitter"))
    while not (control/"select-footage/verification.json").exists():
        time.sleep(30)
    proof=json.loads((control/"select-footage/verification.json").read_text())
    assert proof["rom_sha1"]==ROM_SHA1 and proof["reviewed_decision"]==50000
    audit,=(root/"earned-screen-memory/clean/step-0065536").glob("*.bin")
    tape=audit.read_bytes()
    assert hashlib.sha256(tape).hexdigest()==proof["input_sha256"]
    game=Emerald(*game_paths())
    try:
        game.load(game.boot)
        for n,b in enumerate(tape[:50000],1):
            game.step(b)
            if n%8192==0:
                write_json(out/"status.json",dict(original_buttons=n,expected=50000))
        start=game.screen.copy()
        historic=np.array(Image.open(control/"select-footage/decision-50000.png"))
        if not np.array_equal(start,historic):
            write_json(out/"different-boot-image.json",dict(matched_full_rgb=False,
                       differing_pixels=int(np.any(start!=historic,axis=2).sum())))
            raise RuntimeError("The diagnostic cold boot did not reproduce the held-out image exactly")
        state=game.save()
        rows={}
        frames={}
        for button,name in enumerate(BUTTONS):
            game.load(state)
            samples=[]
            for action in (button,0,0,0):
                rgb,_=game.step(action)
                samples.append(rgb.copy())
            frames[name]=np.stack(samples).astype(np.int16)
            Image.fromarray(game.screen).save(out/(name+".png"))
        for name,images in frames.items():
            diff=np.abs(images-frames["wait"]).max(3)
            fractions=(diff>=24).mean((1,2)).tolist()
            rows[name]=dict(fraction_changed=fractions,persistent=all(v>=.02 for v in fractions[-2:]))
        write_json(out/"result.json",dict(rom_sha1=ROM_SHA1,full_rgb_start_matched=True,
                   original_buttons=50000,independent_alternative_decisions=36,frames_per_decision=REPEAT,
                   outcomes=rows,used_for_training=False))
    finally:
        game.close()


if __name__=="__main__":
    run()
