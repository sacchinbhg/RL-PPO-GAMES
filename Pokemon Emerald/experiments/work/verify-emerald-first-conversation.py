"""Verify the discovered dialogue by replaying the complete training input history from boot."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from emerald import ROM_SHA1
from train_earned_dreamer import write_json

root=project/"runs/pixels-only-01/official-wsl"
arm=root/"paired-self-earned-frontiers/every-sealed-endpoint-01"
report=root/"controls/endpoint-button-sweep/observed-interactions"
out=report/"original-from-boot"
read=lambda p:json.loads(p.read_text())

if __name__=="__main__":
    rows=read(report/"verified.json")["reconstructed_training_tapes"]
    row=max((x for x in rows if x["first_button"]==1),key=lambda x:x["anchor"])
    file=next(p for p in arm.glob("discoveries/work-*/record.json") if read(p)["tape_sha256"]==row["training_record"])
    buttons=read(file)["length"]
    assert hashlib.sha256(file.with_name("game.bin").read_bytes()).hexdigest()==row["training_record"]
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
    subprocess.run([sys.executable,str(project/"pixel_replay.py"),str(file.with_name("game.json")),str(out),
                    "--decision",str(buttons),"--video"],check=True)
    check=read(out/"verification.json")
    before=Image.open(report/f"{row['anchor']}-1-20.png")
    after=Image.open(out/f"decision-{buttons}.png")
    assert np.array_equal(np.asarray(before),np.asarray(after))
    assert check["input_sha256"]==row["training_record"] and check["restores_during_game"]==0 and check["rom_sha1"]==ROM_SHA1
    write_json(out/"matched-dialogue.json",dict(rom_sha1=ROM_SHA1,source_tape=check["input_sha256"],
        buttons=buttons,in_game_restores=0,matched_native_rgb=True,source_kind="autonomous training branch"))
