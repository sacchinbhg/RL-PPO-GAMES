"""Make an editorial frame for the closest observed approach, after the training tape is sealed."""
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
from train_earned_dreamer import write_json

root=project/"runs/pixels-only-01/official-wsl"
review=root/"controls/trainer-contacts/highest-return-review"
read=lambda p:json.loads(p.read_text())
measured=read(review/"measured.json")
if measured["nearest_template_manhattan_tiles"] is None:
    raise SystemExit("This training game never reached the rival's area")
record=read(review/"selection.json")
audit=Path(record["arm"])/record["record"]["audit"]/"game.json"
tape=audit.with_suffix(".bin").read_bytes()
assert hashlib.sha256(tape).hexdigest()==measured["input_sha256"]
at=measured["first_button_at_nearest"]-1
dest=review/"highlights"
dest.mkdir(exist_ok=True)
dataset=next((root/"earned-frontier-curriculum/verified-training-tapes").glob("01f5c6813216*"))
stored,=(dataset/"audit").glob("*.bin")
match=stored.read_bytes()[:at]==tape[:at]
if match:
    size=read(dataset/"manifest.json")["chunk_length"]
    offset=(at-1)//size*size
    with np.load(dataset/f"chunk-{offset:09d}.npz") as chunk:
        im=Image.fromarray(chunk["image"][at-offset])
    im.resize((480,320),Image.Resampling.NEAREST).save(dest/"approach.png")
    write_json(dest/"frame.json",dict(decision_completed=at,distance_before_next_button=measured["nearest_template_manhattan_tiles"],
        verified_prefix_sha256=hashlib.sha256(tape[:at]).hexdigest(),source="original 60 x 40 policy image, enlarged 8x"))
os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
end=min(len(tape),at+12)
selected=dict(steps=end,uncapped_exploration=True,events={k:v for k,v in measured["events"].items() if v<=end},
    maps_first_reached={k:v for k,v in measured["first_maps"].items() if v<=end})
prefix=dest/"approach-from-boot.json"
write_json(prefix,selected)
prefix.with_suffix(".bin").write_bytes(tape[:end])
if not (dest/"replay/verification.json").exists():
    subprocess.run([sys.executable,str(project/"pixel_replay.py"),str(prefix),str(dest/"replay"),"--video","--decision",str(at)],check=True)
