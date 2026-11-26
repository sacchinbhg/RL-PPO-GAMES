"""Record the next cold-verified Route 103 approach on the ordinary emulator for the video diary."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from train_earned_dreamer import digest,write_json
root=project/"runs/pixels-only-01/official-wsl"
review=root/"controls/trainer-contacts/frontier-reviews"
read=lambda p:json.loads(p.read_text())

def capture(marker):
    verified=read(Path(marker["verification"]))
    audit=Path(marker["audit"])
    assert verified["in_game_restores"]==0 and digest(audit.with_suffix(".bin"))==verified["input_sha256"]
    step=verified["first_button_at_nearest"]-1
    output=Path(marker["verification"]).parent/"footage"
    output.mkdir(exist_ok=True)
    if (output/"verification.json").exists():
        return
    end=min(verified["buttons"],step+12)
    prefix=output/"source.json"
    prefix.with_suffix(".bin").write_bytes(audit.with_suffix(".bin").read_bytes()[:end])
    write_json(prefix,dict(steps=end,events={k:n for k,n in verified["events"].items() if n<=end},
               maps_first_reached={k:n for k,n in verified["first_maps"].items() if n<=end},uncapped_exploration=True))
    subprocess.run([sys.executable,str(project/"pixel_replay.py"),str(prefix),str(output),"--video","--decision",str(step)],check=True)

if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so",
        OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    last=5
    while last>1:
        if (review/"closest.json").exists():
            best=read(review/"closest.json")
            if best["distance"]<last:
                capture(best)
                last=best["distance"]
        time.sleep(30)
