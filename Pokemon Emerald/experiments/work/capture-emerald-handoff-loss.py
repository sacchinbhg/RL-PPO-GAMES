"""Save the original-ROM footage of the first independently observed whiteout after the policy handoff."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
trial=project/"runs/pixels-only-01/official-wsl/earned-route-extension/trial-197461"
out=trial/"highlights/first-dreamer-whiteout"
out.mkdir(parents=True,exist_ok=True)
review=json.loads((trial/"handoff-recovery/dreamer.json").read_text())
audit,=(trial/"games/env-0/step-0065536").glob("*.json")
data=json.loads(audit.read_text())
tape=audit.with_suffix(".bin").read_bytes()
assert hashlib.sha256(tape).hexdigest()==review["sha256"]
loss=review["whiteouts"][0]
assert loss["battle_outcome"]==2 and loss["first_party_hp"]==0 and loss["hp_after"]>0
at=loss["decision"]
end=min(len(tape),at+10)
source=out/"from-boot.json"
source.with_suffix(".bin").write_bytes(tape[:end])
source.write_text(json.dumps(dict(steps=end,events={k:v for k,v in data["events"].items() if v<=end},
    maps_first_reached={k:v for k,v in data["maps_first_reached"].items() if v<=end},
    uncapped_exploration=data["uncapped_exploration"]),indent=2))
os.environ.update(EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so",EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),
                  OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
if not (out/"replay/verification.json").exists():
    with (out/"replay.log").open("a") as log:
        subprocess.run([sys.executable,str(project/"pixel_replay.py"),str(source),str(out/"replay"),
                        "--decision",str(at),"--video"],stdout=log,stderr=subprocess.STDOUT,check=True)
proof=json.loads((out/"replay/verification.json").read_text())
assert proof["reviewed_decision"]==at and proof["restores_during_game"]==0
assert proof["clip_through"]>=loss["first_overworld_after"]
(out/"verified.json").write_text(json.dumps(dict(source_audit=str(audit),source_sha256=review["sha256"],
    loss=loss,replay=proof),indent=2))
