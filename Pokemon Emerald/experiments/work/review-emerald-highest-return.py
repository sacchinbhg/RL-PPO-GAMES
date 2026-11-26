"""Independently replay the highest scalar-return training game after the existing diagnostic finishes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
root=project/"runs/pixels-only-01/official-wsl"
out=root/"controls/trainer-contacts/highest-return-review"
out.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text())
while not (root/"earned-route-extension/trial-197461/handoff-recovery/verified.json").exists():
    time.sleep(30)
if (out/"measured.json").exists():
    sys.exit(0)
selection=out/"selection.json"
if selection.exists():
    chosen=read(selection)
else:
    candidates=[]
    for arm in [root/"reward-only-training-forks",*(root/"paired-self-earned-frontiers").glob("*")]:
        if not arm.is_dir() or not (arm/"best.json").is_file() or not (arm/"settings.json").is_file():
            continue
        best,settings=read(arm/"best.json"),read(arm/"settings.json")
        if best["training_only"] and settings["training_forks"] and not settings["receives_saves_from_other_runs"]:
            candidates.append((best["return_cents"],-best["length"],str(arm),best))
    _,_,arm,record=max(candidates,key=lambda row:row[:3])
    chosen=dict(arm=arm,record=record,selected_by="highest durable scalar return, then shortest tape")
    draft=selection.with_suffix(".tmp")
    draft.write_text(json.dumps(chosen,indent=2))
    draft.replace(selection)
record=chosen["record"]
audit=Path(chosen["arm"])/record["audit"]/"game.json"
assert hashlib.sha256(audit.with_suffix(".bin").read_bytes()).hexdigest()==record["tape_sha256"]
os.environ.update(EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so",EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),
                  OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
subprocess.run([sys.executable,str(Path(__file__).with_name("diagnose-emerald-rival-approach.py")),str(audit),str(out/"measured.json")],check=True)
