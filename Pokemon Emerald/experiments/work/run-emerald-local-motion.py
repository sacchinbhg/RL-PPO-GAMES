"""Collect training and held-out button experiments after excluding their shared opening."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from train_earned_dreamer import Trajectory,write_json
root=project/"runs/pixels-only-01/official-wsl"
out=root/"controls/local-screen-motion"
probe=root/"controls/visual-motion/route-training.json"

def run():
    train=next((root/"paired-self-earned-frontiers/visual-slots-0/verified-training-tapes").glob("e30822*"))
    other=next((root/"earned-frontier-curriculum/verified-training-tapes").glob("01f5c6*"))
    a,b=Trajectory(train),Trajectory(other)
    common=next((i for i,(x,y) in enumerate(zip(a.tape,b.tape)) if x!=y),min(len(a.tape),len(b.tape)))
    assert json.loads(probe.read_text())["training_tape_sha256"]==a.source["source_sha256"]
    write_json(out/"registration.json",dict(train=a.source["source_sha256"],test=b.source["source_sha256"],shared_prefix=common,
        fixed_stride=160,three_repeated_actions=True,controls="training button success rates and nearest whole training images",
        fit="for each action select a 12x12 observation patch using blocked training-only validation; four nearest training patches predict scroll",
        evaluation="exclude shared opening and exact training images; realized motion on positions with different outcomes",
        deployment="compare learned image patches against both controls before changing a real explorer",coordinates_used=False))
    for name,src,skip in (("train",train,0),("test",other,common)):
        with (out/"collection.log").open("a") as log:
            subprocess.run([sys.executable,"-u",str(project/"collect_visual_motion_forks.py"),"--dataset",str(src),"--motion",str(probe),
                           "--output",str(out/name),"--stride","160","--skip-prefix",str(skip)],stdout=log,stderr=subprocess.STDOUT,check=True)

if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so",
                      OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
    out.mkdir(exist_ok=True)
    with (out/"job.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
