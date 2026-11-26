"""Compare official online Dreamer with and without battle feedback on self-earned outcome starts."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from earned_starts import EarnedStarts
from pixel_env import PixelPool
from play_frozen_dreamer import load_actor
from train_earned_dreamer import digest,write_json

root=project/"runs/pixels-only-01/official-wsl"
out=root/"outcome-curriculum"
bankdir=out/"starts"
upstream=Path("/opt/emerald-rl/source")
read=lambda path:json.loads(path.read_text())


def call(directory,script,*args):
    with (directory/"experiment.log").open("a") as log:
        subprocess.run([sys.executable,"-u",str(script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True)


def probe(train,ckpt,directory):
    count=len(EarnedStarts(bankdir).records)
    # Both policies encounter the same starting pictures and scalar feedback.
    actor=load_actor(ckpt,train,271351,upstream)
    rows=[]
    for start in range(0,count,4):
        pool=PixelPool(4,True,earned_starts=bankdir,start_offset=start,battle_feedback=True)
        try:
            images=np.stack([pool.command(i,"reset") for i in range(4)])
            rewards=np.zeros(4,np.float32)
            for t in range(768):
                actions=actor(images,rewards,t==0)
                images,rewards,done=pool.step(actions)
                if done.any():
                    break
            for lane in range(min(4,count-start)):
                dest=directory/f"position-{start+lane:02d}"
                pool.command(lane,"audit",str(dest))
                report=read(next(dest.glob("*.json")))
                rows.append(dict(bank_index=start+lane,source_prefix=report["earned_start_prefix"],
                                 decisions=report["steps_since_training_reset"],outcomes=report["battle_feedback"],
                                 progression=report["events"]))
        finally:
            pool.close()
    write_json(directory/"result.json",dict(weight_sha256=digest(ckpt/"agent.pkl"),zero_weight_updates=True,
               training_positions=True,rows=rows))


def run():
    out.mkdir(exist_ok=True)
    initial=Path(read(root/"earned-reset-online-comparison/earned/training-result.json")["checkpoint"])
    write_json(out/"training-plan.json",dict(initial_weight_sha256=digest(initial/"agent.pkl"),
        seed=71299,training_envs=4,boot_workers=1,earned_workers=3,short_episode_decisions=1024,
        training_budget_each=65536,frames_per_decision=24,policy_inputs="RGB, scalar reward and episode flags",
        treatment="ordinary rewards plus 0.5 times own fractional HP increase minus opponent fractional HP increase, plus one for a win or minus one for a loss",
        control="identical starts, original reward",learner="official DreamerV3 size1m, 15 imagined transitions",
        evaluation="768 decisions on each self-earned position; separate fresh-game checks"))
    while not (bankdir/"ready.json").exists() or not (out/"feedback-check.json").exists():
        time.sleep(10)
    for name,shaped in (("combat",True),("ordinary",False)):
        directory=out/name
        directory.mkdir(exist_ok=True)
        train=directory/"train"
        with (root/"controls/large-optimizer.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            while int(next(s.split()[1] for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:")))<2400000:
                time.sleep(15)
            if not (train/"final-checkpoint.json").exists():
                args=["--upstream",upstream,"--logdir",train,"--initial-agent",initial,"--platform","cuda","--parallel-envs",
                      "--envs",4,"--episode",32768,"--short-episode-envs",3,"--short-episode-length",1024,
                      "--earned-start-envs",3,"--earned-starts",bankdir,"--batch",8,"--length",32,"--ratio",16,
                      "--imag-length",15,"--uncapped-exploration","--steps",65536,"--seed",71299]
                if shaped:
                    args.append("--battle-feedback")
                call(directory,project/"upstream_emerald.py",*args)
        ckpt=train/"ckpt"/(train/"ckpt/latest").read_text().strip()
        assert read(train/"final-checkpoint.json")["training_step"]>=65536
        assert digest(ckpt/"agent.pkl")!=digest(initial/"agent.pkl")
        write_json(directory/"trained.json",dict(checkpoint=str(ckpt),sha256=digest(ckpt/"agent.pkl"),initial_sha256=digest(initial/"agent.pkl")))
        measurement=directory/"practice-probe"
        measurement.mkdir(exist_ok=True)
        if not (measurement/"result.json").exists():
            call(measurement,Path(__file__),"--probe",train,ckpt,measurement)


if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so",
                      OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",JAX_PLATFORMS="cuda",XLA_PYTHON_CLIENT_PREALLOCATE="false",
                      EMERALD_REPLAY_MMAP_DIR="/opt/emerald-rl/replay-cache")
    if len(sys.argv)>1 and sys.argv[1]=="--probe":
        probe(*map(Path,sys.argv[2:]))
    else:
        with (out/"learner.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            run()
