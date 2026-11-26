"""Compare official online Dreamer with two earned-start workers against four fresh-start workers."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from pixel_env import PixelPool
from train_earned_dreamer import digest,write_json

root=project/"runs/pixels-only-01/official-wsl"
out=root/"controls/dreamer-earned-resets"
family=root/"earned-reset-online-comparison"
initial=root/"seed-71299-h15/eval-65536/frozen-policy"
read=lambda p:json.loads(p.read_text())


def memory():
    return int(next(s.split()[1] for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:")))


def call(directory,script,*args):
    with (directory/"experiment.log").open("a") as log:
        subprocess.run([sys.executable,"-u",str(project/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True)


def smoke():
    info=read(out/"starts/ready.json")
    full=(out/"starts/source.bin").read_bytes()
    observations=[]
    for offset in (0,4):
        pool=PixelPool(1,True,earned_starts=out/"starts",start_offset=offset)
        try:
            for i in range(2):
                frame=pool.command(0,"reset")
                saved=info["starts"][offset+i]
                with np.load(out/"starts"/saved["filename"],allow_pickle=False) as data:
                    assert np.array_equal(frame,data["image"])
                start=saved["prefix"]
                for button in full[start:start+32]:
                    frame,reward,done=pool.command(0,"act",button)
                    assert not done
                dest=out/"training-worker-test"/f"offset-{offset}-reset-{i}"
                pool.command(0,"audit",str(dest))
                audit,=dest.glob("*.json")
                a=read(audit)
                assert a["training_only"] and a["earned_start_prefix"]==start and a["steps_since_training_reset"]==32
                assert audit.with_suffix(".bin").read_bytes()==full[:start+32]
                observations.append(dict(prefix=start,successor_buttons=a["steps_since_training_reset"],total_replay_buttons=a["steps"]))
        finally:
            pool.close()
    write_json(out/"worker-verified.json",dict(starts=observations,bank_successor_checks=info["real_successors_compared"]))


def train(name,earned):
    directory=family/name
    directory.mkdir(parents=True,exist_ok=True)
    marker=directory/"training-result.json"
    if marker.exists():
        return read(marker)
    train=directory/"train"
    with (root/"controls/large-optimizer.lock").open("a") as slot:
        while True:
            if memory()>=2800000:
                fcntl.flock(slot,fcntl.LOCK_EX)
                if memory()>=2800000:
                    break
                fcntl.flock(slot,fcntl.LOCK_UN)
            time.sleep(30)
        args=["--upstream","/opt/emerald-rl/source","--logdir",train,"--initial-agent",initial,
              "--platform","cuda","--envs",4,"--parallel-envs","--episode",32768,"--short-episode-envs",2,
              "--short-episode-length",2048,"--batch",8,"--length",32,"--ratio",16,"--imag-length",15,
              "--uncapped-exploration","--steps",32784,"--seed",71299]
        if earned:
            args+=["--earned-start-envs",2,"--earned-starts",out/"starts"]
        call(directory,"upstream_emerald.py",*args)
    checkpoint=train/"ckpt"/(train/"ckpt/latest").read_text().strip()
    with (initial/"agent.pkl").open("rb") as f:
        before=pickle.load(f)["params"]
    with (checkpoint/"agent.pkl").open("rb") as f:
        after=pickle.load(f)["params"]
    changed=sorted({k.split("/")[0] for k in before if not np.array_equal(before[k],after[k])})
    assert {"dyn","pol","val"}.issubset(changed)
    transitions=resets=0
    for file in (train/"replay").glob("*.npz"):
        with np.load(file,allow_pickle=False) as chunk:
            first=chunk["is_first"]
            resets+=int(first.sum())
            transitions+=int((~first).sum())
    assert transitions>32000
    result=dict(checkpoint=str(checkpoint),sha256=digest(checkpoint/"agent.pkl"),initial_sha256=digest(initial/"agent.pkl"),
                changed_module_groups=changed,real_emulator_actions_in_replay=transitions,reset_observations=resets)
    write_json(marker,result)
    return result


def run():
    family.mkdir(exist_ok=True)
    write_json(out/"online-registration.json",dict(initial_weights=digest(initial/"agent.pkl"),online_worker_count=4,
        treatment="the last two workers reset to one of eight self-earned original-game states",
        control="all four workers reset to boot",short_episode_workers=[2,3],short_episode_length=2048,
        main_episode_length=32768,training_seed=71299,evaluation_seed=271348,evaluation_steps_per_game=65536,
        training_log_steps_requested=32784,frames_per_decision=24,observation="RGB, scalar reward and episode boundary",
        reward="identical uncapped first-visit and story rewards; saved workers retain their earned reward history",
        initial_replay="empty in both arms",collector="the official Dreamer actor on every worker",
        evaluation="four fresh games per model with zero in-game restores and no further updates"))
    while not (out/"starts/ready.json").exists():
        time.sleep(30)
    if not (out/"worker-verified.json").exists():
        smoke()
    for name,earned in (("earned",True),("boot",False)):
        train(name,earned)
    games={}
    with (root/"controls/followup-evaluation.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for name in ("earned","boot"):
            while memory()<1900000:
                time.sleep(30)
            directory=family/name
            weights=read(directory/"training-result.json")
            dest=directory/"fresh-001"
            if not (dest/"result.json").exists():
                if (dest/"provenance.json").exists():
                    dest=directory/f"fresh-{time.time_ns()}"
                call(directory,"play_frozen_dreamer.py",weights["checkpoint"],directory/"train",dest,"--upstream",
                     "/opt/emerald-rl/source","--limit",65536,"--seed",271348,"--log-whiteouts")
            result=read(dest/"result.json")
            assert result["weight_sha256"]==weights["sha256"]
            games[name]=[]
            for lane in range(4):
                audits=sorted((dest/f"games/env-{lane}").glob("step-*/*.json"))
                a=read(audits[-1])
                assert not a.get("training_only",False)
                games[name].append(dict(steps=a["steps"],events=a["events"],maps=a["maps_first_reached"],
                                       tape_sha256=hashlib.sha256(audits[-1].with_suffix(".bin").read_bytes()).hexdigest()))
    write_json(out/"comparison.json",dict(games=games,training={n:read(family/n/"training-result.json") for n in games}))


if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so",
                      OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",JAX_PLATFORMS="cuda",XLA_PYTHON_CLIENT_PREALLOCATE="false",
                      EMERALD_REPLAY_MMAP_DIR="/opt/emerald-rl/replay-cache")
    out.mkdir(parents=True,exist_ok=True)
    with (out/"online.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (out/"comparison.json").exists():
            run()
