"""Compare matched three-button options with and without the held-out-tested image-motion predictor."""
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
from fit_local_motion import ScrollModel,hashes,load
from train_earned_dreamer import write_json
root=project/"runs/pixels-only-01/official-wsl"
family=root/"paired-self-earned-frontiers"
control=root/"controls/local-screen-motion"
out=control/"live-comparison"
parent=root/"controls/reward-frontier-expansion"
probe=root/"controls/visual-motion/route-training.json"
read=lambda p:json.loads(p.read_text())


def cmd(source,record,dest,steps,capacity,treatment):
    args=[sys.executable,"-u",str(project/"explore_reward_frontiers.py"),"--source",str(source),"--record",str(record),
          "--motion",str(probe),"--output",str(dest),"--steps",str(steps),"--capacity",str(capacity),"--seed","173746",
          "--reward-sweep","--movement-triples"]
    if treatment:
        args+=["--scroll-model",str(control/"fitted")]
    return args


def run():
    sealed=read(control/"decision.json")
    assert sealed["qualifies_for_live_comparison"]
    model=ScrollModel(control/"fitted")
    first,_,_=load(control/"train")
    held,truth,_=load(control/"test")
    mask=~np.isin(hashes(held),hashes(first)) & (truth.min(1)!=truth.max(1))
    estimates=np.stack([model.predictions(frame) for frame in held[mask]])
    tied=np.isclose(estimates,estimates.max(1,keepdims=True),atol=1e-6,rtol=0)
    actual=float(((truth[mask]*tied).sum(1)/tied.sum(1)).mean())
    assert abs(actual-sealed["measured"]["results"]["local_screen"]["realized_scroll"])<1e-5
    plan=read(parent/"plan.json")
    write_json(out/"registration.json",dict(seed=173746,new_buttons_per_arm=131072,source=plan["selected"]["tape_sha256"],
        checkpoint=model.sha,heldout_sensitive_positions=int(mask.sum()),live_method_reproduced_heldout=actual,
        matched="same training tape, 768 reward slots, exhaustive single buttons, symmetric three-decision movement options",
        change="only the treatment ranks candidate movements by predicted scroll when a three-decision movement option is eligible",
        outcome="new areas, game progression, and coarse-cell coverage at matched emulator budgets",
        policy_inputs="RGB and scalar reward; no map or trainer labels"))
    smoke=root/"controls/outcome-collection-smoke"
    for name,mode in (("ordinary",False),("learned",True)):
        dest=out/("smoke-"+name)
        with (out/"smoke.log").open("a") as log:
            subprocess.run(cmd(smoke,smoke/"best.json",dest,2048,16,mode),stdout=log,stderr=subprocess.STDOUT,check=True)
        result=read(dest/"completed.json")
        assert result["status"]["new_buttons"]==2048 and result["status"]["movement_options"]["three_button_choices"]>0
        if mode:
            assert result["status"]["movement_options"]["changed_first_button"]>0
            verified=out/"smoke-cold"
            with (out/"smoke.log").open("a") as log:
                subprocess.run([sys.executable,str(project/"pixel_replay.py"),str(dest/result["best"]["audit"]/"game.json"),
                                str(verified),"--decision",str(result["best"]["length"])],stdout=log,stderr=subprocess.STDOUT,check=True)
            proof=read(verified/"verification.json")
            assert proof["input_sha256"]==result["best"]["tape_sha256"] and proof["restores_during_game"]==0
    children={}
    for name,mode in (("learned",True),("ordinary",False)):
        while int(next(s.split()[1] for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:")))<1800000:
            time.sleep(30)
        dest=family/("sweep-movement-"+name)
        with (out/(name+".log")).open("a") as log:
            children[name]=subprocess.Popen(cmd(plan["source"],parent/"selected-record.json",dest,131072,768,mode),
                stdout=log,stderr=subprocess.STDOUT)
        if name=="learned":
            while not (dest/"initialization.json").exists() and children[name].poll() is None:
                time.sleep(10)
    for name,child in children.items():
        if child.wait()!=0:
            raise RuntimeError(f"Movement comparison failed in {name}")
    write_json(out/"result.json",dict(arms={k:read(family/("sweep-movement-"+k)/"completed.json") for k in children}))


if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so",
                      OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
    out.mkdir(parents=True,exist_ok=True)
    with (out/"job.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
