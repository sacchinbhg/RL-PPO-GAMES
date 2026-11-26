"""Verify and collect self-generated alternatives before evaluating any learned action ranking."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from train_earned_dreamer import write_json
root=project/"runs/pixels-only-01/official-wsl"
out=root/"earned-button-alternatives"

def collect(dest,sources):
    args=[sys.executable,"-u",str(project/"collect_button_alternatives.py"),"--output",str(dest)]
    for path in sources:
        args+=["--dataset",str(path)]
    with (out/"collection.log").open("a") as log:
        subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)

def verify(dest):
    report=json.loads((dest/"completed.json").read_text())
    observed=0
    for game in report["games"]:
        with np.load(dest/game["training_tape"]/"windows.npz") as f:
            n=game["distinct_training_states"]
            assert f["context"].shape==(n,33,40,60,3) and f["past"].shape==(n,32)
            assert f["real_rewards"].shape==f["future"].shape==(n,9,15)
            assert np.all(f["future"][:,:,0]==np.arange(9)) and np.all(f["future"][:,:,1:]==f["future"][:,:1,1:])
            assert game["original_action_matched"]==n and np.isfinite(f["real_rewards"]).all()
            observed+=n
    assert observed==report["training_states"] and observed>0
    return report

if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so",
        OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
    out.mkdir(parents=True,exist_ok=True)
    with (out/"collection.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        early=next((root/"reward-only-training-forks/verified-training-tapes").glob("24479561ff0b*"))
        late=next((root/"earned-frontier-curriculum/verified-training-tapes").glob("01f5c6813216*"))
        small=next((root/"controls/outcome-collection-smoke/verified-training-tapes").glob("769d145e111d*"))
        write_json(out/"study.json",dict(datasets=[early.name,late.name],source_rule="the two self-earned training episodes used in the 1408-update fit",
            source_observations="raw RGB and scalar rewards",eval_for_training=False,
            evaluation="fixed previously held-out 39 windows, with sensitive-window performance and action-only comparison reported separately",
            hypotheses=["a reward forecast can recognize a positive scene while ranking its available actions incorrectly",
                        "actual action alternatives from self-play can improve the ranking"]))
        collect(out/"smoke",[small])
        smoke=verify(out/"smoke")
        write_json(out/"smoke-verified.json",dict(windows=smoke["training_states"],real_branch_buttons=smoke["additional_alternative_buttons"]))
        collect(out/"training",[early,late])
        checked=verify(out/"training")
        write_json(out/"verified.json",dict(windows=checked["training_states"],real_branch_buttons=checked["additional_alternative_buttons"],
            cold_replay_buttons=checked["cold_replay_buttons"],action_sensitive_training_states=sum(x["action_sensitive_states"] for x in checked["games"])))
