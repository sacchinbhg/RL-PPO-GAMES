"""Test rewarding verified combat wins on the same earned training games and official parent."""
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
from battle_reward_events import replay
from train_earned_dreamer import Trajectory,write_json
root=project/"runs/pixels-only-01/official-wsl"
older=root/"outcome-cost-comparison"
out=root/"battle-win-comparison"
parent=root/"seed-71299-h15/eval-65536"
read=lambda p:json.loads(p.read_text())


def call(script,*args):
    with (out/"experiment.log").open("a") as log:
        subprocess.run([sys.executable,"-u",str(project/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True)


def memory():
    return int(next(s.split()[1] for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:")))


def run():
    control=read(older/"control/fit/result.json")
    tapes=control["training_tapes"]
    updates=control["optimizer_updates"]
    assert len(tapes)==2 and updates==1408
    write_json(out/"registration.json",dict(training_tapes=tapes,update_count=updates,control=control["trained_sha256"],
        reward="add 0.25 once per game-reported win to its incoming reward; each game keeps its existing visit cap",
        policy_input="RGB and raw scalar reward during training; frozen evaluation receives the ordinary observations",
        dataset_selection="same two archived, self-earned histories previously used in the survival comparison",
        primary="fresh starts with a starter, lab handoff, later areas, and original-ROM battle victories at 65536 buttons",
        seed=271343,hardware_frames_per_decision=24))
    logs=out/"training-battles"
    data=[older/"verified-training-tapes"/sha for sha in tapes]
    baseline=read(root/"controls/trainer-contacts/route103.json")
    known_wins=[battle["first_nonzero_outcome"][0] for battle in baseline["recorded_battles"]
                if battle["first_nonzero_outcome"] and battle["first_nonzero_outcome"][1]==1]
    assert len(known_wins)==4
    checks=[]
    for sha,dataset in zip(tapes,data):
        file=logs/(sha+".json")
        write_json(out/"status.json",dict(verifying_training_game=sha,time=time.time()))
        if not file.exists():
            replay(dataset,logs)
        record=read(file)
        ordinary=Trajectory(dataset)
        treatment=Trajectory(dataset,file,25)
        assert list(treatment.tape[:baseline["buttons"]])==list((next((root/"paired-self-earned-frontiers/visual-slots-0/verified-training-tapes").glob("e308*/audit/*.bin"))).read_bytes())
        assert [i for i in record["win_decisions"] if i<=baseline["buttons"]]==known_wins
        changes=[]
        for n,(original,shaped) in enumerate(zip(ordinary.episode(),treatment.episode(),strict=True)):
            assert np.array_equal(original[0],shaped[0]) and original[1]==shaped[1] and original[3:]==shaped[3:]
            delta=round(100*(shaped[2]-original[2]))
            if delta:
                assert delta==25
                changes.append(n)
        assert changes==record["win_decisions"]
        checks.append(dict(game=sha,win_decisions=changes,identical_images_and_actions=len(treatment.tape)+1))
    write_json(out/"input-checks.json",dict(games=checks,known_prefix_reproduced_wins=known_wins))
    fit=out/"win-reward/fit"
    while not (fit/"result.json").exists():
        with (root/"controls/large-optimizer.lock").open("a") as gate:
            if memory()>=3000000:
                try:
                    fcntl.flock(gate,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    if memory()>=3000000:
                        write_json(out/"status.json",dict(training=True,time=time.time()))
                        call("train_earned_dreamer.py","--dataset",data[0],"--dataset",data[1],"--source-run",parent,
                            "--output",fit,"--upstream","/opt/emerald-rl/source","--updates",updates,
                            "--battle-win-logs",logs,"--battle-win-cents",25)
                        continue
        time.sleep(30)
    model=read(fit/"result.json")
    for key in ("parent_sha256","upstream_revision","training_tapes","optimizer_updates","imagination_length","observations_trained"):
        assert control[key]==model[key]
    while not (older/"result.json").exists() or memory()<2000000:
        time.sleep(30)
    # Serialize additional four-game evaluations once the original survival comparison frees the GPU slot.
    with (root/"controls/followup-evaluation.lock").open("a") as slot:
        fcntl.flock(slot,fcntl.LOCK_EX)
        trials=sorted((out/"win-reward").glob("fresh-*"))
        trial=next((p for p in trials if (p/"result.json").exists()),None)
        if trial is None:
            trial=out/"win-reward"/f"fresh-{len(trials)+1:03d}"
            call("play_frozen_dreamer.py",fit/"frozen-policy",parent,trial,"--upstream","/opt/emerald-rl/source",
                 "--seed",271343,"--limit",65536,"--log-whiteouts")
    game=read(trial/"result.json")
    assert game["weight_sha256"]==model["trained_sha256"] and game["seed"]==271343 and game["in_game_restores"]==0
    evidence={}
    for name,directory in (("control",next(p for p in sorted((older/"control").glob("fresh-*")) if (p/"result.json").exists())),("win-reward",trial)):
        evidence[name]=[]
        for lane in range(4):
            audit=max((directory/"games"/f"env-{lane}").glob("step-*/*episode-00001.json"),key=lambda p:read(p)["steps"])
            record=read(audit)
            detail=out/"evaluation-audit"/name/f"env-{lane}.json"
            if "starter" in record["events"] and not detail.is_file():
                call_absolute=base/"work/diagnose-emerald-battles.py"
                with (out/"experiment.log").open("a") as log:
                    subprocess.run([sys.executable,str(call_absolute),str(audit),str(detail)],check=True,stdout=log,stderr=subprocess.STDOUT)
            battles=read(detail)["recorded_battles"] if detail.is_file() else []
            won=sum(bool(b["first_nonzero_outcome"] and b["first_nonzero_outcome"][1]==1) for b in battles)
            evidence[name].append(dict(buttons=record["steps"],events=record["events"],maps=record["maps_first_reached"],victories=won))
    write_json(out/"result.json",dict(training=model["trained_sha256"],control=control["trained_sha256"],games=evidence))


if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so",
                      OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",JAX_PLATFORMS="cuda",XLA_PYTHON_CLIENT_PREALLOCATE="false")
    out.mkdir(exist_ok=True)
    with (out/"job.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
