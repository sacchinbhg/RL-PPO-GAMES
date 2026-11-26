"""Compare two official Dreamer fits on identical self-play samples, with one scalar reward difference."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from train_earned_dreamer import Trajectory,write_json

root=project/"runs/pixels-only-01/official-wsl"
family=root/"paired-self-earned-frontiers"
out=root/"outcome-cost-comparison"
parent=root/"seed-71299-h15/eval-65536"
source=next((family/"visual-slots-0/verified-training-tapes").glob("e30822c*"))
upstream="/opt/emerald-rl/source"

def read(p):
    return json.loads(p.read_text())

def available():
    return int(next(s.split()[1] for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:")))

def wait(reason):
    write_json(out/"status.json",dict(time=time.time(),waiting_for=reason,available_kib=available()))
    time.sleep(45)

def call(script,*args):
    with (out/"experiment.log").open("a") as f:
        subprocess.run([sys.executable,"-u",str(project/script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True)

def outcomes(cohort):
    rows=[]
    for record in read(cohort/"completed.json")["records"]:
        audit=read(cohort/record["audit"]/"game.json")
        losses=sum(entry["decision"]>36058 for entry in audit["whiteouts"])
        rows.append(dict(cohort=str(cohort),record=record,branch_whiteouts=losses))
    return rows

def check_stream(data,cost):
    game=Trajectory(data)
    decisions={e["decision"] for e in game.source["verified_whiteouts"]}
    changed=[]
    for at,(left,right) in enumerate(zip(game.episode(False),game.episode(False,cost),strict=True)):
        if not np.array_equal(left[0],right[0]) or left[1]!=right[1] or left[3:]!=right[3:]:
            raise ValueError("A non-reward learner input changed")
        delta=round(100*(left[2]-right[2]))
        if delta!=(cost if at in decisions else 0):
            raise ValueError("Whiteout cost did not align with its successor observation")
        if delta:
            changed.append(at)
    return dict(recorded_buttons=len(game.tape),changed_observations=changed,all_other_inputs_match=True)

def run():
    plan=dict(initial_branches=8,buttons_per_initial_branch=8192,additional_block_if_zero_losses=[8,16384],
              selected_training_games="two distinct tapes with the fewest and most suffix whiteouts; earliest indices break ties",
              reward_cost_cents=[0,50],frozen_evaluation_seed=271343,buttons_per_evaluation_game=65536,
              games_per_arm=4,learner="pinned official DreamerV3",frozen_evaluation_tapes_in_training=False)
    if (out/"plan.json").exists() and read(out/"plan.json")!=plan:
        raise ValueError("A different experiment already occupies this directory")
    write_json(out/"plan.json",plan)
    first=family/"all-outcome-attempts"
    while not (first/"completed.json").exists():
        wait("initial eight self-generated attempts")
    choices=outcomes(first)
    if not any(r["branch_whiteouts"] for r in choices):
        follow=family/"all-outcome-attempts-longer"
        while available()<1400000:
            wait("memory for longer self-play")
        call("collect_outcome_branches.py","--dataset",source,"--output",follow,"--seed",260696,
             "--branches",8,"--steps-per-branch",16384,"--inherited-search-buttons",321586)
        choices=outcomes(follow)
    ordered=sorted(choices,key=lambda r:(r["branch_whiteouts"],r["record"]["index"]))
    if not ordered[-1]["branch_whiteouts"]:
        write_json(out/"result.json",dict(outcome="no whiteouts in either fixed self-play block; the reward terms are identical"))
        return
    fewest=ordered[0]
    others=[r for r in ordered if r["record"]["tape_sha256"]!=fewest["record"]["tape_sha256"]]
    most=max(others,key=lambda r:(r["branch_whiteouts"],-r["record"]["index"]))
    pair=[fewest,most]
    registration=dict(rule="fewest and most suffix whiteouts among the fixed cohort; distinct tapes, earliest index breaks ties",
                      cost_cents=50,fresh_evaluation_seed=271343,buttons_per_fresh_game=65536,selection=pair)
    write_json(out/"selected.json",registration)
    datasets=[]
    checks=[]
    for item in pair:
        cohort=Path(item["cohort"])
        record=item["record"]
        write_json(out/"status.json",dict(time=time.time(),exporting_training_attempt=record["index"],
                   buttons=record["length"],verified_training_exports=len(datasets)))
        call("export_reward_branch.py",cohort,out/"verified-training-tapes","--record",cohort/record["audit"]/"record.json")
        dataset=out/"verified-training-tapes"/record["tape_sha256"]
        datasets.append(dataset)
        checks.append(check_stream(dataset,50))
    write_json(out/"input-checks.json",dict(input_checks=checks))
    updates=128*math.ceil((max(r["record"]["length"] for r in pair)+1)/(128*32))
    results={}
    for name,cost in (("control",0),("loss-cost",50)):
        fit=out/name/"fit"
        while not (fit/"result.json").exists():
            if available()<3000000:
                wait("3000000 KiB for the paired Dreamer fit")
                continue
            with (root/"controls/large-optimizer.lock").open("a") as slot:
                try:
                    fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    wait("active optimizer")
                    continue
                if available()<3000000:
                    continue
                write_json(out/"status.json",dict(time=time.time(),training=name,optimizer_updates=updates))
                call("train_earned_dreamer.py","--dataset",datasets[0],"--dataset",datasets[1],"--source-run",parent,
                     "--output",fit,"--updates",updates,"--upstream",upstream,"--whiteout-cost-cents",cost)
        results[name]=read(fit/"result.json")
    for key in ("parent_sha256","upstream_revision","training_tapes","optimizer_updates","observations_trained","imagination_length"):
        assert results["control"][key]==results["loss-cost"][key]
    games={}
    for name in ("control","loss-cost"):
        arm=out/name
        existing=sorted(arm.glob("fresh-*"))
        trial=next((p for p in existing if (p/"result.json").exists()),None)
        if trial is None:
            while available()<1700000:
                wait("memory for frozen games")
            trial=arm/f"fresh-{len(existing)+1:03d}"
            write_json(out/"status.json",dict(time=time.time(),evaluating=name,directory=str(trial)))
            call("play_frozen_dreamer.py",arm/"fit/frozen-policy",parent,trial,"--upstream",upstream,
                 "--seed",271343,"--limit",65536,"--log-whiteouts")
        result=read(trial/"result.json")
        assert result["seed"]==271343 and result["in_game_restores"]==0 and result["weight_sha256"]==results[name]["trained_sha256"]
        games[name]=[]
        for i in range(4):
            game=max((read(p) for p in (trial/"games"/f"env-{i}").glob("step-*/*.json")),key=lambda row:row["steps"])
            games[name].append(dict(events=game["events"],steps=game["steps"],maps=game["maps_first_reached"],whiteouts=game["whiteouts"]))
    write_json(out/"result.json",dict(registration=registration,games=games))

if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so",
                      OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",JAX_PLATFORMS="cuda",XLA_PYTHON_CLIENT_PREALLOCATE="false")
    out.mkdir(exist_ok=True)
    with (out/"job.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
