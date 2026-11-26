"""Prospectively compare scalar-return selection with return plus new self-generated actions."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from train_earned_dreamer import write_json
root=project/"runs/pixels-only-01/official-wsl"
parent=root/"seed-71299-h15/eval-65536"
family=root/"paired-self-earned-frontiers"
arms=[family/"systematic-reward-frontiers",family/"systematic-reward-frontiers-seed-173742"]
out=root/"training-selection-comparison"
upstream=Path("/opt/emerald-rl/source")
BUDGET=131072
read=lambda path:json.loads(path.read_text())


def suffix(a,b):
    common=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
    return len(a)-common


def select():
    modes=("return-only","return-and-new-actions")
    groups={name:[] for name in modes}
    prior={name:[] for name in modes}
    parent_sha=None
    for arm in arms:
        origin=read(arm/"branch-origin.json")
        cfg=read(arm/"settings.json")
        source=origin["parent_record"]
        tape=(Path(origin["source_run"])/source["audit"]/"game.bin").read_bytes()
        sha=hashlib.sha256(tape).hexdigest()
        assert sha==source["tape_sha256"]==cfg["source_sha256"] and source["training_only"]
        assert parent_sha in (None,sha) and cfg["training_forks"] and not cfg["receives_saves_from_other_runs"]
        parent_sha=sha
        assert read(arm/"status.json")["new_buttons"]>=BUDGET
        candidates=[]
        for record_file in sorted((arm/"discoveries").glob("work-*/record.json")):
            record=read(record_file)
            if not 0<record["work"]-origin["inherited_training_buttons"]<=BUDGET:
                continue
            assert record["training_only"] and record_file.parent.resolve()==(arm/record["audit"]).resolve()
            candidate=record_file.with_name("game.bin").read_bytes()
            assert hashlib.sha256(candidate).hexdigest()==record["tape_sha256"] and len(candidate)==record["length"]
            candidates.append((record,candidate))
        assert candidates
        for mode in modes:
            rows=[]
            for record,buttons in candidates:
                distinct=min(suffix(buttons,prefix) for prefix in [tape,*prior[mode]])
                rows.append((record,buttons,distinct))
            returns=[r["return_cents"] for r,_,_ in rows]
            suffixes=[d for _,_,d in rows]
            def order(row):
                r,_,d=row
                if mode=="return-only":
                    return (-r["return_cents"],r["length"],r["tape_sha256"])
                rank_return=sum(x<r["return_cents"] for x in returns)+.5*returns.count(r["return_cents"])
                rank_suffix=sum(x<d for x in suffixes)+.5*suffixes.count(d)
                return (-min(rank_return,rank_suffix),-rank_return-rank_suffix,r["tape_sha256"])
            record,buttons,distinct=min(rows,key=order)
            prior[mode].append(buttons)
            groups[mode].append(dict(source=str(arm),record=record,candidates=len(rows),new_suffix=distinct))
    return groups


def available():
    return int(next(s.split()[1] for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:")))


def call(script,*args):
    with (out/"experiment.log").open("a") as log:
        subprocess.run([sys.executable,"-u",str(project/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True)


def run():
    plan=dict(budget_per_training_search=BUDGET,seeds=[173741,173742],frozen_evaluation_seed=271345,
        hardware_frames_per_button=24,buttons_per_fresh_game=65536,games_per_arm=4,
        rule="one sealed periodic tape per seed within the first 131072 new buttons",
        control="highest scalar return, then shortest, then SHA",
        alternative="maximize weaker midrank of return and new suffix, then their sum, then SHA",
        new_suffix="buttons after longest shared prefix with the inherited tape or the previous seed's selected tape",
        training="no map names, positions, trainer information, or fresh evaluation data enter selection",
        fit="same official parent; match optimizer updates across both pairs, rounded up to cover the longest game",
        identical_data="deduplicate identical ordered datasets")
    marker=out/"registration.json"
    if marker.exists() and read(marker)!=plan:
        raise ValueError("Existing registration differs")
    write_json(marker,plan)
    while min(read(a/"status.json")["new_buttons"] for a in arms)<BUDGET:
        time.sleep(30)
    groups=select()
    write_json(out/"selected.json",groups)
    length=max(entry["record"]["length"] for group in groups.values() for entry in group)
    updates=128*math.ceil((length+1)/(128*32))
    datasets=out/"verified-training-tapes"
    for group in groups.values():
        for entry in group:
            arm=Path(entry["source"])
            record=entry["record"]
            call("export_reward_branch.py",arm,datasets,"--record",arm/record["audit"]/"record.json")
            assert not read(datasets/record["tape_sha256"]/"manifest.json")["frozen_evaluation_data_used"]
    fitted={}
    results={}
    for name,group in groups.items():
        tapes=tuple(e["record"]["tape_sha256"] for e in group)
        if tapes in fitted:
            results[name]=dict(identical_inputs_to=fitted[tapes])
            continue
        fitted[tapes]=name
        arm=out/name
        fit=arm/"fit"
        while not (fit/"result.json").exists():
            with (root/"controls/large-optimizer.lock").open("a") as gate:
                if available()>=3000000:
                    try:
                        fcntl.flock(gate,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        if available()>=3000000:
                            call("train_earned_dreamer.py","--dataset",datasets/tapes[0],"--dataset",datasets/tapes[1],
                                "--source-run",parent,"--output",fit,"--upstream",upstream,"--updates",updates)
                            continue
            time.sleep(30)
        details=read(fit/"result.json")
        assert details["training_tapes"]==list(tapes) and details["optimizer_updates"]==updates
        results[name]=dict(weights=details["trained_sha256"],parent=details["parent_sha256"])
    for name in fitted.values():
        arm=out/name
        prior=sorted(arm.glob("fresh-*"))
        game=next((p for p in prior if (p/"result.json").exists()),None)
        if game is None:
            # Let the two matched survival evaluations finish before admitting another long GPU player.
            while available()<2000000 or not (root/"outcome-cost-comparison/result.json").exists():
                time.sleep(30)
            game=arm/f"fresh-{len(prior)+1:03d}"
            with (root/"controls/followup-evaluation.lock").open("a") as slot:
                fcntl.flock(slot,fcntl.LOCK_EX)
                call("play_frozen_dreamer.py",arm/"fit/frozen-policy",parent,game,"--upstream",upstream,
                    "--seed",271345,"--limit",65536,"--log-whiteouts")
        outcome=read(game/"result.json")
        assert outcome["weight_sha256"]==results[name]["weights"] and outcome["in_game_restores"]==0 and outcome["seed"]==271345
        results[name]["fresh_directory"]=str(game)
    write_json(out/"result.json",dict(updates=updates,arms=results))


if __name__=="__main__":
    out.mkdir(exist_ok=True)
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so",
                      OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",JAX_PLATFORMS="cuda",XLA_PYTHON_CLIENT_PREALLOCATE="false")
    with (out/"job.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
