"""Select improved self-earned histories by scalar return, train on them and evaluate from boot."""
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from reward_branch_search import write_json

root=project/"runs/pixels-only-01/official-wsl"
family=root/"earned-frontier-curriculum"
paired=root/"paired-self-earned-frontiers"
teacher=root/"reward-only-training-forks"
early=teacher/"verified-training-tapes/24479561ff0b9fe07b3932aeb611e70d3b7b7f4b42a9f6fe428cea84e19fa412"
parent=root/"seed-71299-h15/eval-65536"


def read(path):
    return json.loads(path.read_text())


def choose(minimum):
    eligible=[]
    for arm in [teacher,*paired.glob("*")]:
        if not arm.is_dir() or not (arm/"settings.json").is_file() or not (arm/"best.json").is_file():
            continue
        settings,record=read(arm/"settings.json"),read(arm/"best.json")
        if not settings.get("training_forks") or not record["training_only"]:
            continue
        if settings.get("receives_saves_from_other_runs"):
            from earned_starts import verified_reset_lineage
            if not (arm/"earned-start-provenance.json").exists() or not verified_reset_lineage(arm,settings):
                continue
        if record["return_cents"]>=minimum:
            eligible.append((arm,record))
    return max(eligible,key=lambda row:(row[1]["return_cents"],-row[1]["length"])) if eligible else None


def call(log,script,*args):
    command=[sys.executable,"-u",str(project/script),*map(str,args)]
    with log.open("a") as dest:
        subprocess.run(command,stdout=dest,stderr=subprocess.STDOUT,check=True)


def collect():
    # Only cumulative scalar return enters this selection. Map names and story flags remain audit data.
    baseline=read(paired/"continued-reward-search/branch-origin.json")["parent_record"]["return_cents"]
    while True:
        done=[read(p) for p in family.glob("stage-*/trained-memory.json")]
        threshold=max([baseline]+[r["return_cents"] for r in done])+100
        selected=choose(threshold)
        if selected is None:
            time.sleep(60)
            continue
        arm,record=selected
        sha=record["tape_sha256"]
        path=arm/record["audit"]/"record.json"
        tape=path.with_name("game.bin").read_bytes()
        audit=read(path.with_name("game.json"))
        if (hashlib.sha256(tape).hexdigest()!=sha or len(tape)!=record["length"]
                or len(tape)!=audit["steps"] or audit["return_cents"]!=record["return_cents"]):
            raise ValueError("The saved action history and its training record disagree")
        if not path.exists():
            # Earlier search workers wrote best.json and the game, without a per-discovery record.
            write_json(path,record)
        if read(path)!=record:
            raise ValueError("Conflicting records for the same training discovery")
        stage=family/f"stage-{record['return_cents']:06d}-{sha[:12]}"
        stage.mkdir(exist_ok=True)
        registration=dict(source=str(arm),source_record=str(path),source_tape_sha256=sha,return_cents=record["return_cents"],
            source_training_search_buttons=record["work"],decisions=record["length"],minimum_return_cents=threshold,
            selection="highest durable scalar return among self-generated training arms; shorter tape breaks ties")
        write_json(stage/"selection.json",registration)
        call(stage/"export.log","export_reward_branch.py",arm,family/"verified-training-tapes","--record",path)
        data=family/"verified-training-tapes"/sha
        if read(data/"manifest.json")["frozen_evaluation_data_used"]:
            raise ValueError("Evaluation contamination")
        budget=max(65536,2**math.ceil(math.log2(record["length"]+8192)))
        # If a previous attempt was interrupted, its tapes are preserved in its own directory.
        attempts=sorted(stage.glob("fresh-*"))
        finished=next((p for p in attempts if (p/"result.json").exists()),None)
        if finished is None:
            finished=stage/f"fresh-{len(attempts)+1:03d}"
            call(stage/"memory.log","earned_memory.py","--dataset",early,"--dataset",data,
                 "--model",stage/"model","--output",finished,"--seed","197461","--limit",budget)
        result=read(finished/"result.json")
        write_json(stage/"trained-memory.json",dict(**registration,fresh_evaluation=str(finished),
            model_sha256=result["model_sha256"],button_budget=budget,verified_badge=result["verified_badge"]))
        if result["verified_badge"]:
            return


def official():
    while True:
        first=root/"earned-official-updates/adaptation/result.json"
        stages=[p.parent for p in family.glob("stage-*/selection.json") if not (p.parent/"official-comparison.json").exists()]
        available=[]
        for stage in stages:
            selected=read(stage/"selection.json")
            data=family/"verified-training-tapes"/selected["source_tape_sha256"]
            if (data/"manifest.json").exists():
                available.append((selected["return_cents"],stage,data))
        mem={line.split(":")[0]:int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
        if not first.exists() or not available:
            time.sleep(60)
            continue
        _,stage,data=max(available)
        selected=read(stage/"selection.json")
        # Round up to a fixed block so the longer lane sees at least one complete original-ROM game.
        updates=max(1152,128*math.ceil((selected["decisions"]+1)/(128*32)))
        budget=max(65536,2**math.ceil(math.log2(selected["decisions"]+8192)))
        fit=stage/"official-fit"
        if not (fit/"result.json").exists():
            if mem["MemAvailable"] < 3000000:
                time.sleep(60)
                continue
            with (root/"controls/large-optimizer.lock").open("a") as gate:
                try:
                    fcntl.flock(gate,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(60)
                    continue
                call(stage/"official.log","train_earned_dreamer.py","--dataset",early,"--dataset",data,"--source-run",parent,
                     "--output",fit,"--upstream","/opt/emerald-rl/source","--updates",updates)
        elif mem["MemAvailable"] < 1700000:
            time.sleep(60)
            continue
        call(stage/"official.log","play_frozen_dreamer.py",fit/"frozen-policy",parent,stage/"official-fresh",
             "--upstream","/opt/emerald-rl/source","--limit",budget)
        check=read(fit/"result.json")
        result=read(stage/"official-fresh/result.json")
        if result["weight_sha256"]!=check["trained_sha256"]:
            raise ValueError("The frozen run used a different model")
        win=(stage/"official-fresh/verified-stone-badge.json").exists()
        write_json(stage/"official-comparison.json",dict(optimizer_updates=updates,fixed_button_budget=budget,
            checkpoint=check["trained_sha256"],fresh_evaluation=result,verified_badge=win))
        if win:
            return


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--world-model",action="store_true")
    world=parser.parse_args().world_model
    family.mkdir(exist_ok=True)
    with (family/("official.lock" if world else "selection.lock")).open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (official if world else collect)()
