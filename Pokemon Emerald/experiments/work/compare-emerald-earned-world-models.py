"""Compare fresh games at identical horizons and independently replay the first new learned starter."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from reward_branch_search import write_json
root=project/"runs/pixels-only-01/official-wsl"
read=lambda path:json.loads(path.read_text())
events=("bedroom","clock","sent_to_meet_rival","met_rival","route101","heard_birch","starter","lab_handoff",
        "rival_battle","pokedex","wally","woods","stone_badge")


def report(arm,horizon):
    trial=arm/"fresh-65536"
    control=root/"controls/direct-parent-65536"
    adapted=read(arm/"adaptation/result.json")
    metadata=read(trial/"provenance.json")
    reference=read(control/"result.json")
    assert metadata["weight_sha256"]==adapted["trained_sha256"] and reference["weight_sha256"]==adapted["parent_sha256"]
    assert metadata["seed"]==reference["seed"] and metadata["buttons_per_game"]==reference["buttons_per_game"]==65536
    assert metadata["in_game_restores"]==reference["in_game_restores"]==0 and metadata["weight_updates"]==reference["weight_updates"]==0
    comparisons={}
    for label,run in (("unchanged_parent",control),("trained_world_model",trial)):
        runs=[]
        for i in range(4):
            candidates=sorted((run/"games"/f"env-{i}").glob("step-*/*episode-00001.json"),
                              key=lambda p:int(p.parent.name.split("-")[-1]),reverse=True)
            if not candidates:
                return
            file=candidates[0]
            info=read(file)
            if info["steps"]<horizon and "stone_badge" not in info["events"]:
                return
            milestones={k:n for k,n in info["events"].items() if n<=horizon}
            runs.append(dict(worker=i,events=milestones,source=str(file)))
        comparisons[label]=dict(counts={name:sum(name in r["events"] for r in runs) for name in events},games=runs)
    write_json(arm/f"compared-at-{horizon}.json",dict(horizon=horizon,seed=metadata["seed"],games_per_arm=4,
               updates=adapted["optimizer_updates"],source_hash=adapted["parent_sha256"],adapted_hash=adapted["trained_sha256"],
               same_original_rom=True,same_policy_runner=True,comparisons=comparisons))


def verify_first(arm,order=0):
    trial=arm/"fresh-65536"
    dest=arm/("first-neural-starter" if order==0 else f"neural-starter-{order+1}")
    if (dest/"verified.json").exists():
        return
    candidates={}
    for a in (trial/"games").glob("*/step-*/*episode-00001.json"):
        data=read(a)
        if "starter" in data["events"]:
            worker=a.parent.parent.name
            candidate=(data["events"]["starter"],data["steps"],a)
            if worker not in candidates or candidate<candidates[worker]:
                candidates[worker]=candidate
    if len(candidates)<=order:
        return
    first,_,audit=sorted(candidates.values())[order]
    data=read(audit)
    tape=audit.with_suffix(".bin").read_bytes()
    assert len(tape)==data["steps"]
    dest.mkdir(parents=True,exist_ok=True)
    cut=dest/"game.json"
    cut.with_suffix(".bin").write_bytes(tape[:first])
    write_json(cut,dict(steps=first,events={k:n for k,n in data["events"].items() if n<=first},
               maps_first_reached={k:n for k,n in data.get("maps_first_reached",{}).items() if n<=first},
               uncapped_exploration=data["uncapped_exploration"]))
    subprocess.run([sys.executable,str(project/"pixel_replay.py"),str(cut),str(dest/"replay"),"--event","starter","--video"],check=True)
    proof=read(dest/"replay/verification.json")
    assert proof["verified_decision"]==first and proof["restores_during_game"]==0 and proof["input_sha256"]==hashlib.sha256(tape[:first]).hexdigest()
    fit=read(arm/"adaptation/result.json")
    overlap={}
    for sha in fit["training_tapes"]:
        options=[root/"earned-online-datasets"/sha,root/"reward-only-training-forks/verified-training-tapes"/sha,
                 root/"paired-self-earned-frontiers/visual-slots-0/verified-training-tapes"/sha]
        source=next(p for p in options if (p/"manifest.json").is_file())
        recorded,=(source/"audit").glob("*.bin")
        prior=recorded.read_bytes()
        assert hashlib.sha256(prior).hexdigest()==sha
        overlap[sha]=next((i for i,(a,b) in enumerate(zip(tape[:first],prior)) if a!=b),min(first,len(prior)))
    write_json(dest/"verified.json",dict(source=str(audit),weight_sha256=fit["trained_sha256"],policy="frozen official Dreamer",
               first_starter_at=first,matching_first_buttons_per_training_game=overlap,replay=proof))


def monitor():
    arms=[root/"earned-official-updates",*[root/"earned-experience-comparison"/name for name in ("shared-opening","diverse-opening")]]
    while True:
        for arm in arms:
            if not (arm/"fresh-65536/provenance.json").is_file():
                continue
            for n in (32768,65536):
                if not (arm/f"compared-at-{n}.json").exists():
                    report(arm,n)
            verify_first(arm)
        if all((a/"compared-at-65536.json").exists() for a in arms):
            return
        time.sleep(45)


if __name__=="__main__":
    if len(sys.argv)==3:
        verify_first(root/sys.argv[1],int(sys.argv[2])-1)
    else:
        with (root/"earned-experience-comparison/comparison-monitor.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            monitor()
