"""Read the ongoing experiments without communicating with a policy or emulator."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
family=root/"paired-self-earned-frontiers"
important=("met_rival","starter","lab_handoff","rival_battle","pokedex","wally","stone_badge")
later=("Route102","PetalburgCity","Route104","PetalburgWoods","RustboroCity","RustboroCity_Gym")


def read(p):
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return {}


def freshest(arm,worker):
    candidates=[]
    for directory in (arm/"games"/worker,*(arm/"in-progress").glob("*/"+worker)):
        if directory.is_dir():
            steps=[p for p in directory.glob("step-*") if p.name[5:].isdigit()]
            if steps:
                candidates.extend(max(steps,key=lambda p:int(p.name[5:])).glob("*episode-00001.json"))
    candidates+=list((arm/"external-audit").glob("*/"+worker+"/*episode-00001.json"))
    candidates+=list((arm/"latest"/worker).glob("*episode-00001.json"))
    return max((read(p) for p in candidates),key=lambda a:a.get("steps",0),default={})


def games(arm,n=4):
    if not (arm/"provenance.json").is_file():
        return None
    logs=[freshest(arm,f"env-{i}") for i in range(n)]
    if n==1 and not logs[0]:
        completed=sorted(arm.glob("step-*"),key=lambda p:int(p.name[5:]))
        if completed:
            logs=[read(p) for p in completed[-1].glob("*.json")]
    counts=Counter()
    map_counts=Counter()
    for idx,audit in enumerate(logs):
        counts.update(k for k in important if k in audit.get("events",{}))
        live=read(arm/"live-map-replay"/f"env-{idx}"/"status.json")
        maps=set(audit.get("maps_first_reached",{})) | set(live.get("maps_first_reached",{}))
        map_counts.update(k for k in ("OldaleTown","Route103",*later) if k in maps)
    status=read(arm/"status.json")
    return dict(per_game=status.get("decisions_per_game",status.get("decisions",min((a.get("steps",0) for a in logs),default=0))),
                finished=(arm/"result.json").is_file() or len(list((arm/"external-audit").glob("*/*/*episode-00001.json")))>=n,
                story=dict(counts),geography=dict(map_counts))


def collect():
    path=root/"seed-71299-h15"
    arms={
        "ordinary 524304":path/"eval-524288",
        "unchanged 262144-button games":path/"eval-65536-game-262144",
        "1152 updates":root/"earned-official-updates/fresh-65536",
        "1152 updates longer":root/"earned-official-updates/long-pair/adapted",
        "unchanged longer":root/"earned-official-updates/long-pair/unchanged",
        "2176 shared opening":root/"earned-experience-comparison/shared-opening/fresh-65536",
        "2176 diverse opening":root/"earned-experience-comparison/diverse-opening/fresh-65536"}
    for trial in sorted((path/"unbounded-seed-271341").glob("fresh-*")):
        arms[f"unchanged until badge {trial.name}"]=trial
    for trial in sorted((root/"earned-official-updates/uncapped-training").glob("fresh-*")):
        arms[f"uncapped adaptation {trial.name}"]=trial
    for trial in sorted((root/"outcome-cost-comparison").glob("*/fresh-*")):
        arms[f"whiteout comparison {trial.parent.name} {trial.name}"]=trial
    for fit in sorted((root/"earned-frontier-curriculum").glob("stage-*")):
        arms[f"frontier Dreamer {fit.name}"]=fit/"official-fresh"
    data={name:entry for name,arm in arms.items() if (entry:=games(arm))}
    recovery={}
    for trial in sorted((root/"earned-route-recovery").glob("trial-*")):
        if not (trial/"provenance.json").exists():
            continue
        recovery[trial.name]=dict(progress=read(trial/"status.json"),first_changed=read(trial/"first-recovery-change.json"))
        for i,label in enumerate(("dreamer_with_recovery","uniform")):
            audit=freshest(trial,f"env-{i}")
            recovery[trial.name][label]={k:audit[k] for k in ("steps","events","maps_first_reached","whiteouts","maps_after_whiteout") if k in audit}
    training={}
    original=root/"reward-only-training-forks"
    for arm in [original,*sorted(p for p in family.iterdir() if p.is_dir())]:
        origin=read(arm/"branch-origin.json")
        if arm!=original and "inherited_training_buttons" not in origin:
            continue
        status=read(arm/"status.json")
        best=read(arm/"best.json")
        human=[k for k in later if (arm/"first-maps"/k/"game.json").is_file()]
        inherited=origin.get("inherited_training_buttons",0)
        row=dict(extra=status.get("work",inherited)-inherited,
                 best=status.get("best_return_cents"),saved=best.get("return_cents"),later_maps=human)
        if "best_selection_return_cents" in status:
            row["shared_novelty_score"]=status["best_selection_return_cents"]
        if not status:
            row["warmup"]=read(arm/"initialization.json").get("cold_replay_buttons",0)
        training[arm.name]=row
    proof=[p.name for p in (root/"forest-watch").glob("first-*.json") if read(p).get("entered_forest")]
    result=dict(at=datetime.now(timezone.utc).isoformat(),forest_proofs=proof,
                forest_watch=read(root/"forest-watch/last-check.json"),frozen_games=data,training_search=training,
                route_recovery=recovery)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save",action="store_true")
    args=parser.parse_args()
    status=collect()
    if args.save:
        dest=root/"live-scoreboard.json"
        draft=dest.with_suffix(".tmp")
        draft.write_text(json.dumps(status,indent=2))
        draft.replace(dest)
    print(json.dumps(status,separators=(",",":")))
