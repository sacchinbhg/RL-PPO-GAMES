"""Verify first Petalburg Woods entries from the autonomous experiment's saved original-ROM games."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
root=project/"runs/pixels-only-01/official-wsl"
output=root/"forest-watch"
sys.path.insert(0,str(project))
from train_earned_dreamer import write_json


def inputs():
    for arm in [root/"reward-only-training-forks",*(root/"paired-self-earned-frontiers").glob("*")]:
        if arm.is_dir():
            yield "training-discovery",arm,"first-maps/PetalburgWoods/game.json"
            yield "training-discovery",arm,"discoveries/*/game.json"
            yield "training-discovery",arm,"first-trainers/trainer-*/game.json"
    for arm in [*(root/"earned-reset-online-comparison").glob("*/train"),*(root/"outcome-curriculum").glob("*/train")]:
        yield "training-official-dreamer",arm,"in-progress/*/*/step-*/*episode-*.json"
        yield "training-official-dreamer",arm,"external-audit/*/*/*episode-*.json"
    arm=root/"earned-button-alternatives/training"
    if arm.is_dir():
        yield "training-button-alternative",arm,"first-maps/PetalburgWoods/game.json"
        yield "training-button-alternative",arm,"first-trainers/trainer-*/game.json"
        yield "training-button-alternative",arm,"terminal/*/game.json"
    for arm in root.glob("seed-*-h*/eval-*"):
        provenance=arm/"provenance.json"
        if provenance.is_file() and "agent_sha256" in json.loads(provenance.read_text()):
            yield "frozen-official-policy",arm,"in-progress/*/*/step-*/*episode-00001.json"
            yield "frozen-official-policy",arm,"external-audit/*/*/*episode-00001.json"
            yield "frozen-official-policy",arm,"live-map-replay/*/first-maps/PetalburgWoods/game.json"
    direct=list((root/"seed-71299-h15/unbounded-seed-271341").glob("fresh-*"))
    direct+=list((root/"reward-only-training-forks/actor-fits").glob("*/fresh-*"))
    direct.append(root/"earned-official-updates/fresh-65536")
    direct.extend((root/"earned-official-updates/uncapped-training").glob("fresh-*"))
    direct.extend((root/"outcome-cost-comparison").glob("*/fresh-*"))
    direct.extend((root/"training-selection-comparison").glob("*/fresh-*"))
    direct.extend((root/"battle-win-comparison").glob("*/fresh-*"))
    direct.extend((root/"earned-reset-online-comparison").glob("*/fresh-*"))
    direct.extend(root/"earned-official-updates/long-pair"/name for name in ("adapted","unchanged"))
    direct.extend((root/"earned-experience-comparison").glob("*/fresh-65536"))
    for arm in direct:
        if (arm/"provenance.json").exists():
            yield "frozen-dreamer-policy",arm,"first-maps/*/*/PetalburgWoods/*.json"
            yield "frozen-dreamer-policy",arm,"games/*/step-*/*episode-00001.json"
    for arm in (root/"earned-visual-model").glob("*/fresh-*"):
        yield "frozen-tabular-policy",arm,"step-*/*.json"
    for arm in [*(root/"earned-route-extension").glob("trial-*"),*(root/"outcome-curriculum").glob("*combat-training/fresh-*")]:
        for i,label in enumerate(("earned-memory-then-dreamer","earned-memory-then-uniform")):
            yield label,arm,f"first-maps/env-{i}/*/PetalburgWoods/*.json"
            yield label,arm,f"games/env-{i}/step-*/*episode-00001.json"
    for arm in (root/"earned-route-recovery").glob("trial-*"):
        for i,label in enumerate(("earned-memory-dreamer-visual-recovery","recovery-trial-uniform-control")):
            yield label,arm,f"first-maps/env-{i}/*/PetalburgWoods/*.json"
            yield label,arm,f"games/env-{i}/step-*/*episode-00001.json"
    for label,kind in (("clean","frozen-earned-memory"),("perturbed","memory-robustness-trial"),
                       ("exact-perturbed","exact-memory-robustness-trial"),("exact-clean","exact-frozen-earned-memory"),
                       ("loop-recovery-clean","memory-loop-recovery"),("loop-recovery-perturbed","memory-loop-recovery-perturbed"),
                       ("loop-replication-control","memory-loop-control-replication"),("loop-replication-treatment","memory-loop-recovery-replication")):
        arm=root/"earned-screen-memory"/label
        if (arm/"provenance.json").is_file():
            yield kind,arm,"first-maps/*/*/PetalburgWoods/*.json"
            yield kind,arm,"step-*/*.json"
    for arm in (root/"earned-screen-memory").glob("persistent-*-*"):
        yield "frozen-button-effect-policy",arm,"first-maps/*/*/PetalburgWoods/*.json"
        yield "frozen-button-effect-policy",arm,"step-*/*.json"
    for stage in (root/"earned-frontier-curriculum").glob("stage-*"):
        for arm in stage.glob("fresh-*"):
            yield "frozen-earned-memory",arm,"first-maps/*/*/PetalburgWoods/*.json"
        arm=stage/"official-fresh"
        if (arm/"provenance.json").exists():
            yield "frozen-dreamer-policy",arm,"first-maps/*/*/PetalburgWoods/*.json"


def inspect(kind, arm, audit, info):
    forest=info.get("maps_first_reached",{}).get("PetalburgWoods")
    events=info.get("events",{})
    later_reward="woods" in events
    legacy_progress="maps_first_reached" not in info and any(k in events for k in ("rival_battle","pokedex","wally"))
    if not forest and not (later_reward or legacy_progress):
        return
    tape=audit.with_suffix(".bin").read_bytes()
    if len(tape)!=info["steps"]:
        raise ValueError(f"Incomplete game tape at {audit}")
    sha=hashlib.sha256(tape).hexdigest()
    dest=output/kind/sha
    if (dest/"result.json").exists():
        return
    dest.mkdir(parents=True,exist_ok=True)
    if not forest:
        # Legacy games had story flags only. Establish the actual first map entry by replay.
        subprocess.run([sys.executable,str(project.parent.parent/"work/audit-emerald-path-progress.py"),
                        str(audit),str(dest/"geography")],check=True)
        measured=json.loads((dest/"geography/verified.json").read_text())
        forest=measured["maps_first_reached"].get("PetalburgWoods")
        if not forest:
            write_json(dest/"result.json",dict(entered_forest=False,source_audit=str(audit)))
            return
        info=dict(info,maps_first_reached=measured["maps_first_reached"])
    # Preserve all buttons from boot through the exact first entry. Later buttons stay in the source audit.
    cut={k:v for k,v in info["events"].items() if v<=forest}
    maps={k:v for k,v in info["maps_first_reached"].items() if v<=forest}
    saved=dest/"first-entry.json"
    saved.with_suffix(".bin").write_bytes(tape[:forest])
    write_json(saved,dict(events=cut,maps_first_reached=maps,steps=forest,uncapped_exploration=info.get("uncapped_exploration",False)))
    subprocess.run([sys.executable,str(project/"pixel_replay.py"),str(saved),str(dest/"replay"),
                    "--map","PetalburgWoods","--video"],check=True)
    proof=json.loads((dest/"replay/verification.json").read_text())
    if proof["checked_map"]!="PetalburgWoods" or proof["verified_decision"]!=forest or proof["restores_during_game"]!=0:
        raise ValueError("The original-ROM playback did not verify the first forest entry")
    provenance=arm/"provenance.json"
    if kind=="frozen-tabular-policy":
        provenance=arm.parent/"fit.json"
    metadata=json.loads(provenance.read_text()) if provenance.exists() else {}
    result=dict(entered_forest=True,kind=kind,source_run=str(arm),source_audit=str(audit),
                complete_source_sha256=sha,forest_entry=forest,first_entry_sha256=proof["input_sha256"],
                original_rom_sha1=proof["rom_sha1"],replay=str(dest/"replay/verification.json"),
                weight_sha256=metadata.get("weight_sha256") or metadata.get("agent_sha256") or metadata.get("model_sha256"))
    write_json(dest/"result.json",result)
    if not (output/f"first-{kind}.json").exists():
        write_json(output/f"first-{kind}.json",result)
    print(json.dumps(result),flush=True)


def inspect_rival(kind,arm,audit,info):
    encounter_audit=audit.parent.parent.name=="first-trainers" and audit.parent.name in {
        f"trainer-{i:05d}" for i in (520,523,526,529,532,535)}
    story=info.get("events",{}).get("rival_battle")
    if (not story and not encounter_audit) or "first_visit_cells" not in info:
        return
    dest=root/"controls/trainer-contacts"/kind
    if (dest/"first-verified-win.json").exists():
        return
    data=audit.with_suffix(".bin").read_bytes()
    when=story or info["steps"]
    prefix=hashlib.sha256(data[:when]).hexdigest()
    attempt=dest/prefix
    marker=attempt/"review.json"
    if marker.exists():
        return
    attempt.mkdir(parents=True,exist_ok=True)
    battles=attempt/"battles.json"
    if not battles.exists():
        subprocess.run([sys.executable,str(Path(__file__).with_name("diagnose-emerald-battles.py")),str(audit),str(battles)],check=True)
    result=json.loads(battles.read_text())
    rivals=set(result["observed_route103_rival_ids"])
    encounters=[b for b in result["recorded_battles"] if b["previous_overworld_map"]=="Route103" and rivals.intersection(b["trainer_ids"])]
    first=dest/"first-verified-encounter.json"
    if encounters and not first.exists():
        assert result["in_game_restores"]==0 and result["input_sha256"]==hashlib.sha256(data).hexdigest()
        write_json(first,dict(source=str(audit),run=str(arm),controller_kind=kind,first_battle=encounters[0],
                   verified_from_boot=True,rom_sha1=result["rom_sha1"],input_sha256=result["input_sha256"]))
    wins=[b for b in result["recorded_battles"] if b["previous_overworld_map"]=="Route103"
          and rivals.intersection(b["trainer_ids"]) and b["first_nonzero_outcome"] and b["first_nonzero_outcome"][1]==1
          and b["first_nonzero_outcome"][0]<=when]
    proof=dict(source=str(audit),run=str(arm),controller_kind=kind,story_decision=story,
               rival_win=bool(wins),winning_battles=wins,rom_sha1=result["rom_sha1"],prefix_sha256=prefix)
    if wins:
        saved=attempt/"first-win.json"
        saved.with_suffix(".bin").write_bytes(data[:when])
        events={k:v for k,v in info["events"].items() if v<=when}
        write_json(saved,dict(steps=when,events=events,uncapped_exploration=info.get("uncapped_exploration",False)))
        subprocess.run([sys.executable,str(project/"pixel_replay.py"),str(saved),str(attempt/"replay"),
                        "--event","rival_battle","--video"],check=True)
        write_json(dest/"first-verified-win.json",proof)
    write_json(marker,proof)


def monitor(once):
    checked=set()
    retry={}
    version=Path(__file__).stat().st_mtime_ns
    while True:
        if not once and Path(__file__).stat().st_mtime_ns!=version:
            os.execv(sys.executable,[sys.executable,"-u",*sys.argv])
        readable=0
        for kind,arm,pattern in inputs():
            # A verified first result in a class is retained; continue to check other controllers.
            if (output/f"first-{kind}.json").exists():
                continue
            for audit in arm.glob(pattern):
                tries,after=retry.get(audit,(0,0))
                if audit in checked or time.time()<after or not audit.with_suffix(".bin").exists():
                    continue
                try:
                    info=json.loads(audit.read_text())
                except (FileNotFoundError,json.JSONDecodeError):
                    continue
                try:
                    inspect(kind,arm,audit,info)
                    inspect_rival(kind,arm,audit,info)
                except (OSError,ValueError,subprocess.CalledProcessError) as error:
                    retry[audit]=(tries+1,time.time()+min(1800,60*2**min(tries,5)))
                    write_json(output/"last-error.json",dict(time=time.time(),audit=str(audit),
                               error=str(error),attempt=tries+1))
                    if once:
                        raise
                    continue
                checked.add(audit)
                retry.pop(audit,None)
                readable+=1
        write_json(output/"last-check.json",dict(time=time.time(),audits_checked=len(checked),new_this_check=readable,
                                                 waiting_for_recheck=len(retry)))
        if once:
            return
        time.sleep(60)


if __name__=="__main__":
    relief=root/"controls/overnight-resource-recovery"
    if (relief/"requested").exists() and not (relief/"relief.json").exists():
        subprocess.run([sys.executable,str(Path(__file__).with_name("emerald-relieve-guest-memory.py"))],check=True)
    parser=argparse.ArgumentParser()
    parser.add_argument("--once",action="store_true")
    args=parser.parse_args()
    output.mkdir(exist_ok=True)
    with (output/"monitor.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        monitor(args.once)
