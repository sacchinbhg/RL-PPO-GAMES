"""Use identical paid search history and a fixed new-emulator-step budget for the two explorers."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from option_explorer import run
from reward_branch_search import write_json


def main(motion=False,landmarks=0,behavior=None,systematic=False,reward_landmarks=False,recent_landmarks=0,shared=False):
    family=project/"runs/pixels-only-01/official-wsl/paired-self-earned-frontiers"
    control=family/"continued-reward-search"
    origin=json.loads((control/"branch-origin.json").read_text())
    name="learned-visual-motion" if motion else "learned-button-durations"
    name=f"{name}-landmarks-{landmarks}" if landmarks else name
    if behavior:
        name+=f"-{behavior}-proposal"
        origin=dict(origin,additional_budget=131072)
    if systematic:
        name+="-systematic-options"
        origin=dict(origin,additional_budget=262144)
    if reward_landmarks:
        name+="-reward-landmarks"
    if recent_landmarks:
        name+=f"-recent-{recent_landmarks}"
    if shared:
        name+="-shared-visit-reward"
    target=family/name
    probe=family.parent/"controls/visual-motion/route-training.json" if motion else None
    if not target.exists():
        relative=f"discoveries/work-{origin['inherited_training_buttons']:010d}"
        source=control/relative
        record=json.loads((source/"record.json").read_text())
        assert record["tape_sha256"]==origin["parent_record"]["tape_sha256"]==hashlib.sha256((source/"game.bin").read_bytes()).hexdigest()
        assert not origin["first_game_frozen_evaluation_used"] and record["training_only"]
        with tempfile.TemporaryDirectory(dir=family) as tmp:
            draft=Path(tmp)/"arm"
            shutil.copytree(source,draft/relative)
            write_json(draft/"best.json",record)
            write_json(draft/"branch-origin.json",dict(**origin,control=str(control),comparison_budget="real emulator buttons"))
            draft.rename(target)
    with (target/"run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        budget=origin["inherited_training_buttons"]+origin["additional_budget"]
        if motion:
            while True:
                mem={line.split(":")[0]:int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
                if mem["MemAvailable"]>=450000 and mem["MemAvailable"]+mem["SwapFree"]>=850000:
                    break
                time.sleep(15)
        proposal=dict(kind=behavior) if behavior else None
        if behavior=="dreamer":
            proposal.update(checkpoint=family.parent/"earned-official-updates/adaptation/frozen-policy",
                            parent=family.parent/"seed-71299-h15/eval-65536",upstream=Path("/opt/emerald-rl/source"))
        run(target,budget,probe,landmarks,proposal,systematic,reward_landmarks,recent_landmarks,shared)
        outcome=json.loads((target/"best.json").read_text())
        status=json.loads((target/"status.json").read_text())
        if not outcome["terminal"] and status["work"]!=budget:
            raise RuntimeError("Option comparison stopped short")
        write_json(target/"completed.json",dict(status=status,best=outcome))


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--motion",action="store_true")
    parser.add_argument("--landmarks",type=int,choices=[0,64,128],default=0)
    parser.add_argument("--proposal",choices=["dreamer","uniform"])
    parser.add_argument("--systematic-options",action="store_true")
    parser.add_argument("--reward-landmarks",action="store_true")
    parser.add_argument("--recent-landmarks",type=int,default=0)
    parser.add_argument("--shared-visit-reward",action="store_true")
    args=parser.parse_args()
    main(args.motion,args.landmarks,args.proposal,args.systematic_options,args.reward_landmarks,args.recent_landmarks,args.shared_visit_reward)
