"""Continue the highest-return arm after an equal-budget, self-generated training comparison."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

project = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL"
sys.path.insert(0, str(project))
from reward_branch_search import train, write_json

family = project / "runs/pixels-only-01/official-wsl/paired-self-earned-frontiers"


def read(path):
    return json.loads(path.read_text())


def main():
    source = read(family / "selection.json")
    arms = [family / f"visual-slots-{slots}" for slots in (0, 160)]
    while not all((arm / "completed.json").exists() for arm in arms):
        time.sleep(15)
    trials = [(arm, read(arm / "completed.json")) for arm in arms]
    extra = 0
    for arm, trial in trials:
        if trial["best"]["terminal"]:
            print("Found a terminal training discovery for independent verification:", arm, flush=True)
            return
        additional = trial["status"]["work"] - source["inherited_work"]
        assert additional == source["additional_search_buttons"]
        extra += additional
    arm, result = max(trials, key=lambda row: (row[1]["best"]["return_cents"], -row[1]["best"]["length"]))
    record = result["best"]
    discovery = arm / record["audit"]
    assert read(discovery / "record.json") == record
    assert hashlib.sha256((discovery / "game.bin").read_bytes()).hexdigest() == record["tape_sha256"]
    inherited = source["inherited_work"] + extra
    seed = 465721
    new_buttons = 1000000
    output = family / "continued-reward-search"
    if not output.exists():
        with tempfile.TemporaryDirectory(dir=family) as scratch:
            stage = Path(scratch) / "run"
            relative = f"discoveries/work-{inherited:010d}"
            copied = stage / relative
            shutil.copytree(discovery, copied)
            # Charge both completed arms. Their common ancestor was executed once.
            branch_sum = sum(t["status"]["branches"] for _, t in trials)
            ancestor = read(arm / "discoveries" / f"work-{source['inherited_work']:010d}" / "record.json")
            combined_branches = branch_sum - ancestor["branches"]
            opening = dict(record, audit=relative, work=inherited, branches=combined_branches)
            write_json(copied / "record.json", opening)
            write_json(stage / "best.json", opening)
            write_json(stage / "branch-origin.json", dict(parent_arm=arm.name, parent_record=record,
                seed=seed, parent_comparison={p.name: t for p,t in trials},
                criterion="highest saved scalar return among equal-budget arms; shorter tape breaks ties",
                inherited_training_buttons=inherited, both_prior_cold_replays=2*source["replayed_buttons_at_start"],
                own_cold_replay=record["length"], additional_budget=new_buttons,
                inherited_restore_branches=combined_branches, first_game_frozen_evaluation_used=False))
            stage.rename(output)
    origin = read(output / "branch-origin.json")
    assert origin["parent_record"]["tape_sha256"] == record["tape_sha256"] and origin["inherited_training_buttons"] == inherited
    print("Continuing from an independently earned training history", arm.name, record["tape_sha256"], flush=True)
    with (output / "search.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "completed.json").exists():
            return
        train(output, seed, inherited + new_buttons, read(arm / "settings.json").get("unrewarded_screen_slots", 0))
        latest = read(output / "status.json")
        best = read(output / "best.json")
        if latest["work"] != inherited + new_buttons and not best["terminal"]:
            raise RuntimeError("Continued search ended early")
        write_json(output / "completed.json", dict(status=latest, best=best))


if __name__ == "__main__":
    main()
