"""Start two training-only search arms from the same automatically selected, verified earned history."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

project = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL"
root = project / "runs/pixels-only-01/official-wsl"
parent = root / "reward-only-training-forks"
family = root / "paired-self-earned-frontiers"


def main():
    family.mkdir(exist_ok=True)
    setup = family / "selection.json"
    if not setup.exists():
        options = [json.loads(p.read_text()) for p in (parent / "verified-training-tapes").glob("*/manifest.json")]
        chosen = max(options, key=lambda r: (r["return_cents"], -r["decisions"]))
        assert not chosen["frozen_evaluation_data_used"]
        config = dict(source_tape=chosen["source_sha256"], inherited_work=chosen["search_buttons_at_discovery"],
                      replayed_buttons_at_start=chosen["decisions"], seed=99281, additional_search_buttons=131072,
                      selection="maximum scalar return, then shortest among verified self-generated training tapes")
        setup.write_text(json.dumps(config, indent=2))
    config = json.loads(setup.read_text())
    source = parent / "discoveries" / f"work-{config['inherited_work']:010d}"
    record = json.loads((source / "record.json").read_text())
    assert hashlib.sha256((source / "game.bin").read_bytes()).hexdigest() == record["tape_sha256"] == config["source_tape"]
    processes = []
    for count in (160, 0):
        arm = family / f"visual-slots-{count}"
        if not arm.exists():
            with tempfile.TemporaryDirectory(dir=family) as tmp:
                stage = Path(tmp) / "arm"
                dest = stage / record["audit"]
                shutil.copytree(source, dest)
                (stage / "best.json").write_text(json.dumps(record, indent=2))
                (stage / "branch-origin.json").write_text(json.dumps(config, indent=2))
                stage.rename(arm)
        if (arm / "completed.json").exists():
            continue
        cmd = [sys.executable, "-u", str(project / "reward_branch_search.py"), "--logdir", str(arm),
               "--seed", str(config["seed"]), "--steps", str(config["inherited_work"]+config["additional_search_buttons"]),
               "--unrewarded-screen-slots", str(count)]
        with (arm / "search.log").open("a") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        processes.append((arm, process))
    for arm, proc in processes:
        if proc.wait() != 0:
            raise RuntimeError(str(arm))
        status = json.loads((arm / "status.json").read_text())
        outcome = json.loads((arm / "best.json").read_text())
        if not outcome["terminal"]:
            assert status["work"] == config["inherited_work"] + config["additional_search_buttons"]
        (arm / "completed.json").write_text(json.dumps({"status":status,"best":outcome}, indent=2))


if __name__ == "__main__":
    import fcntl
    family.mkdir(exist_ok=True)
    with (family / "pair.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
