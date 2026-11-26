"""Compare matched explorer checkpoints and admit a second sweep seed when capacity becomes available."""
import fcntl
import json
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
family=root/"paired-self-earned-frontiers"
out=root/"controls/reward-sweep"
read=lambda path:json.loads(path.read_text())

def checkpoints(arm,thresholds,warm):
    cfg=read(arm/"settings.json")
    origin=read(arm/"branch-origin.json")
    logs=[(read(p),p) for p in sorted((arm/"discoveries").glob("work-*/record.json"))]
    points={}
    for budget in thresholds:
        candidates=[(r,p) for r,p in logs if r["work"]-origin["inherited_training_buttons"]>=budget]
        if candidates:
            record,file=min(candidates,key=lambda item:item[0]["work"])
            measured=read(file.with_name("game.json"))
            cells=record["distinct_cells_across_training"]
            assert cells==measured["distinct_cells_across_training"]
            points[budget]=dict(real_new_decisions=record["work"]-origin["inherited_training_buttons"],total_coarse_cells=cells,
                additional_coarse_cells=cells-warm,tape=record["tape_sha256"],frame_repeat=cfg["frame_repeat"],seed=cfg["seed"])
    return points


def run():
    plan=read(root/"controls/reward-frontier-expansion/plan.json")
    raw=read(root/"controls/trainer-contacts/highest-return-review/measured.json")
    assert raw["input_sha256"]==plan["selected"]["tape_sha256"] and raw["first_visit_cells"]==449
    thresholds=[16384,32768,65536,131072,262144]
    a=family/"systematic-reward-frontiers"
    b=family/"sweep-reward-frontiers"
    second=family/"sweep-reward-frontiers-seed-173742"
    write_json(out/"replication-plan.json",dict(seed=173742,matched_control="systematic-reward-frontiers-seed-173742",
        only_if="more new coarse cells at both the first 16k and 32k sealed checkpoints on seed 173741",
        starts="once the first original control ends and WSL has at least 2 GB available",
        observation="the first 16k checkpoint had been reviewed before registering replication"))
    job=None
    while True:
        results={n:checkpoints(p,thresholds,449) for n,p in (("ordinary",a),("sweep",b))}
        if (second/"status.json").exists():
            results.update(second_ordinary=checkpoints(family/"systematic-reward-frontiers-seed-173742",thresholds,449),
                           second_sweep=checkpoints(second,thresholds,449))
        write_json(out/"matched-coverage.json",dict(source=raw["input_sha256"],inherited_coarse_cells=449,
            rule="first sealed branch with at least the nominal number of new decisions",arms=results))
        if job and job.poll() is not None:
            if job.returncode:
                raise RuntimeError("Second sweep worker failed")
            if 262144 in results.get("second_ordinary",{}):
                return
        if job is None and 32768 in results["sweep"] and 32768 in results["ordinary"]:
            qualifies=all(results["sweep"][k]["additional_coarse_cells"]>results["ordinary"][k]["additional_coarse_cells"] for k in thresholds[:2])
            if not qualifies:
                if 262144 in results["sweep"] and 262144 in results["ordinary"]:
                    return
            elif (a/"completed.json").exists() and not (second/"completed.json").exists():
                available=int(next(s.split()[1] for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:")))
                if available>=2000000:
                    with (out/"seed-173742.log").open("a") as log:
                        job=subprocess.Popen([sys.executable,"-u",str(project/"explore_reward_frontiers.py"),
                            "--source",plan["source"],"--record",str(root/"controls/reward-frontier-expansion/selected-record.json"),
                            "--output",str(second),"--motion",str(root/"controls/visual-motion/route-training.json"),
                            "--steps",str(plan["steps"]),"--capacity",str(plan["capacity"]),"--seed","173742","--reward-sweep"],
                            stdout=log,stderr=subprocess.STDOUT)
            elif (second/"completed.json").exists() and 262144 in results.get("second_ordinary",{}):
                return
        time.sleep(60)


if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so",
                      OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
    with (out/"coverage.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
