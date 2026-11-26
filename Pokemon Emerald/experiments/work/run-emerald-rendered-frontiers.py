"""Validate decision-frame rendering on real training forks, then use it in a second seeded exploration run."""
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
from reward_branch_search import write_json
root=project/"runs/pixels-only-01/official-wsl"
out=root/"controls/last-frame-render/frontier-integration"
registration=root/"controls/reward-frontier-expansion"
family=root/"paired-self-earned-frontiers"
read=lambda p:json.loads(p.read_text())

def train(source,record,dest,seed,steps,capacity,reduced):
    args=[sys.executable,"-u",str(project/"explore_reward_frontiers.py"),"--source",str(source),
          "--record",str(record),"--motion",str(root/"controls/visual-motion/route-training.json"),
          "--output",str(dest),"--seed",str(seed),"--steps",str(steps),"--capacity",str(capacity)]
    if reduced:
        args[2]=str(base/"work/renderer-experimental/explore_reward_frontiers.py")
        args.append("--draw-decision-frames-only")
    with (out/"run.log").open("a") as log:
        subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)

def main():
    baseline=out/"stock-smoke"
    candidate=out/"smoke"
    if not (out/"verification.json").exists():
        for treatment,folder in (("stock",baseline),("reduced",candidate)):
            if not (folder/"stream.json").exists():
                with (out/"run.log").open("a") as log:
                    subprocess.run([sys.executable,"-u",str(base/"work/trace-emerald-frontier-stream.py"),str(folder),treatment],
                                   stdout=log,stderr=subprocess.STDOUT,check=True)
        stream=read(baseline/"stream.json")
        assert read(candidate/"stream.json")==stream and stream["restored_screen_count"]>1
        reference=read(baseline/"completed.json")
        actual=read(candidate/"completed.json")
        assert actual["best"]==reference["best"]
        for k,v in reference["status"].items():
            if k!="steps_per_second":
                assert actual["status"][k]==v,k
        checked=[]
        ordinary=sorted((baseline/"discoveries").glob("*/game.json"))
        rendered=sorted((candidate/"discoveries").glob("*/game.json"))
        assert len(ordinary)==len(rendered)>0
        for stock,fast in zip(ordinary,rendered):
            assert stock.parent.name==fast.parent.name
            audit=read(fast)
            assert audit.pop("frameskip_option_queries")>0 and audit==read(stock)
            assert fast.with_suffix(".bin").read_bytes()==stock.with_suffix(".bin").read_bytes()
            checked.append(stock.parent.name)
        write_json(out/"verification.json",dict(collected_buttons=512,training_forks=actual["status"]["branches"],
                   matched_tapes_and_audits=checked,observation_action_reward_and_restore_stream=stream,all_policy_choices_identical=True))
    first=root/"controls/last-frame-render/comparison-36058.json"
    other=root/"controls/last-frame-render/24479561ff0b/comparison-16573.json"
    control=family/"learned-visual-motion-landmarks-64-uniform-proposal/completed.json"
    plan=read(registration/"plan.json")
    write_json(out/"second-seed-plan.json",dict(seed=173742,real_buttons=plan["steps"],source=plan["selected"]["tape_sha256"],
               first_seed=173741,hardware_frames_per_button=24,entry="two equivalent full training replays and a completed uniform-proposal comparison"))
    while not (other.exists() and control.exists()):
        time.sleep(30)
    reduced=True
    for sample in (first,other):
        measured=read(sample)
        if not measured["all_images_states_and_rewards_match"] or measured["cpu_speedup"]<=1:
            reduced=False
            write_json(out/"render-decision.json",dict(status="kept stock rendering",failed_check=str(sample)))
            break
    while int(next(s.split()[1] for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:")))<1650000:
        time.sleep(20)
    dest=family/"systematic-reward-frontiers-seed-173742"
    renderer="reduced" if reduced else "stock"
    write_json(out/"queue.json",dict(time=time.time(),experiment=str(dest),status="starting second exploration seed",renderer=renderer))
    train(Path(plan["source"]),registration/"selected-record.json",dest,173742,plan["steps"],plan["capacity"],reduced)
    write_json(out/"result.json",dict(status="second exploration seed complete",renderer=renderer,training=read(dest/"completed.json")))

if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),
        EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
        PYTHONPATH=str(project))
    out.mkdir(parents=True,exist_ok=True)
    with (out/"job.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
