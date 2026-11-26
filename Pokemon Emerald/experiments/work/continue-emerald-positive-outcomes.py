"""Continue each distinct self-earned positive training outcome with all nine initial buttons."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from earned_starts import EarnedStarts,verified_reset_lineage
from reward_branch_env import ForkEnv
from reward_branch_search import visual_code,write_json
from screen_explorer import ScreenExplorer

root=project/"runs/pixels-only-01/official-wsl"
bankdir=root/"controls/dreamer-earned-resets/starts"
losses=root/"controls/all-losses-practice"
family=root/"paired-self-earned-frontiers"
sources=family/"every-sealed-endpoint-01"
out=family/"continue-earned-wins"
read=lambda p:json.loads(p.read_text())


def prefix_length(tape,expected,maximum):
    low=max(0,len(tape)-maximum)
    hasher=hashlib.sha256(tape[:low])
    for n in range(low,len(tape)+1):
        if hasher.hexdigest()==expected:
            return n
        hasher.update(tape[n:n+1])
    raise ValueError("Practice history does not extend its declared training start")


def expand(env,bank,cohort,file,model,rng,work,branches,best,run):
    original=read(file)
    tape=file.with_suffix(".bin").read_bytes()
    assert original["training_only"] and original["action_source"]=="training" and original["scalar_outcome"]==100
    anchor=prefix_length(tape,original["source_prefix"],768)
    if not any(t.startswith(tape[:anchor]) for t in cohort) or len(tape)!=original["steps"]:
        raise ValueError("Practice start was not earned within the registered cohort")
    prefix,index=max((s["prefix"],i) for i,s in enumerate(bank.records) if s["prefix"]<=anchor and tape.startswith(bank.tape[:s["prefix"]]))
    image=env.command("earned_start",index)
    for button in tape[prefix:]:
        image,_,finished,used=env.command("step",(button,1))
        assert not finished and used==1
    with tempfile.TemporaryDirectory() as tmp:
        env.command("audit",tmp)
        checked=read(Path(tmp)/"game.json")
        assert checked["events"]==original["events"] and checked["return_cents"]==original["return_cents"]
        assert checked["first_visit_cells"]==original["first_visit_cells"] and checked["maps_first_reached"]==original["maps_first_reached"]
        assert (Path(tmp)/"game.bin").read_bytes()==tape
    env.command("save",0)
    start_image=visual_code(image)
    for initial in rng.permutation(9):
        image=env.command("restore",0)
        assert visual_code(image)==start_image
        remaining=0
        branches+=1
        for i in range(512):
            state=visual_code(image)
            if not i:
                button=int(initial)
                remaining=0
                model.visits[state]+=1
                model.transitions.setdefault(state,{})
            elif remaining:
                remaining-=1
                model.visits[state]+=1
                model.transitions.setdefault(state,{})
            else:
                button=model.choose(state)
                remaining=int(rng.choice([0,0,1,3,7]))
            image,cents,done,used=env.command("step",(button,1))
            assert used==1
            model.observe(state,button,visual_code(image),cents/100)
            work+=1
            if done:
                break
        relative=f"discoveries/work-{work:010d}"
        dest=run/relative
        env.command("audit",str(dest))
        a=read(dest/"game.json")
        row=dict(work=work,branches=branches,return_cents=a["return_cents"],length=a["steps"],screen=visual_code(image).hex(),
                 audit=relative,tape_sha256=hashlib.sha256((dest/"game.bin").read_bytes()).hexdigest(),training_only=True,
                 terminal=bool(done),distinct_cells_across_training=a["distinct_cells_across_training"],
                 source_win_tape=hashlib.sha256(tape).hexdigest(),source_training_prefix=original["source_prefix"],first_button=int(initial))
        write_json(dest/"record.json",row)
        if best is None or row["return_cents"]>best["return_cents"] or done:
            best=row
            write_json(run/"best.json",row)
        write_json(run/"status.json",dict(new_buttons=work,branches=branches,best_return_cents=best["return_cents"]))
        if done:
            break
    return work,branches,best


def main(winning_run, out):
    out.mkdir(parents=True,exist_ok=True)
    bank=EarnedStarts(bankdir)
    source_settings=read(sources/"settings.json")
    origin=read(sources/"branch-origin.json")
    cfg=dict(seed=173761,training_forks=True,receives_saves_from_other_runs=True,uses_verified_own_training_snapshots=True,
             external_playthrough_or_human_save=False,source_sha256=bank.info["source_sha256"],
             bank_snapshot_sha256=[r["sha256"] for r in bank.records],frame_repeat=24,
             policy_inputs="current RGB and scalar reward",selection="first training win from each distinct loss-practice start, irrespective of opponent",
             action_schedule="all nine first buttons followed by 511 single 24-frame decisions from an empirical pixel transition model",
             work_counter="new forward decisions only; reconstruction and prior training reported separately")
    write_json(out/"settings.json",cfg)
    write_json(out/"branch-origin.json",dict(origin,seed=cfg["seed"]))
    write_json(out/"earned-start-provenance.json",bank.info)
    ids=set(read(losses/"cohort.json")["histories"])
    cohort={}
    for p in sources.glob("discoveries/work-*/record.json"):
        r=read(p)
        if r["tape_sha256"] not in ids:
            continue
        t=p.with_name("game.bin").read_bytes()
        assert hashlib.sha256(t).hexdigest()==r["tape_sha256"] and verified_reset_lineage(sources,source_settings,t)
        cohort[r["tape_sha256"]]=t
    assert set(cohort)==ids
    if (out/"status.json").exists():
        raise FileExistsError("Keep the existing forward study; use a new output for another trial")
    env=ForkEnv(out/"first-maps",shared_visit_reward=True,trainer_audits=out/"first-trainers",earned_starts=bankdir)
    seen=set()
    model=ScreenExplorer(np.random.default_rng(cfg["seed"]))
    rng=np.random.default_rng([cfg["seed"],1])
    work=branches=0
    best=None
    try:
        while True:
            files=sorted((winning_run/"training").glob("*/game.json"),key=lambda p:p.stat().st_mtime_ns)
            for f in files:
                a=read(f)
                if a["scalar_outcome"]!=100 or a["source_prefix"] in seen:
                    continue
                work,branches,best=expand(env,bank,cohort.values(),f,model,rng,work,branches,best,out)
                seen.add(a["source_prefix"])
                write_json(out/"completed-starts.json",dict(count=len(seen),prefixes=sorted(seen)))
                if best["terminal"]:
                    return
            if (winning_run/"completed.json").exists():
                write_json(out/"completed.json",dict(extended_positive_training_starts=len(seen),new_buttons=work,best=best))
                return
            time.sleep(20)
    finally:
        env.close()


if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--winning-run", type=Path, default=losses)
    parser.add_argument("--output", type=Path, default=out)
    args=parser.parse_args()
    main(args.winning_run,args.output)
