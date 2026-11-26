"""Try all controller buttons at equally spaced ends of autonomous training branches."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from earned_starts import EarnedStarts, verified_reset_lineage
from reward_branch_env import ForkEnv
from reward_branch_search import visual_code, write_json
from screen_explorer import ScreenExplorer


def read(path):
    return json.loads(path.read_text())


def select(arms, threshold):
    histories={}
    steps=0
    origins=[]
    for arm in arms:
        config=read(arm/"settings.json")
        origin=read(arm/"branch-origin.json")
        origins.append(origin)
        charged=0
        for path in sorted(arm.glob("discoveries/work-*/record.json")):
            row=read(path)
            tape=path.with_name("game.bin").read_bytes()
            if (not row["training_only"] or hashlib.sha256(tape).hexdigest()!=row["tape_sha256"]
                    or not verified_reset_lineage(arm,config,tape)):
                raise ValueError("Invalid self-earned source history")
            histories.setdefault(row["tape_sha256"],(row,tape))
            charged=row["work"]-origin["inherited_training_buttons"]
            if charged>=threshold:
                break
        if charged<threshold:
            raise ValueError("Every source arm must reach the same registered budget")
        steps+=charged
    if any(o["parent_record"]!=origins[0]["parent_record"] or o["source_run"]!=origins[0]["source_run"] for o in origins):
        raise ValueError("Source arms must have the same training parent")
    return histories,steps,origins[0]


def run(arms, bankdir, output, threshold, offsets, lengths, seed):
    if min(threshold,*lengths)<1 or min(offsets)<0 or len(set(offsets))!=len(offsets) or len(set(arms))!=len(arms):
        raise ValueError("Invalid endpoint sampling schedule")
    bank=EarnedStarts(bankdir)
    histories,charged,origin=select(arms,threshold)
    parent=origin["parent_record"]
    if parent["tape_sha256"]!=bank.info["source_sha256"]:
        raise ValueError("Source bank and histories disagree")
    output.mkdir(parents=True,exist_ok=True)
    settings=dict(training_forks=True,inputs="RGB, scalar reward, termination",seed=seed,frame_repeat=24,
        receives_saves_from_other_runs=True,uses_verified_own_training_snapshots=True,external_playthrough_or_human_save=False,
        source_sha256=parent["tape_sha256"],bank_snapshot_sha256=[s["sha256"] for s in bank.records],
        source_arms=[str(arm.resolve()) for arm in arms],inherited_new_work=charged,source_threshold=threshold,
        source_histories=sorted(histories),offsets=offsets,round_lengths=lengths,branch_count_scope="this endpoint experiment",
        selection="every sealed endpoint through the first matched checkpoint, regardless of return, map or content",
        buttons="all nine buttons at every distinct prefix followed by three waits, then empirical pixel-model exploration")
    if (output/"settings.json").exists():
        if read(output/"settings.json")!=settings:
            raise ValueError("Different experiment occupies the output")
        if (output/"completed.json").exists():
            return
        raise FileExistsError("Preserve interrupted branches in their original directory and use a new output")
    write_json(output/"settings.json",settings)
    write_json(output/"branch-origin.json",dict(origin,seed=seed,inherited_training_buttons=parent["work"]+charged,
               source_endpoint_search_buttons=charged,source_endpoint_arms=settings["source_arms"]))
    write_json(output/"earned-start-provenance.json",bank.info)
    rng=np.random.default_rng(seed)
    policy=ScreenExplorer(np.random.default_rng([seed,1]))
    ordered=list(histories)
    rng.shuffle(ordered)
    env=ForkEnv(output/"first-maps",shared_visit_reward=True,trainer_audits=output/"first-trainers",earned_starts=bankdir)
    anchors={}
    verified=set()
    work=rebuilt=branches=0
    best=None
    try:
        with tempfile.TemporaryDirectory() as temp:
            for horizon in lengths:
                for offset in offsets:
                    visited=set()
                    for sha in ordered:
                        record,tape=histories[sha]
                        n=len(tape)-offset
                        start=max(((r["prefix"],i) for i,r in enumerate(bank.records)
                                   if r["prefix"]<=n and tape.startswith(bank.tape[:r["prefix"]])),default=None)
                        if start is None:
                            continue
                        key=hashlib.sha256(tape[:n]).hexdigest()
                        if key in visited:
                            continue
                        visited.add(key)
                        if key not in anchors:
                            prefix,index=start
                            image=env.command("earned_start",index)
                            for button in tape[prefix:n]:
                                image,_,done,_=env.command("step",(button,1))
                                rebuilt+=1
                                if done:
                                    raise ValueError("Terminal prefix cannot be extended")
                            if offset==0:
                                env.command("audit",temp)
                                audit=read(Path(temp)/"game.json")
                                if (visual_code(image).hex()!=record["screen"] or audit["return_cents"]!=record["return_cents"]
                                        or (Path(temp)/"game.bin").read_bytes()!=tape):
                                    raise ValueError("Original saved branch ending failed verification")
                                verified.add(sha)
                            slot=len(anchors)
                            env.command("save",slot)
                            anchors[key]=slot
                        if sha not in verified:
                            raise ValueError("Verify each endpoint before exploring its earlier positions")
                        buttons=rng.permutation(9)
                        for first in buttons:
                            image=env.command("restore",anchors[key])
                            for i in range(horizon):
                                state=visual_code(image)
                                if i<4:
                                    button=int(first) if i==0 else 0
                                    policy.visits[state]+=1
                                    policy.transitions.setdefault(state,{})
                                else:
                                    button=policy.choose(state)
                                image,cents,done,used=env.command("step",(button,1))
                                assert used==1
                                policy.observe(state,button,visual_code(image),cents/100)
                                work+=1
                                if done:
                                    break
                            branches+=1
                            cumulative=parent["work"]+charged+work
                            relative=f"discoveries/work-{cumulative:010d}"
                            target=output/relative
                            env.command("audit",str(target))
                            audit=read(target/"game.json")
                            row=dict(work=cumulative,branches=branches,return_cents=audit["return_cents"],length=audit["steps"],
                                     distinct_cells_across_training=audit["distinct_cells_across_training"],audit=relative,
                                     screen=visual_code(image).hex(),tape_sha256=hashlib.sha256((target/"game.bin").read_bytes()).hexdigest(),
                                     terminal=bool(done),training_only=True,source_history=sha,source_prefix_length=n,first_button=int(first))
                            write_json(target/"record.json",row)
                            if best is None or row["return_cents"]>best["return_cents"] or done:
                                best=row
                                write_json(output/"best.json",row)
                            status=dict(new_buttons=work,reconstruction_decisions=rebuilt,branches=branches,
                                        verified_endpoints=len(verified),anchors=len(anchors),round_horizon=horizon,best_return_cents=best["return_cents"])
                            write_json(output/"status.json",status)
                            if done:
                                write_json(output/"completed.json",dict(best=best,status=status))
                                return
            write_json(output/"completed.json",dict(best=best,status=status))
    finally:
        env.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",type=Path,nargs="+",required=True)
    p.add_argument("--bank",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--threshold",type=int,default=16384)
    p.add_argument("--offsets",type=int,nargs="+",default=[0,8,16,32])
    p.add_argument("--lengths",type=int,nargs="+",default=[20,512])
    p.add_argument("--seed",type=int,default=173751)
    args=p.parse_args()
    if args.offsets[0]!=0:
        p.error("First offset must be zero to verify the source endpoints")
    run(args.arm,args.bank,args.output,args.threshold,args.offsets,args.lengths,args.seed)
