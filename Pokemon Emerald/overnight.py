"""Run and evaluate a time-budgeted Emerald world-policy experiment on this PC."""

import argparse
import ctypes
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import os
import shutil
import subprocess
import time


PROJECT = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def put(path, data):
    tmp = path.with_name(path.name + ".new")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def save(path, data):
    put(path, json.dumps(data, indent=2) + "\n")


def latest_row(path):
    if not path.exists():
        return {}
    with path.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 16384))
        for line in reversed(f.read().splitlines()):
            try:
                return json.loads(line)
            except (ValueError, UnicodeError):
                continue
    return {}


def launch_command(command, log):
    with log.open("ab") as stream:
        return subprocess.Popen(command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def trial_stats(report):
    rows = report["trials"]
    return dict(games=len(rows), clock=sum(r["clock"] for r in rows),
                early_clock=sum(r["first_milestone_decision"].get("clock", 10**12) <= 2048 for r in rows),
                starter=sum(r["starter"] for r in rows), max_badges=max(r["badges"] for r in rows),
                hall=sum(r["hall_of_fame"] for r in rows), maps=sorted({m for r in rows for m in r["maps"]}))


def publish(out, state):
    row=state.get("latest", {})
    baseline=state["starting_steps"]
    elapsed=state.get("elapsed_seconds",0)
    train=state.get("checkpoint_summary",{})
    entries=state["evaluations"]
    lines=["# Emerald overnight run", "", f"**{state['phase']}** · Updated {state['updated_utc']} (UTC).", "",
           f"Training started from the full-gradient world policy at {baseline:,} decisions. "
           f"The budget is {state['budget_seconds']/3600:g} additional hours on the RTX 4060 Laptop GPU.",
           f"Expected training finish on this PC: {state.get('expected_finish_local','waiting for the first update')}. Final gameplay evaluation follows.",
           "", f"Training time credited so far: {elapsed/3600:.3f} hours. "
           f"Latest real decisions: {row.get('total_training_steps',baseline):,} total, "
           f"{row.get('total_training_steps',baseline)-baseline:,} since the starting checkpoint.",
           "", f"Latest training checkpoint observed {len(train.get('maps',[]))} maps in its current attempt. "
           f"Clock: {train.get('clock','pending')}. Starter: {train.get('starter','pending')}. "
           f"Most badges: {train.get('badges','pending')}. Hall of Fame: {train.get('hall_of_fame','pending')}. "
           "Training can restore archive states.", "", "## Frozen policies playing from boot", "",
           "| Checkpoint | Real decisions | Games | Clock by 2,048 | Clock by cutoff | Starter | Most badges | Hall of Fame |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for e in entries:
        s=e["scores"]
        lines.append(f"| [{e['name']}]({e['report']}) | {e['steps']:,} | {s['games']} | {s['early_clock']} | {s['clock']} | {s['starter']} | {s['max_badges']} | {s['hall']} |")
    lines+=["", "Baseline and hourly evaluations allow 8,192 decisions per game. The final evaluation allows 32,768. "
            "Each trial starts at boot with the policy selecting every button. [YouTube notes](YOUTUBE_NOTES.md) · [Machine-readable status](status.json)"]
    if state.get("running_eval"):
        lines+=["",f"Evaluation in progress: {state['running_eval']}."]
    if state.get("error"):
        lines+=["",f"Run issue: {state['error']}"]
    notes=["# Overnight YouTube notes", "", "## During training", "",
           "The starting archive already contained the bedroom clock and both Littleroot houses. "
           "Each new training discovery below has the native frame and the button history from boot. Archive restores are enabled during training.", ""]
    for event in state["discoveries"]:
        notes.append(f"- At {event['steps']:,} total real decisions, training first recorded **{event['name']}**. "
                     f"[Frame]({event['directory']}/frame.png) · [Recorded buttons]({event['directory']}/buttons.bin) · [Event]({event['directory']}/event.json)")
    notes+=["", "## Independent gameplay", ""]
    for e in entries:
        s=e["scores"]
        folder=Path(e["report"]).parent.as_posix()
        notes.append(f"- **{e['name']}** at {e['steps']:,} training decisions: {s['clock']}/{s['games']} clocks and "
                     f"{s['starter']}/{s['games']} starters, up to {s['max_badges']} badges. "
                     f"Maps observed: {', '.join(s['maps']) or 'none'}. [Native frames and button history]({folder}/moments/HIGHLIGHTS.md)")
    save(out/"status.json",state)
    put(out/"PROGRESS.md","\n".join(lines)+"\n")
    put(out/"YOUTUBE_NOTES.md","\n".join(notes)+"\n")


def supervise(args):
    cfg=read(PROJECT/"local.json")
    out=args.out.resolve()
    parent=args.restart_from.resolve()
    parent_info=read(parent.parent/"training.json")
    if parent_info.get("imagination_update")!="full-bptt" or not parent.is_file():
        raise ValueError("Choose a verified full-bptt world checkpoint")
    out.mkdir(parents=True,exist_ok=True)
    with parent.open("rb") as source, (out/"run.json").open("x",encoding="utf-8") as f:
        json.dump(dict(started_utc=now(), budget_seconds=args.hours*3600, seed=args.seed,
                       parent=str(parent), parent_sha256=hashlib.file_digest(source,"sha256").hexdigest(),
                       training_envs=8, training_episode=512, horizon=64, hourly_seconds=3600,
                       evaluation_episodes=4, evaluation_budget=8192, final_episodes=8, final_budget=32768),f,indent=2)
    base=parent_info["total_training_steps"]
    initial=out/"baseline"
    initial.mkdir()
    shutil.copy2(parent.parent/"policy.pt",initial/"policy.pt")
    command=[cfg["python"],"-u",str(PROJECT/"experiment.py")]
    backend=parent_info.get("backend","mgba")
    core=cfg["native_core" if backend=="native" else "core"]
    common=["--core",core,"--backend",backend,"--rom",cfg["rom"],"--mode","world","--device","cuda"]
    state=dict(phase="Starting training",updated_utc=now(),budget_seconds=args.hours*3600,
               starting_steps=base,backend=backend,supervisor_pid=os.getpid(),discoveries=[],evaluations=[],attempts=[])
    if args.mirror:
        state["primary_run"]=str(out)
    mirror=args.mirror.resolve() if args.mirror else None
    mirrored=set()
    def refresh():
        state["updated_utc"]=now()
        publish(out,state)
        if mirror:
            try:
                mirror.mkdir(parents=True,exist_ok=True)
                folders=[Path(e["report"]).parent for e in state["evaluations"]]
                folders.extend(Path(e["directory"]) for e in state["discoveries"])
                for relative in folders:
                    if relative not in mirrored:
                        shutil.copytree(out/relative,mirror/relative,dirs_exist_ok=True)
                        mirrored.add(relative)
                for filename in ("status.json","PROGRESS.md","YOUTUBE_NOTES.md"):
                    shutil.copy2(out/filename,mirror/filename)
                state.pop("desktop_sync_error",None)
            except OSError as exc:
                state["desktop_sync_error"]=str(exc)
    queued=[("baseline",base,initial/"policy.pt",False)]
    seen=set()
    known=set()
    train_process=eval_process=None
    eval_job=None
    finished=False
    credit=0.
    attempt=0
    def start_train():
        nonlocal attempt,train_process,folder
        attempt+=1
        folder=out/f"train-{attempt:02}"
        folder.mkdir()
        budget=max(.001,state["budget_seconds"]-credit)
        train_process=launch_command(command+["train",*common,"--out",str(folder),
            "--restart-from",str(parent),"--imagination-update","full-bptt","--seed",str(args.seed+attempt-1),
            "--envs","8","--episode","512","--horizon","64","--train-seconds",str(budget),
            "--checkpoint-seconds","600","--snapshot-seconds","3600"],folder/"console.log")
        state["attempts"].append(dict(directory=folder.name,started_utc=now(),pid=train_process.pid,credited_before=credit))
        state["phase"]="Training"
        state["trainer_pid"]=train_process.pid
    def start_eval(name,steps,weights,deep,retry=0):
        nonlocal eval_process,eval_job
        dest=out/"evaluations"/(name if retry==0 else name+"-retry")
        dest.mkdir(parents=True,exist_ok=True)
        eval_process=launch_command(command+["eval",*common,"--out",str(dest),"--weights",str(weights),
            "--seed",str(246821 if deep else 245811),"--envs",str(8 if deep else 4),
            "--episode",str(32768 if deep else 8192),"--capture-moments"],dest/"console.log")
        eval_job=(name,steps,weights,deep,retry,dest)
        state["running_eval"]=name
    folder=None
    kernel=ctypes.windll.kernel32 if os.name=="nt" else None
    try:
        if kernel:
            kernel.SetThreadExecutionState(0x80000001)
        start_train()
        refresh()
        while True:
            cp=folder/"training.json"
            if not finished:
                row=latest_row(folder/"metrics.jsonl")
                if row:
                    state["latest"]=row
                    state["elapsed_seconds"]=credit+row["training_elapsed_seconds"]
                    start=datetime.fromisoformat(row["training_started_utc"])
                    state["expected_finish_local"]=(start+timedelta(seconds=state["budget_seconds"]-credit)).astimezone().isoformat(timespec="seconds")
                if cp.exists():
                    info=read(cp)
                    state["checkpoint_summary"]={k:info[k] for k in ("maps","clock","starter","badges","hall_of_fame","total_training_steps","archive_cells")}
                for path in sorted((folder/"discoveries").glob("*/event.json")):
                    if path.parent.name in known:
                        continue
                    event=read(path)
                    state["discoveries"].append(dict(name=event["event"],steps=event["total_training_steps"],
                        directory=path.parent.relative_to(out).as_posix()))
                    known.add(path.parent.name)
                for path in sorted((folder/"snapshots").glob("*/snapshot.json")):
                    if path in seen:
                        continue
                    s=read(path)
                    seen.add(path)
                    if not s["finished"]:
                        queued.append((f"step-{s['total_training_steps']}",s["total_training_steps"],path.parent/"policy.pt",False))
                code=train_process.poll()
                if code is not None:
                    info=read(cp) if cp.exists() else {}
                    if info.get("finished") and code==0:
                        finished=True
                        state["phase"]="Training finished; evaluating final policy"
                        state["training_finished_utc"]=now()
                        state["elapsed_seconds"]=credit+info["training_elapsed_seconds"]
                        final_steps=info["total_training_steps"]
                        final_weights=folder/"snapshots"/f"{final_steps:010d}"/"policy.pt"
                        queued.append(("final",final_steps,final_weights,True))
                        state["final_checkpoint"]=str(folder/"checkpoint.pt")
                    elif attempt<4:
                        state["attempts"][-1]["exit_code"]=code
                        if info and (folder/"checkpoint.pt").exists():
                            credit+=info["training_elapsed_seconds"]
                            parent=folder/"checkpoint.pt"
                        start_train()
                    else:
                        raise RuntimeError(f"Training exited {code} in {folder}; see console.log")
            if eval_process and eval_process.poll() is not None:
                name,steps,weights,deep,retry,dest=eval_job
                result=dest/"evaluation.json"
                if eval_process.returncode==0 and result.exists():
                    report=read(result)
                    if report["resets"]!=0 or len(report["trials"])!=(8 if deep else 4):
                        raise ValueError(f"Evaluation validation failed at {dest}")
                    state["evaluations"].append(dict(name=name,steps=steps,report=result.relative_to(out).as_posix(),
                        scores=trial_stats(report),policy_sha256=report["policy_sha256"]))
                    eval_process=None
                    state.pop("running_eval",None)
                elif retry==0:
                    start_eval(name,steps,weights,deep,1)
                else:
                    state.setdefault("evaluation_errors",[]).append(str(dest))
                    state.pop("running_eval",None)
                    eval_process=None
            if eval_process is None and queued and state.get("latest"):
                start_eval(*queued.pop(0))
            if finished and eval_process is None and not queued:
                if not any(e["name"]=="final" for e in state["evaluations"]):
                    raise RuntimeError("Final evaluation failed; see evaluation logs")
                state["phase"]="Completed" if not state.get("evaluation_errors") else "Completed with missing interim evaluations"
                state["finished_utc"]=now()
                refresh()
                break
            refresh()
            time.sleep(15)
    except BaseException as exc:
        state.update(phase="Interrupted",error=str(exc))
        refresh()
        raise
    finally:
        if kernel:
            kernel.SetThreadExecutionState(0x80000000)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--restart-from",type=Path,required=True)
    parser.add_argument("--hours",type=float,default=12)
    parser.add_argument("--seed",type=int,default=251124)
    parser.add_argument("--mirror",type=Path)
    args=parser.parse_args()
    if not math.isfinite(args.hours) or args.hours <= 0:
        parser.error("Choose a finite, positive training duration")
    supervise(args)
