"""Follow one original fixed-budget frozen game by independently reproducing its growing action tape."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from emerald import Emerald, ROM_SHA1, verify_rom
from pixel_env import Score, first_map_audit, game_paths
from pixel_replay import verify_starter
from reward_branch_search import write_json


def run(folder,index,wait_for=None):
    if wait_for:
        while not wait_for.exists():
            time.sleep(15)
    import fcntl
    worker=f"env-{index}"
    out=folder/"live-map-replay"/worker
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/"follow.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/"result.json").exists():
        return
    identity=json.loads((folder/"provenance.json").read_text())
    assert identity["agent_sha256"] and identity["platform"]=="cuda"
    core,rom=game_paths()
    verify_rom(rom)
    g=Emerald(str(core),str(rom))
    score=Score(identity["uncapped_exploration"])
    history=bytearray()
    starter=None
    source=None
    def status():
        return dict(source_audit=str(source),steps=len(history),events=score.events,
                    maps_first_reached=score.first_maps,first_visit_cells=len(score.cells),starter=starter,
                    policy_weight_sha256=identity["agent_sha256"],rom_sha1=ROM_SHA1,
                    emulator_restores_after_boot=0,uncapped_exploration=identity["uncapped_exploration"])
    try:
        g.load(g.boot)
        while True:
            audits=list((folder/"in-progress").glob(f"*/{worker}/step-*/*episode-00001.json"))
            audits.extend((folder/"external-audit").glob(f"*/{worker}/*episode-00001.json"))
            candidates=[]
            for p in audits:
                try:
                    n=json.loads(p.read_text())["steps"]
                except (FileNotFoundError,json.JSONDecodeError):
                    continue
                if n>len(history):
                    candidates.append((n,p))
            if not candidates:
                if len(history)>=identity["truncation"] or "stone_badge" in score.events:
                    write_json(out/"result.json",dict(**status(),input_sha256=hashlib.sha256(history).hexdigest()))
                    return
                time.sleep(15)
                continue
            length,source=max(candidates)
            audit=json.loads(source.read_text())
            tape=source.with_suffix(".bin").read_bytes()
            if len(tape)!=length or not tape.startswith(history):
                raise ValueError("The official frozen game changed its prior inputs")
            def preserve(directory):
                directory.mkdir(parents=True,exist_ok=True)
                (directory/"game.bin").write_bytes(history)
                write_json(directory/"game.json",status())
            for button in tape[len(history):]:
                pic,key=g.step(button)
                prior_maps=len(score.first_maps)
                reward,terminal=score.read(g,key)
                history.append(button)
                n=len(history)
                if score.events.get("starter")==n:
                    starter=verify_starter(g,n)
                if len(score.first_maps)!=prior_maps:
                    if score.events!={k:v for k,v in audit["events"].items() if v<=n}:
                        raise ValueError("Independent story events differ from official frozen gameplay")
                    first_map_audit(out/"first-maps",next(reversed(score.first_maps)),preserve)
                if n%4096==0 or reward>=1:
                    write_json(out/"status.json",status())
            if score.events!=audit["events"] or len(score.cells)!=audit["first_visit_cells"]:
                raise ValueError("The official and independent game audits diverged")
            write_json(out/"status.json",status())
    finally:
        g.close()
        lock.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("run",type=Path)
    parser.add_argument("index",type=int)
    parser.add_argument("--after",type=Path)
    args=parser.parse_args()
    run(args.run.resolve(),args.index,args.after)
