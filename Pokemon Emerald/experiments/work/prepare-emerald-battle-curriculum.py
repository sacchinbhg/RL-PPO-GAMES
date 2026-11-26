"""Construct intervention states from prior generic outcomes on one self-generated training game."""
from collections import deque
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from battle_feedback import BattleFeedback
from earned_starts import EarnedStarts,checksum,observation_hash,restore_score,score_record
from emerald import Emerald,ROM_SHA1,REPEAT
from pixel_env import Score,game_paths
from train_earned_dreamer import write_json

root=project/"runs/pixels-only-01/official-wsl"
out=root/"outcome-curriculum"
bankdir=out/"starts"
read=lambda path:json.loads(path.read_text())


def source():
    spec=importlib.util.spec_from_file_location("practice",base/"work/learn-emerald-battle-options.py")
    practice=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(practice)
    cohort=practice.training_tapes()
    folder=root/"controls/all-losses-practice/training"
    for p in sorted(folder.glob("*/game.json"),key=lambda p:(p.stat().st_mtime_ns,p.name)):
        record=read(p)
        if record["scalar_outcome"]<0:
            tape=p.with_suffix(".bin").read_bytes()
            anchor=practice.prefix_length(tape,record["source_prefix"],768)
            if (record["action_source"]!="training" or not record["training_only"] or len(tape)!=record["steps"]
                    or not any(history.startswith(tape[:anchor]) for history in cohort)):
                raise ValueError("Practice game must extend its own training cohort")
            return p,record,tape
    raise ValueError("No training loss has been recorded")


def prepare():
    bankdir.mkdir(parents=True,exist_ok=True)
    if (bankdir/"ready.json").exists():
        EarnedStarts(bankdir)
        return
    path,record,tape=source()
    sha=checksum(tape)
    write_json(out/"selection.json",dict(training_game=str(path),sha256=sha,buttons=len(tape),
               rule="first completed negative outcome in the registered training retry arm, irrespective of opponent or map",
               offsets_before_resolved_reward=[64,192,384],separate_uniform_samples_per_outcome=8,
               source_prefix=record["source_prefix"],no_evaluation_episodes_used=True))
    core,rom=game_paths()
    game=Emerald(core,rom)
    score=Score(True)
    signal=BattleFeedback()
    window=deque(maxlen=15)
    selected={1:{},-1:{}}
    roots={}
    pulses=[]
    raw=0
    try:
        game.load(game.boot)
        for n,b in enumerate(tape,1):
            image,key=game.step(b)
            rew,done=score.read(game,key)
            raw+=round(100*rew)
            previous=(signal.wins,signal.losses)
            signal.read(game,key)
            if window and len(window[-1]["follow"])<32:
                window[-1]["follow"].append(observation_hash(image,key,rew,done))
            if n%32==0:
                window.append(dict(n=n,blob=game.save(),image=image.copy(),score=score_record(score),descriptor=list(key),raw=raw,follow=[]))
            if (signal.wins,signal.losses)!=previous:
                sign=1 if signal.wins>previous[0] else -1
                pulses.append((n,sign))
                for offset in (64,192,384):
                    latest=next((node for node in reversed(window) if node["n"]<=n-offset),None)
                    if latest is not None:
                        selected[sign][latest["n"]]=latest
            if n%2048==0:
                write_json(out/"cold-progress.json",dict(decisions=n,expected=len(tape),feedback=signal.audit()))
        if (score.events!=record["events"] or score.first_maps!=record["maps_first_reached"]
                or len(score.cells)!=record["first_visit_cells"] or raw!=record["return_cents"]
                or not pulses or pulses[-1]!=(len(tape),-1)):
            raise ValueError("Independent original-game replay disagreed with the training loss")
        for sign,group in selected.items():
            keys=sorted(group)
            if not keys:
                raise ValueError("Both positive and negative examples are required for this paired test")
            for i in np.linspace(0,len(keys)-1,min(8,len(keys)),dtype=int):
                n=keys[i]
                roots[n]=group[n]
        write_json(out/"cold-replay.json",dict(sha256=sha,rom_sha1=ROM_SHA1,in_game_restores=0,decisions=len(tape),
                   matched_story_rewards_and_maps=True,outcomes=pulses,feedback=signal.audit(),
                   candidate_starts={k:sorted(v) for k,v in selected.items()}))
        entries=[]
        for n,node in sorted(roots.items()):
            name=f"prefix-{n:08d}.npz"
            file=bankdir/name
            np.savez(file,state=np.frombuffer(node["blob"],np.uint8),image=node["image"],
                     score=np.frombuffer(json.dumps(node["score"]).encode(),np.uint8))
            game.load(node["blob"])
            check=restore_score(node["score"])
            assert len(node["follow"])==32 and list(game.descriptor())==node["descriptor"]
            for expected,b in zip(node["follow"],tape[n:n+32],strict=True):
                im,key=game.step(b)
                rew,done=check.read(game,key)
                assert observation_hash(im,key,rew,done)==expected
            entries.append(dict(prefix=n,filename=name,sha256=checksum(file.read_bytes()),descriptor=node["descriptor"],
                                scalar_return_cents=node["raw"]))
        (bankdir/"source.bin").write_bytes(tape)
        write_json(bankdir/"ready.json",dict(source_sha256=sha,source_buttons=len(tape),source_return_cents=raw,
                   core_sha256=checksum(core.read_bytes()),rom_sha1=ROM_SHA1,frames_per_decision=REPEAT,training_only=True,
                   all_starts_verified=True,starts=entries,real_successors_compared=32*len(entries)))
        EarnedStarts(bankdir)
    finally:
        game.close()


if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
    prepare()
