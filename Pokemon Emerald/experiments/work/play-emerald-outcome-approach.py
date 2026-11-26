"""Learn the approach before the first recorded failure and test two frozen continuations from boot."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from earned_memory import fit_episodes
from earned_starts import EarnedStarts,checksum
from emerald import Emerald
from pixel_env import Score,game_paths
from train_earned_dreamer import Trajectory,write_json

root=project/"runs/pixels-only-01/official-wsl"
out=root/"outcome-curriculum"
read=lambda p:json.loads(p.read_text())


def fit():
    model=out/"approach-model"
    selection=read(out/"selection.json")
    tape=Path(selection["training_game"]).with_suffix(".bin").read_bytes()
    assert checksum(tape)==selection["sha256"]
    cut=(len(tape)-max(selection["offsets_before_resolved_reward"]))//32*32
    bank=EarnedStarts(root/"controls/dreamer-earned-resets/starts")
    best=None
    for meta in (root/"earned-frontier-curriculum/verified-training-tapes").glob("*/manifest.json"):
        trace=Trajectory(meta.parent)
        for i,mark in enumerate(bank.records):
            n=mark["prefix"]
            if n<=cut and tape[:n]==bank.tape[:n] and trace.tape[:n]==tape[:n] and (best is None or n>best[0]):
                best=n,i,trace
    model.mkdir(exist_ok=True)
    write_json(model/"training-origin.json",dict(training_game_sha256=selection["sha256"],used_prefix_sha256=checksum(tape[:cut]),
               prefix_length=cut,shared_verified_images=best[0] if best else 0,
               cutoff_rule="longest registered interval before a generic loss, aligned down to 32 decisions",
               withheld_suffix=len(tape)-cut,policy_uses_game_addresses=False))
    def episode():
        game=Emerald(*game_paths())
        try:
            n=0
            score=Score(True)
            game.load(game.boot)
            image=game.boot_image
            incoming=0.0
            if best:
                n,i,trace=best
                image,score,_=bank.restore(game,i)
                for t,row in enumerate(trace.episode(uncapped=True)):
                    if t==n:
                        assert np.array_equal(row[0],image)
                        incoming=row[2]
                        break
                    assert row[1]==tape[t]
                    yield row
            for t in range(n,cut):
                yield image,tape[t],incoming,t==0,False,False
                image,key=game.step(tape[t])
                incoming,done=score.read(game,key)
                if done:
                    raise ValueError("Source prefix ends before the first negative outcome")
            yield image,0,incoming,False,True,False
            write_json(model/"training-end.json",dict(rgb_sha256=checksum(image.tobytes()),steps=score.decisions,
                       story=score.events,first_visit_cells=len(score.cells)))
        finally:
            game.close()
    fit_episodes([(checksum(tape[:cut]),cut,episode)],model)
    return model,cut


def run():
    model,cut=fit()
    while not (out/"starts/ready.json").exists():
        time.sleep(10)
    bank=EarnedStarts(out/"starts")
    i=next(i for i,r in enumerate(bank.records) if r["prefix"]==cut)
    game=Emerald(*game_paths())
    try:
        image,score,_=bank.restore(game,i)
        check=read(model/"training-end.json")
        assert checksum(image.tobytes())==check["rgb_sha256"] and score.events==check["story"] and len(score.cells)==check["first_visit_cells"]
    finally:
        game.close()
    write_json(model/"verified.json",dict(prefix=cut,confirmed_against_independent_original_ROM_boot=True,restores_in_evaluation=0))
    parent=root/"earned-reset-online-comparison/earned"
    for name,weights,config in (("before-combat-training",parent/"training-result.json",parent/"train"),
                                ("after-combat-training",out/"combat/trained.json",out/"combat/train")):
        while not weights.exists():
            time.sleep(15)
        folder=out/name
        folder.mkdir(exist_ok=True)
        completed=any(folder.glob("fresh-*/result.json"))
        if completed:
            continue
        dest=folder/f"fresh-{len(list(folder.glob('fresh-*')))+1:03d}"
        with (folder/"experiment.log").open("a") as log:
            subprocess.run([sys.executable,"-u",str(project/"extend_earned_route.py"),"--model",str(model),
                            "--checkpoint",read(weights)["checkpoint"],"--parent",str(config),"--upstream","/opt/emerald-rl/source",
                            "--output",str(dest),"--limit","65536","--seed","271353","--patience","512"],stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so",
                      OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",JAX_PLATFORMS="cuda",XLA_PYTHON_CLIENT_PREALLOCATE="false")
    run()
