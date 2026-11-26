"""Measure fixed palette conversion on a shared input history, then compare frozen memory in two native games."""
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

work=Path(__file__).resolve().parent
project=work.parent/"outputs/EmeraldRL"
root=project/"runs/pixels-only-01/official-wsl"
sys.path.insert(0,str(project))
from earned_memory import Memory
from native_emerald import NativeEmerald
from train_earned_dreamer import Trajectory,write_json
spec=importlib.util.spec_from_file_location("native_training",work/"explore-native-earned-screens.py")
native=importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
out=root/"controls/native-visual-transfer"


def palette():
    source=next((root/"paired-self-earned-frontiers/visual-slots-0/verified-training-tapes").glob("e30822*"))
    training=Trajectory(Path("//?/"+source.as_posix()))
    cfg=native.read(project/"local.json")
    game=NativeEmerald(cfg["native_core"],cfg["rom"])
    frame=game.boot_image
    errors=[[],[],[]]
    exact=[0,0,0]
    try:
        for i,(observed,b,_,_,_,_) in enumerate(training.episode()):
            if i%32==0:
                for j,variant in enumerate((frame,frame & np.array([248,248,248],np.uint8),frame & np.array([248,252,248],np.uint8))):
                    exact[j]+=int(np.array_equal(variant,observed))
                    errors[j].append(float(np.abs(variant.astype(np.int16)-observed).mean()))
            if i==8192:
                break
            frame,_=game.step(b)
    finally:
        game.close()
    return dict(original_tape=training.source["source_sha256"],sampled=len(errors[0]),interval=32,
        variants=["unmodified","5-5-5","5-6-5"],mean_pixel_errors=[float(np.mean(r)) for r in errors],exact=exact)


def play(name,mask,seed=197461,uniform=False,allow_approximate=True):
    folder=out/name
    folder.mkdir(exist_ok=True)
    model=root/"earned-screen-memory/model"
    policy=Memory(model,seed,allow_approximate)
    rng=np.random.default_rng(seed)
    env=native.NativeGame(folder,training=False)
    image=env.initial
    incoming=0
    start=time.monotonic()
    try:
        for i in range(1,65537):
            observed=image & mask if mask is not None else image
            if uniform:
                button=int(rng.integers(9))
            else:
                button=policy.choose(observed,incoming/100)
                policy.executed(button)
            image,incoming,done=env.step(button)
            if i%16384==0:
                write_json(folder/"status.json",dict(buttons=i,events=env.events,known_screens=0 if uniform else i-policy.usage["unknown_screen"]))
            if done:
                break
        audit=env.record(folder/"audit")
        write_json(folder/"result.json",dict(**audit,controller="uniform" if uniform else policy.identity["model_sha256"],
            button_sources=dict(policy.usage),seconds=time.monotonic()-start,seed=seed,game_restores=0,optimizer_updates=0,
            approximate_matching=allow_approximate))
        for marker in (folder/"first-maps").glob("*/game.json"):
            saved=native.read(marker)
            if saved["training_only"]:
                saved["training_only"]=False
                write_json(marker,saved)
    finally:
        env.game.close()


if __name__=="__main__":
    out.mkdir(exist_ok=True)
    stats=palette()
    write_json(out/"palette.json",stats)
    quant=np.array([248,248,248] if stats["exact"][1]>=stats["exact"][2] else [248,252,248],np.uint8)
    write_json(out/"four-game-plan.json",dict(seeds=list(range(197461,197465)),buttons_per_game=65536,
        primary="starter and later native maps; corroborate any claimed original-game skill separately",
        arms=["unmodified RGB memory","quantized RGB memory","uniform random buttons"],deliberate_control="uniform",
        earlier_observation="the first raw pilot had already finished when the replicated controls were registered"))
    for seed in range(197461,197465):
        for label,mask in (("raw",None),("quantized",quant),("uniform",None)):
            name=label if seed==197461 else f"{label}-{seed}"
            if not (out/name/"result.json").exists():
                play(name,mask,seed,label=="uniform")
