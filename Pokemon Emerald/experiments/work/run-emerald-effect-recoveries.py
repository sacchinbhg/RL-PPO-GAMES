"""Check the learned button-effect model, then compare frozen controllers in fresh games."""
import json
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from earned_memory import play
from fit_local_motion import hashes
from persistent_effect_policy import EffectPlanner
from train_earned_dreamer import write_json

root=project/"runs/pixels-only-01/official-wsl"
effects=root/"controls/persistent-button-effects"
family=root/"earned-screen-memory"
model=root/"earned-frontier-curriculum/stage-003948-c2da70ebe5f2/model"
read=lambda p:json.loads(p.read_text())
seed=197487


def run():
    planner=EffectPlanner(effects,"local",seed)
    with np.load(effects/"heldout/examples.npz") as examples, np.load(effects/"model.npz") as weights:
        images=examples["rgb"]
        truth=examples["effects"][:,1:].astype(float)
        sensitive=(truth.max(1)!=truth.min(1)) & ~np.isin(hashes(images),hashes(weights["train"]))
        pred=np.stack([planner.predictions(im) for im in images[sensitive]])
        tied=np.isclose(pred,pred.max(1,keepdims=True),atol=1e-7,rtol=0)
        achieved=float(((tied*truth[sensitive]).sum(1)/tied.sum(1)).mean())
    assert abs(achieved-read(effects/"result.json")["reports"]["local_screen"]["effect_probability"])<1e-7
    loop=root/"controls/retrieval-loops"
    image=np.array(Image.open(loop/"select-footage/decision-50000.png"))[::4,::4].copy()
    scores=planner.predictions(image)
    real=read(loop/"heldout-button-effects/result.json")
    assert real["full_rgb_start_matched"] and not real["used_for_training"]
    for i in range(5):
        planner.observe(image)
        choice=planner.select(image,0,True,i+1)
        if i<4:
            assert choice==0
    assert planner.calls==1
    write_json(effects/"inference-verified.json",dict(measured_heldout=achieved,heldout_sensitive=int(sensitive.sum()),
        frozen_wurmple_screen_local=scores.tolist(),frozen_wurmple_screen_frequency=planner.prior.tolist(),
        buttons=planner.buttons.tolist(),trigger_on_fifth_equal_frame=True,chosen_on_this_draw=choice,
        training_used_frozen_screen=False,weights=planner.sha))
    write_json(effects/"fresh-registration.json",dict(seed=seed,limit=131072,frames_per_decision=24,memory_model=read(model/"fit.json")["model_sha256"],
        effect_model=planner.sha,comparison="empirical training button frequency versus current-image predictions on the same recovery trigger",
        triggering="first button of a retrieval recovery, following four quiet decisions",perturbations=0,in_game_restores=0,
        outcomes="original-ROM game progress and eligibility of the tested action change",order=["local","frequency"]))
    for mode in ("local","frequency"):
        runs=sorted(family.glob("persistent-"+mode+"-*"))
        if any((r/"result.json").exists() for r in runs):
            continue
        while int(next(l.split()[1] for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:")))<1700000:
            time.sleep(30)
        out=family/f"persistent-{mode}-{len(runs)+1:03d}"
        play(model,out,seed,131072,0,True,True,effects_directory=effects,effect_ranking=mode)
    audits={}
    tapes={}
    for mode in ("local","frequency"):
        r=next(p for p in sorted(family.glob("persistent-"+mode+"-*")) if (p/"result.json").exists())
        result=read(r/"result.json")
        file,=(r/f"step-{result['decisions']:07d}").glob("*.json")
        audit=read(file)
        tapes[mode]=file.with_suffix(".bin").read_bytes()
        audits[mode]=dict(maps=audit["maps_first_reached"],events=audit["events"],steps=audit["steps"],
                          original_inputs=hashlib.sha256(tapes[mode]).hexdigest(),effect_choices=result["effect_choices"],
                          first_changed_same_frame=result["first_different_ranked_choice"])
    shared=next((i for i,(a,b) in enumerate(zip(tapes["local"],tapes["frequency"])) if a!=b),min(map(len,tapes.values())))
    write_json(effects/"fresh-comparison.json",dict(games=audits,identical_first_buttons=shared))


if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
    run()
