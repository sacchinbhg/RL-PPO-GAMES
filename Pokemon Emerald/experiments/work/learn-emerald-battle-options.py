"""Predict outcomes of sustained buttons from self-earned images and generic battle rewards."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from earned_starts import EarnedStarts,verified_reset_lineage
from emerald import Emerald
from fit_local_motion import descriptors
from pixel_env import game_paths
from train_earned_dreamer import write_json

def module(name):
    spec=importlib.util.spec_from_file_location(name,base/"work"/(name+".py"))
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

reward_module=module("practice-emerald-negative-outcomes")
prefix_length=module("continue-emerald-positive-outcomes").prefix_length
root=project/"runs/pixels-only-01/official-wsl"
observations=root/"controls/all-losses-practice"
sources=root/"paired-self-earned-frontiers/every-sealed-endpoint-01"
out=root/"controls/battle-option-model"
read=lambda p:json.loads(p.read_text())


def training_tapes():
    ids=set(read(observations/"cohort.json")["histories"])
    cfg=read(sources/"settings.json")
    tapes={}
    for p in sources.glob("discoveries/work-*/record.json"):
        r=read(p)
        if r["tape_sha256"] in ids:
            t=p.with_name("game.bin").read_bytes()
            assert hashlib.sha256(t).hexdigest()==r["tape_sha256"] and verified_reset_lineage(sources,cfg,t)
            tapes[r["tape_sha256"]]=t
    assert set(tapes)==ids
    return list(tapes.values())


def measure(game,bank,cohort,file,known,rng):
    a=read(file)
    tape=file.with_suffix(".bin").read_bytes()
    name=a["source_prefix"]
    n=prefix_length(tape,name,768)
    prefix=tape[:n]
    assert a["action_source"]=="training" and any(t.startswith(prefix) for t in cohort)
    start,index=max((r["prefix"],i) for i,r in enumerate(bank.records) if r["prefix"]<=n and prefix.startswith(bank.tape[:r["prefix"]]))
    image,score,_=bank.restore(game,index)
    reward_signal=reward_module.OutcomeReward()
    total=bank.records[index]["scalar_return_cents"]
    for b in prefix[start:]:
        image,key=game.step(b)
        reward,_=score.read(game,key)
        total+=round(100*reward)
        reward_signal.read(game,key)
    snap=game.save()
    feature=descriptors(image[None])[0]
    # Exclude earlier records from the same uninterrupted prefix in either direction.
    prior=[(np.mean((v["feature"]-feature)**2),v) for v in known if not (prefix.startswith(v["prefix"]) or v["prefix"].startswith(prefix))]
    prior.sort(key=lambda x:x[0])
    prediction=np.mean([r["wins"] for _,r in prior[:4]],axis=0) if prior else np.full(9,1/9)
    chosen=np.flatnonzero(np.isclose(prediction,prediction.max()))
    folder=out/"positions"/name
    folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/"pretrial-prediction.json",dict(prior_candidates=len(prior),scores=prediction.tolist(),tied_buttons=chosen.tolist(),steps=n))
    trials=[]
    for raw_button in rng.permutation(9):
        button=int(raw_button)
        game.load(snap)
        current=deepcopy(score)
        signal=deepcopy(reward_signal)
        value=total
        actions=bytearray(prefix)
        pulse=0
        for step in range(1,769):
            _,key=game.step(button)
            scalar,done=current.read(game,key)
            value+=round(100*scalar)
            pulse=signal.read(game,key)
            actions.append(button)
            if pulse or done:
                break
        trials.append(dict(button=button,decisions=step,scalar_outcome=pulse))
        if pulse>0 or done:
            dest=out/"training"/(name[:12]+f"-{button:02d}")
            dest.mkdir(parents=True,exist_ok=True)
            (dest/"game.bin").write_bytes(actions)
            write_json(dest/"game.json",dict(steps=len(actions),events=current.events,first_visit_cells=len(current.cells),
                       maps_first_reached=current.first_maps,uncapped_exploration=True,return_cents=value,
                       training_only=True,source_prefix=name,scalar_outcome=pulse,action_source="training"))
        write_json(folder/"trials.json",trials)
    wins=np.array([r["scalar_outcome"]>0 for r in sorted(trials,key=lambda r:r["button"])],float)
    write_json(folder/"completed.json",dict(prefix=name,original_prefix_steps=n,all_options_measured=True,
               wins=int(wins.sum()),real_decisions=sum(r["decisions"] for r in trials),prior_candidates=len(prior),
               predicted_choice_win_fraction=float(wins[chosen].mean()),uniform_option_win_fraction=float(wins.mean())))
    return dict(feature=feature,prefix=prefix,wins=wins)


if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"registration.json",dict(starts="each distinct registered generic-loss practice prefix in order of first use by the training arm",
               actions="all nine buttons; repeat one button until the next scalar win/loss or 768 decisions",seed=173797,frames_per_decision=24,
               predictor="four closest earlier training RGB images; every button weighted equally before any examples",
               evaluation="rank before measuring each new start and exclude ancestor or descendant action prefixes; related game openings can still overlap",
               map_or_opponent_input=False,game_tactics_supplied=False))
    cohort=training_tapes()
    bank=EarnedStarts(root/"controls/dreamer-earned-resets/starts")
    rng=np.random.default_rng(173797)
    known=[]
    seen=set()
    game=Emerald(*game_paths())
    try:
        while True:
            for f in sorted((observations/"training").glob("*/game.json"),key=lambda p:p.stat().st_mtime_ns):
                name=read(f)["source_prefix"]
                if name in seen:
                    continue
                known.append(measure(game,bank,cohort,f,known,rng))
                seen.add(name)
                write_json(out/"status.json",dict(measured_starts=len(seen),starts_with_winning_option=sum(r["wins"].any().item() for r in known),
                           real_option_wins=int(sum(r["wins"].sum() for r in known))))
            if (observations/"completed.json").exists():
                write_json(out/"completed.json",dict(measured_starts=len(seen)))
                break
            time.sleep(20)
    finally:
        game.close()
