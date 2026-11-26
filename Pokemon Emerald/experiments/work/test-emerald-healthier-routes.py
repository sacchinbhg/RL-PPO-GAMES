"""Counterfactual self-imitation on every self-earned interval that ends with lower health."""
from bisect import bisect_left
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from earned_starts import EarnedStarts
from emerald import Emerald
from pixel_env import game_paths
from train_earned_dreamer import write_json

root=project/"runs/pixels-only-01/official-wsl"
out=root/"controls/health-preserving-self-imitation"
read=lambda p:json.loads(p.read_text())


def resource(game):
    hp=int.from_bytes(game.ram(0x020244ec+86,2),"little")
    maximum=int.from_bytes(game.ram(0x020244ec+88,2),"little")
    if not 0<=hp<=maximum or not maximum:
        raise ValueError("Invalid health for the self-earned party")
    return hp/maximum


def visual(image):
    return hashlib.blake2b(image.tobytes(),digest_size=16).digest()


def evaluate(game,bank,index,folder,seed):
    first,following=bank.records[index:index+2]
    start,end=first["prefix"],following["prefix"]
    reference=bank.tape[start:end]
    original,score,_=bank.restore(game,index)
    keys=[visual(original)]
    values=[resource(game)]
    samples={}
    future=defaultdict(list)
    raw=first["scalar_return_cents"]
    for i,button in enumerate(reference):
        if i%32==0:
            samples[i]=(game.save(),deepcopy(score),raw)
        image,key=game.step(button)
        reward,done=score.read(game,key)
        raw+=round(100*reward)
        assert not done
        keys.append(visual(image))
        future[keys[-1]].append(i+1)
        values.append(resource(game))
    expected_events=set(score.events)
    with np.load(bank.directory/following["filename"],allow_pickle=False) as arrays:
        if not np.array_equal(image,arrays["image"]) or raw!=following["scalar_return_cents"]:
            write_json(folder/"unmatched-control.json",dict(raw=raw,expected=following["scalar_return_cents"],image_match=bool(np.array_equal(image,arrays["image"]))))
            return []
    harm=next(i for i in range(1,len(values)) if values[i]<values[i-1])
    origin=max(0,harm-192)//32*32
    baseline=dict(start=start,end=end,first_negative_scalar=start+harm,intervention=start+origin,
                  original_final_fraction=values[-1],original_scalar_return=raw,matched_original_screen=True)
    write_json(folder/"control.json",baseline)
    rng=np.random.default_rng([seed,index])
    results=[]
    for first_button in rng.permutation(9):
        blob,copied,total=samples[origin]
        tracker=deepcopy(copied)
        game.load(blob)
        actions=bytearray()
        pointer=origin
        matched_full=False
        rejoined=None
        quiet=0
        # A short symmetric perturbation, followed by actions learned from the original route.
        for t in range(min(end-start-origin+256,1792)):
            if t<8:
                b=int(first_button)
            else:
                label=visual(image)
                positions=future.get(label,())
                j=bisect_left(positions,pointer)
                near=[p for p in positions[max(0,j-2):j+2] if abs(p-pointer)<=96]
                if near:
                    pointer=min(near,key=lambda p:(abs(p-pointer),-p))
                    quiet=0
                else:
                    quiet+=1
                # Rejoin an observed post-loss screen with more health. The rest of
                # the route must then work with its original buttons without any help.
                eligible=[p for p in positions if harm<=p<=pointer+256 and p>=max(harm,pointer-96)
                          and resource(game)>values[p]+1e-6]
                if eligible:
                    rejoined=min(eligible,key=lambda p:abs(p-pointer))
                    break
                if pointer>=len(reference):
                    break
                b=int(rng.integers(9)) if quiet>32 and quiet%8==0 else reference[pointer]
            image,key=game.step(b)
            reward,done=tracker.read(game,key)
            total+=round(100*reward)
            actions.append(b)
            pointer=min(len(reference),pointer+1)
            if done:
                break
        if rejoined is not None:
            for b in reference[rejoined:]:
                image,key=game.step(b)
                reward,done=tracker.read(game,key)
                total+=round(100*reward)
                actions.append(b)
                if done:
                    break
            matched_full=visual(image)==keys[-1] and set(tracker.events)==expected_events
        hp=resource(game)
        improved=matched_full and hp>values[-1]+1e-6 and total>=raw
        result=dict(first_button=int(first_button),real_decisions=len(actions),rejoined_at=start+rejoined if rejoined is not None else None,
                    matched_final_training_screen_and_events=matched_full,final_health_fraction=hp,return_cents=total,
                    improved=bool(improved),terminal=bool(done))
        if improved or done:
            target=folder/f"improved-{first_button}"
            target.mkdir(exist_ok=True)
            tape=bank.tape[:start+origin]+actions
            (target/"game.bin").write_bytes(tape)
            write_json(target/"game.json",dict(steps=len(tape),events=tracker.events,first_visit_cells=len(tracker.cells),
                       maps_first_reached=tracker.first_maps,uncapped_exploration=True,return_cents=total,training_only=True,
                       health_fraction=hp,training_source=bank.info["source_sha256"]))
            result["tape_sha256"]=hashlib.sha256(tape).hexdigest()
        results.append(result)
        write_json(folder/"trials.json",results)
    return results


if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
    bank=EarnedStarts(root/"controls/dreamer-earned-resets/starts")
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"registration.json",dict(source=bank.info["source_sha256"],selection="all adjacent earned start pairs with a net decrease in party health fraction",
               order="largest scalar decrease first",perturbation="each of nine buttons for eight decisions, from a 32-decision boundary 192 decisions before first decrease",
               continuation="match current pixels to the agent's observed input history; after rejoining a healthier matching post-loss screen play the original remaining buttons",
               accepts="same final image and milestones, equal or greater ordinary scalar return and strictly higher health fraction",
               frames_per_decision=24,training_only=True,actor_observes="RGB and scalar health fraction",position_or_trainer_used=False))
    game=Emerald(*game_paths())
    try:
        levels=[]
        for i in range(len(bank.records)):
            bank.restore(game,i)
            levels.append(resource(game))
        selected=sorted((i for i in range(len(levels)-1) if levels[i+1]<levels[i]),key=lambda i:levels[i+1]-levels[i])
        write_json(out/"selected.json",dict(interval_indices=selected,health_fractions=levels))
        total=wins=0
        for index in selected:
            folder=out/f"interval-{index:02}"
            folder.mkdir(exist_ok=True)
            if (folder/"trials.json").exists() and len(read(folder/"trials.json"))==9:
                rows=read(folder/"trials.json")
            else:
                rows=evaluate(game,bank,index,folder,173773)
            total+=sum(r["real_decisions"] for r in rows)
            wins+=sum(r["improved"] for r in rows)
            write_json(out/"status.json",dict(last_interval=index,actual_alternative_decisions=total,healthier_matching_routes=wins))
        write_json(out/"completed.json",dict(eligible_intervals=len(selected),actual_alternative_decisions=total,healthier_matching_routes=wins))
    finally:
        game.close()
