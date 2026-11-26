"""Retry fixed offsets before every negative battle outcome in a predeclared training cohort."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from earned_starts import EarnedStarts, verified_reset_lineage
from emerald import Emerald
from pixel_env import game_paths
from reward_branch_search import visual_code,write_json
from screen_explorer import ScreenExplorer

root=project/"runs/pixels-only-01/official-wsl"
source=root/"paired-self-earned-frontiers/every-sealed-endpoint-01"
out=root/"controls/all-losses-practice"
read=lambda p:json.loads(p.read_text())


class OutcomeReward:
    def __init__(self):
        self.active=self.zero=self.paid=False

    def read(self,game,key):
        battle=key[0]==6
        field=key[0]==5 and game.verified_story(key) is not None
        if battle and not self.active:
            self.active=True
            self.zero=self.paid=False
        pulse=0
        if self.active and (battle or field):
            value=game.ram(0x0202433a,1)[0]
            self.zero|=value==0
            if self.zero and value and not self.paid:
                pulse={1:100,2:-100}.get(value,0)
                self.paid=True
            if field:
                self.active=False
        return pulse


def main():
    out.mkdir(parents=True,exist_ok=True)
    cfg=read(source/"settings.json")
    plan=dict(cohort="first 64 sealed 512-decision continuations in chronological work order, including every button and context",
              feedback="generic +1 for a won battle and -1 for a lost battle; the controller receives pixels and this scalar",
              eligibility="every negative scalar in the cohort, irrespective of location or opponent",
              offsets=[128,384],anchors="floor to 32 decisions within the original continuation",
              training_attempts_per_anchor=8,evaluation_attempts_per_anchor=4,horizon=768,seed=173759,
              evaluation="freeze learned action values and compare with uniform buttons on the same earned starts",
              fresh_game=False,training_uses_map_or_trainer_ids=False,frames_per_decision=24)
    write_json(out/"registration.json",plan)
    files=sorted(source.glob("discoveries/work-*/record.json"))
    selected=[(p,r) for p in files if (r:=read(p))["length"]-r["source_prefix_length"]==512][:64]
    if len(selected)<64:
        raise RuntimeError("The registered 64 branch outcomes are not complete")
    write_json(out/"cohort.json",dict(histories=[r["tape_sha256"] for _,r in selected]))
    if (out/"completed.json").exists():
        return
    game=Emerald(*game_paths())
    bank=EarnedStarts(root/"controls/dreamer-earned-resets/starts")
    anchor_cache={}
    losses={}
    try:
        for count,(p,r) in enumerate(selected,1):
            tape=p.with_name("game.bin").read_bytes()
            assert hashlib.sha256(tape).hexdigest()==r["tape_sha256"] and verified_reset_lineage(source,cfg,tape)
            anchor=r["source_prefix_length"]
            prefix=tape[:anchor]
            key=hashlib.sha256(prefix).digest()
            if key not in anchor_cache:
                start,index=max((s["prefix"],i) for i,s in enumerate(bank.records) if s["prefix"]<=anchor and prefix.startswith(bank.tape[:s["prefix"]]))
                image,score,_=bank.restore(game,index)
                total=bank.records[index]["scalar_return_cents"]
                signal=OutcomeReward()
                for b in tape[start:anchor]:
                    image,state=game.step(b)
                    rew,_=score.read(game,state)
                    total+=round(100*rew)
                    signal.read(game,state)
                anchor_cache[key]=(game.save(),image,score,total,signal)
            blob,image,score,total,signal=deepcopy(anchor_cache[key])
            game.load(blob)
            snaps={}
            for i,button in enumerate(tape[anchor:]):
                if i%32==0:
                    snaps[i]=(game.save(),image.copy(),deepcopy(signal),deepcopy(score),total)
                image,state=game.step(button)
                rew,_=score.read(game,state)
                total+=round(100*rew)
                if signal.read(game,state)<0:
                    for offset in plan["offsets"]:
                        relative=max(0,i+1-offset)//32*32
                        history=tape[:anchor+relative]
                        losses.setdefault(hashlib.sha256(history).hexdigest(),(history,snaps[relative]))
            assert total==r["return_cents"] and visual_code(image).hex()==r["screen"]
            write_json(out/"collection-status.json",dict(complete_histories=count,eligible_training_starts=len(losses)))
        model=ScreenExplorer(np.random.default_rng(plan["seed"]))
        rng=np.random.default_rng([plan["seed"],1])
        summary=Counter()
        total_actions=0
        for mode in ("training","frozen","uniform"):
            for name,(prefix,snapshot) in losses.items():
                for attempt in range(8 if mode=="training" else 4):
                    blob,image,signal,score,raw=deepcopy(snapshot)
                    game.load(blob)
                    history=bytearray(prefix)
                    pulse=0
                    trail=[]
                    for _ in range(plan["horizon"]):
                        state=visual_code(image)
                        if mode=="training":
                            button=model.choose(state)
                        elif mode=="uniform" or state not in model.transitions:
                            button=int(rng.integers(9))
                        else:
                            values=[]
                            for b in range(9):
                                hits=sum(model.transitions[state].get(b,{}).values())
                                values.append(model.reward_sum.get((state,b),0)/max(1,hits))
                            choices=np.flatnonzero(np.isclose(values,max(values)))
                            button=int(rng.choice(choices))
                        image,key=game.step(button)
                        event,done=score.read(game,key)
                        raw+=round(100*event)
                        pulse=signal.read(game,key)
                        history.append(button)
                        total_actions+=1
                        if mode=="training":
                            model.observe(state,button,visual_code(image),pulse/100)
                            trail.append((state,button))
                        if pulse or done:
                            break
                    if mode=="training" and pulse:
                        for distance,pair in enumerate(reversed(trail[:-1]),1):
                            model.reward_sum[pair]+=(pulse/100)*(.99**distance)
                    summary[mode,{100:"won",-100:"lost",0:"unresolved"}[pulse]]+=1
                    target=out/mode/f"{name[:12]}-{attempt:02d}"
                    target.mkdir(parents=True,exist_ok=True)
                    (target/"game.bin").write_bytes(history)
                    write_json(target/"game.json",dict(steps=len(history),first_visit_cells=len(score.cells),events=score.events,
                        maps_first_reached=score.first_maps,return_cents=raw,uncapped_exploration=True,training_only=True,
                        source_prefix=name,scalar_outcome=pulse,action_source=mode))
                    status=dict(real_retry_decisions=total_actions,results={m:{k:v for (label,k),v in summary.items() if label==m} for m in ("training","frozen","uniform")})
                    write_json(out/"status.json",status)
        write_json(out/"completed.json",dict(eligible_starts=len(losses),actions=total_actions,
                   results={m:{k:v for (label,k),v in summary.items() if label==m} for m in ("training","frozen","uniform")}))
    finally:
        game.close()


if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
    main()
