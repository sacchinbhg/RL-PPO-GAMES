"""Cold-replay each comparison arm and time its return to outdoor routes after an actual whiteout."""
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from emerald import MAPS
from pixel_env import Score,WhiteoutLog
import pixel_replay
from reward_branch_search import write_json
root=project/"runs/pixels-only-01/official-wsl"
old=root/"earned-route-extension/trial-197461"
experiment=root/"earned-route-recovery"
out=experiment/"original-rom-recovery"
read=lambda p:json.loads(p.read_text())

def check(arm,index,label,video):
    dest=out/label
    dest.mkdir(parents=True,exist_ok=True)
    result=dest/"verified.json"
    if result.exists():
        return read(result)
    audit,=(arm/"games"/f"env-{index}"/"step-0065536").glob("*.json")
    info=read(audit)
    observed=[]
    class ObservedScore(Score):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            self.loss=WhiteoutLog()
            self.counts=[]
            observed.append(self)
        def read(self,game,key):
            reward,done=super().read(game,key)
            self.loss.observe(game,self.decisions,key)
            if len(self.counts)<len(self.loss.events):
                self.counts.append({})
            if self.counts and key[0]==5 and game.verified_story(key) is not None:
                name=MAPS[key[1]][key[2]]
                self.counts[-1][name]=self.counts[-1].get(name,0)+1
            return reward,done
    review=info["maps_after_whiteout"][0]["Route101"] if video else info["steps"]
    with patch.object(pixel_replay,"Score",ObservedScore):
        pixel_replay.replay(audit,dest/"replay",None,video,decision=review)
    measured,=observed
    proof=read(dest/"replay/verification.json")
    assert proof["restores_during_game"]==0 and measured.events==info["events"]
    if "whiteouts" in info:
        assert measured.loss.events==info["whiteouts"] and measured.loss.maps_after==info["maps_after_whiteout"]
    rows=[]
    for event,maps,counts in zip(measured.loss.events,measured.loss.maps_after,measured.counts):
        rows.append(dict(loss=event,first_map_after=maps,overworld_decisions=counts,
                    decisions_from_loss_to_route101=maps["Route101"]-event["decision"] if "Route101" in maps else None))
    data=dict(controller=label,button_budget=info["steps"],original_rom_sha1=proof["rom_sha1"],
              no_in_game_restores=True,tape_sha256=proof["input_sha256"],recoveries=rows)
    write_json(result,data)
    return data

if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),
        EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    out.mkdir(parents=True,exist_ok=True)
    while not (experiment/"comparison.json").exists():
        time.sleep(20)
    comparison=read(experiment/"comparison.json")
    assert comparison["matched_dreamer_before_intervention"] and comparison["matched_uniform_control"]
    new=Path(comparison["trial"])
    results={}
    for arm,index,label,video in [(new,0,"dreamer-with-recovery",True),(old,0,"dreamer-alone",False),(new,1,"uniform",False)]:
        write_json(out/"status.json",dict(time=time.time(),checking=label))
        results[label]=check(arm,index,label,video)
    write_json(out/"comparison.json",results)
