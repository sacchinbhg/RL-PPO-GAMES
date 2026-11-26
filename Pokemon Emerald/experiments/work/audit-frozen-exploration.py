"""Explain which parts of the game consumed a frozen episode after Mom's TV scene."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from emerald import Emerald, MAPS, PHASES
from pixel_env import Score, game_paths

official=project/"runs/pixels-only-01/official-wsl"


def audit(path):
    evidence=json.loads(path.read_text())
    tape=path.with_suffix(".bin").read_bytes()
    cut=evidence["events"].get("sent_to_meet_rival")
    g=Emerald(*game_paths())
    score=Score(evidence.get("uncapped_exploration",False))
    maps, phases = Counter(), Counter()
    whole_map_counts, whole_phase_counts = Counter(), Counter()
    first_any = {}
    entered={}
    all_maps, all_cells = set(),set()
    last_rew=0
    try:
        g.load(g.boot)
        for n,action in enumerate(tape,1):
            _,key=g.step(action)
            reward,_=score.read(g,key)
            whole_phase_counts[PHASES[key[0]]]+=1
            if reward: last_rew=n
            if cut and n>cut:
                phases[PHASES[key[0]]]+=1
            if key[0]!=5 or g.verified_story(key) is None:
                continue
            name=MAPS[key[1]][key[2]]
            whole_map_counts[name]+=1
            first_any.setdefault(name,n)
            all_maps.add(name)
            all_cells.add(key[1:5])
            if cut and n>cut:
                maps[name]+=1
                entered.setdefault(name,n)
        assert score.events==evidence["events"] and len(tape)==evidence["steps"]
    finally:
        g.close()
    return dict(source=path.relative_to(official).as_posix(), events=score.events, steps=len(tape),
                unique_cells=len(all_cells), areas=sorted(all_maps), last_positive_reward=last_rew,
                map_decisions=whole_map_counts, phase_decisions=whole_phase_counts, first_map_visits=first_any,
                post_tv_phases=phases,post_tv_area_counts=maps,first_visits_after_tv=entered)


if __name__=="__main__":
    phase=sys.argv[1] if len(sys.argv)>1 else "eval-65536"
    arms=sys.argv[2:] or ["seed-71299-h15","seed-71299-h15-reward-replay"]
    filegroups=[sorted((official/name/phase/"external-audit").glob("*/*/*episode-00001.json"),
                       key=lambda p:p.parent.name) for name in arms]
    assert all(len(group)==4 for group in filegroups)
    with ProcessPoolExecutor(max_workers=2) as executor:
        rows=list(executor.map(audit,sum(filegroups,[])))
    suffix = "-" + "-".join(arms) if sys.argv[2:] else ""
    (official/"controls"/f"frozen-exploration-{phase}{suffix}.json").write_text(json.dumps(rows,indent=2))
    for row in rows: print(row["source"].split("/")[0],row["events"],row["post_tv_area_counts"],flush=True)
