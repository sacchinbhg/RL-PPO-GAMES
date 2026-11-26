"""Compare independent buttons with direction-agnostic temporal persistence."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import numpy as np

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from emerald import Emerald
from pixel_env import Score, game_paths

official=project/"runs/pixels-only-01/official-wsl"
original=official/"controls/button-persistence"
extended = sys.argv[1:] == ["--extend-uniform"]
out=official/"controls/uniform-32768" if extended else original
seed, duration = 271340, 32768 if extended else 16384


def play(job):
    i, persistence = job
    tag="persistent" if persistence else "independent"
    dest=out/tag/f"game-{i:02}"
    dest.mkdir(parents=True,exist_ok=True)
    audit=dest/"episode.json"
    if audit.exists():
        return json.loads(audit.read_text())
    # All choices are fixed before loading the ROM. Each button has equal marginal probability.
    draws=np.random.default_rng(np.random.SeedSequence([seed,i]))
    actions=draws.integers(9,size=duration).astype(np.uint8)
    if persistence:
        coins=np.random.default_rng(np.random.SeedSequence([seed,i,5423])).random(duration)
        new=np.where(coins>=persistence,np.arange(duration),0)
        actions=actions[np.maximum.accumulate(new)]
    if not persistence and i<4:
        previous,=(official/f"controls/uniform-seed-271340/game-{i}").glob("*.bin")
        assert actions[:8192].tobytes()==previous.read_bytes()
    if extended:
        old = original/tag/f"game-{i:02}"/"episode.json"
        assert actions[:16384].tobytes() == old.with_suffix(".bin").read_bytes()
        old_cutoff = json.loads(old.read_text())["cutoffs"]["16384"]
    game=Emerald(*game_paths())
    score=Score()
    cells=set()
    total=0.
    cutoffs={}
    try:
        game.load(game.boot)
        for n,button in enumerate(actions,1):
            _,key=game.step(int(button))
            reward,done=score.read(game,key)
            total+=reward
            if key[0]==5 and game.verified_story(key) is not None:
                cells.add(key[1:5])
            if n in (8192,16384,duration) or done:
                cutoffs[n]=dict(events=dict(score.events),distinct_cells=len(cells))
                if extended and n == 16384:
                    assert cutoffs[n] == old_cutoff
            if done:
                break
        record=dict(game=i,seed=seed,persistence=persistence,events=score.events,steps=n,
                    first_visit_cells=len(score.cells),uncapped_exploration=False,return_sum=total,
                    cutoffs=cutoffs)
        (dest/"episode.bin").write_bytes(actions[:n].tobytes())
        temp=dest/"episode.partial.json"
        temp.write_text(json.dumps(record,indent=2))
        temp.replace(audit)
        return record
    finally:
        game.close()


if __name__=="__main__":
    jobs=[(i,p) for i in range(8) for p in ((0.,) if extended else (0.,.75))]
    rows=[]
    with ProcessPoolExecutor(max_workers=2 if extended else 3) as pool:
        for future in as_completed([pool.submit(play,j) for j in jobs]):
            record=future.result()
            rows.append(record)
            print(record["game"],record["persistence"],record["events"],flush=True)
    (out/"results.json").write_text(json.dumps(sorted(rows,key=lambda x:(x["game"],x["persistence"])),indent=2))
