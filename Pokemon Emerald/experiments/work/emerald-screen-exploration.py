"""Independent exploration pilot using only the same RGB, scalar reward and nine buttons."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

project = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL"
sys.path.insert(0,str(project))
from pixel_env import PixelPool
from screen_explorer import ScreenExplorer, screen_key

root = project/"runs/pixels-only-01/official-wsl"
out = root/"controls/screen-exploration-32768"


def play(index, limit=32768, directory=None, encode=screen_key):
    seed = 271340
    directory = directory or out/f"game-{index:02}"
    directory.mkdir(parents=True,exist_ok=True)
    result = directory/"result.json"
    if result.exists():
        return json.loads(result.read_text())
    policy = ScreenExplorer(np.random.SeedSequence([seed,index]))
    pool = PixelPool(1)
    total = 0.
    try:
        state = encode(pool.command(0,"reset"))
        for step in range(1,limit+1):
            button = policy.choose(state)
            img, reward, done = pool.command(0,"act",button)
            following = encode(img)
            policy.observe(state,button,following,reward)
            state = following
            total += reward
            if step%8192==0 or reward>=4 or done:
                audit_dir = directory/f"step-{step:06}"
                pool.command(0,"audit",str(audit_dir))
                if step in (32768, 131072) and step < limit:
                    previous = out if step == 32768 else out / "selected-131072"
                    old,=(previous/f"game-{index:02}"/f"step-{step:06}").glob("*.bin")
                    current,=audit_dir.glob("*.bin")
                    assert current.read_bytes()==old.read_bytes()
                    assert json.loads(old.with_suffix(".json").read_text()) == json.loads(current.with_suffix(".json").read_text())
            if done:
                break
    finally:
        pool.close()
    record = json.loads(next(audit_dir.glob("*.json")).read_text())
    assert record["steps"] == step
    record.update(game=index, distinct_screen_codes=len(policy.visits),
                  observed_screen_button_pairs=sum(len(v) for v in policy.transitions.values()),
                  return_sum=total, seed=seed)
    if limit>32768:
        record["verified_original_prefix"]=131072 if limit>131072 and step>=131072 else 32768
    result.write_text(json.dumps(record,indent=2))
    print(index,record["events"],"screens",len(policy.visits),flush=True)
    return record


if __name__ == "__main__":
    out.mkdir(parents=True,exist_ok=True)
    if sys.argv[1:] in (["--extend-best"], ["--extend-best-further"]):
        pilot=json.loads((out/"results.json").read_text())["games"]
        assert len(pilot)==4
        best=max(pilot,key=lambda row:(row["return_sum"],-row["game"]))
        limit = 262144 if sys.argv[1:] == ["--extend-best-further"] else 131072
        target=out/f"selected-{limit}"
        target.mkdir(exist_ok=True)
        (target/"selection.json").write_text(json.dumps(dict(
            selection="highest scalar return in four 32768-button pilot games; lowest index breaks ties",
            selected=best["game"],pilot_returns=[x["return_sum"] for x in pilot],
            limit=limit),indent=2))
        play(best["game"],limit,target/f"game-{best['game']:02}")
        sys.exit(0)
    if sys.argv[1:]:
        raise SystemExit("Expected no arguments or --extend-best")
    with ProcessPoolExecutor(max_workers=2) as tasks:
        rows = list(tasks.map(play,range(4)))
    (out/"results.json").write_text(json.dumps(dict(controller="online empirical RGB transition counts",
        games=rows, budgets=[8192,16384,24576,32768], receives=["RGB","scalar reward","episode end"],
        fresh_boot=True, emulator_restores_during_game=0, shared_with_dreamer=False),indent=2))
