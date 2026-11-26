"""Compare reward schedules on the same recorded buttons without contacting a learner."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

project = Path(__file__).resolve().parents[1] / "outputs" / "EmeraldRL"
sys.path.insert(0, str(project))
from emerald import Emerald
from pixel_env import Score, game_paths

official = project / "runs/pixels-only-01/official-wsl"


def compare(audit):
    expected = json.loads(audit.read_text())
    tape = audit.with_suffix(".bin").read_bytes()
    seen = set()
    extra = []
    recent_credit = [0, 0]
    returns = [0., 0.]
    score = Score()
    env = Emerald(*game_paths())
    try:
        env.load(env.boot)
        for step, action in enumerate(tape, 1):
            _, key = env.step(action)
            reward, _ = score.read(env, key)
            add = 0.
            if key[0] == 5 and env.verified_story(key) is not None and key[1:5] not in seen:
                seen.add(key[1:5])
                if len(seen) > 50:
                    add = .02
                    extra.append(step)
            for k, value in enumerate((reward, reward + add)):
                returns[k] += value
                if value:
                    recent_credit[k] = step
        assert score.events == expected["events"] and len(tape) == expected["steps"]
        assert len(score.cells) == expected["first_visit_cells"] == min(len(seen), 50)
    finally:
        env.close()
    return dict(audit=audit.relative_to(official).as_posix(), buttons=len(tape),
                distinct_cells=len(seen), lost_new_cell_signals=len(extra),
                first_missed=extra[0] if extra else None, last_missed=extra[-1] if extra else None,
                baseline_return=round(returns[0], 4), uncapped_return=round(returns[1], 4),
                last_positive_capped=recent_credit[0], last_positive_uncapped=recent_credit[1])


if __name__ == "__main__":
    arm = official / "seed-71299-h15"
    groups = [sorted((arm / phase / "external-audit").glob("*/*/*.json"))
              for phase in ("train", "eval-65536")]
    groups.append(sorted((official / "controls/uniform-seed-271340").glob("game-*/*.json")))
    assert all(len(group) == 4 for group in groups)
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(compare, sum(groups, [])))
    out = official / "controls/reward-cap-replay.json"
    out.write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(row["audit"], row["distinct_cells"], row["lost_new_cell_signals"],
              row["last_positive_capped"], row["last_positive_uncapped"], flush=True)
