"""Independently replay completed rival games and locate their next bottleneck."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

project = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL"
sys.path.insert(0, str(project))
from emerald import Emerald, MAPS, PHASES
from pixel_env import Score, game_paths

root = project / "runs/pixels-only-01/official-wsl"


def audit(path):
    evidence = json.loads(path.read_text())
    cut = evidence["events"]["met_rival"]
    buttons = path.with_suffix(".bin").read_bytes()
    assert len(buttons) == evidence["steps"]
    game = Emerald(*game_paths())
    score = Score(evidence.get("uncapped_exploration", False))
    seen_cells, edges = set(), set()
    transitions, missed_cells = [], []
    post_maps, post_phases = Counter(), Counter()
    last_map = None
    last_reward = 0
    first_route = None
    try:
        game.load(game.boot)
        for step, button in enumerate(buttons, 1):
            _, key = game.step(button)
            reward, _ = score.read(game, key)
            if reward:
                last_reward = step
            if step > cut:
                post_phases[PHASES[key[0]]] += 1
            if key[0] != 5 or game.verified_story(key) is None:
                continue
            area = MAPS[key[1]][key[2]]
            cell = key[1:5]
            if step > cut:
                post_maps[area] += 1
                if cell not in seen_cells and cell not in score.cells:
                    missed_cells.append(step)
            seen_cells.add(cell)
            if area == "Route101" and first_route is None:
                first_route = step
            if last_map is not None and last_map != area:
                edge = (last_map, area)
                transitions.append(dict(step=step, source=last_map, destination=area,
                                        first_traversal=edge not in edges, reward=reward))
                edges.add(edge)
            last_map = area
        assert score.events == evidence["events"]
    finally:
        game.close()
    return dict(source=path.relative_to(root).as_posix(), steps=len(buttons), events=score.events,
                post_rival_phases=post_phases, post_rival_areas=post_maps,
                first_route101_overworld=first_route, post_rival_unpaid_new_cells=missed_cells,
                distinct_cells=len(seen_cells), last_reward=last_reward, transitions=transitions)


if __name__ == "__main__":
    groups = ("*/train/external-audit/*/*/*.json", "*/eval-65536-long/external-audit/*/*/*.json")
    files = sorted(p for group in groups for p in root.glob(group)
                   if "met_rival" in json.loads(p.read_text())["events"])
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(audit, files))
    output = root / "controls/after-rival-initial-games.json"
    output.write_text(json.dumps(results, indent=2))
    for row in results:
        print(row["source"], row["post_rival_areas"], row["first_route101_overworld"], flush=True)
