"""Inspect an already finished playthrough. The trainer never reads this output."""
from collections import Counter
import json
from pathlib import Path
import sys
project = Path(__file__).resolve().parents[1] / "outputs" / "EmeraldRL"
sys.path.insert(0, str(project))
from emerald import Emerald, MAPS, PHASES
from pixel_env import Score, game_paths

training = project / "runs/pixels-only-01/official-wsl/seed-71299-h15/train"
audit, = (training / "external-audit").glob("*/env-1/*.json")
evidence = json.loads(audit.read_text())
tape = audit.with_suffix(".bin").read_bytes()
cut = evidence["events"]["sent_to_meet_rival"]
g = Emerald(*game_paths())
score, maps, phases = Score(), Counter(), Counter()
seen = set()
first_arrival, paid_out, last_reward = {}, None, 0
try:
    g.load(g.boot)
    for i, action in enumerate(tape, 1):
        _, key = g.step(action)
        reward, _ = score.read(g, key)
        if reward:
            last_reward = i
        if len(score.cells) == 50 and paid_out is None:
            paid_out = i
        if i <= cut:
            continue
        phases[PHASES[key[0]]] += 1
        if key[0] != 5 or not g.verified_story(key):
            continue
        group, number = key[1:3]
        name = MAPS[group][number]
        maps[name] += 1
        first_arrival.setdefault(name, i)
        seen.add(key[1:5])
    assert score.events == evidence["events"] and len(tape) == evidence["steps"]
finally:
    g.close()
result = dict(source=str(audit.relative_to(project)), events=score.events, buttons=len(tape),
              after_story_decision=cut, phase_counts_after_story=phases, map_counts_after_story=maps,
              map_first_visits_after_story=first_arrival, unique_overworld_cells_after_story=len(seen),
              first_visit_bonus_fully_paid_at=paid_out, final_positive_reward_at=last_reward)
out = training.parent / "highlights/after-clock-diagnostic.json"
out.write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
