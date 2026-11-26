"""Replay a completed tape for human-readable plateau diagnostics, isolated from collection."""
from collections import Counter
import json
from pathlib import Path
import sys

project = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL"
sys.path.insert(0, str(project))
from emerald import BUTTONS, Emerald, MAPS, PHASES, verify_rom
from pixel_env import game_paths, Score
from screen_explorer import screen_key

root = project / "runs/pixels-only-01/official-wsl/controls/screen-exploration-32768"


def run():
    source = root / "selected-131072/game-01/step-131072"
    audit, = source.glob("*.json")
    expected = json.loads(audit.read_text())
    tape = audit.with_suffix(".bin").read_bytes()
    assert len(tape) == expected["steps"]
    folder = root / "selected-131072/diagnostics"
    folder.mkdir(exist_ok=True)
    core, rom = game_paths()
    verify_rom(rom)
    game = Emerald(str(core), str(rom))
    score = Score()
    windows = []
    try:
        game.load(game.boot)
        for n, button in enumerate(tape, 1):
            if (n - 1) % 8192 == 0:
                actions, states, phases, maps = (Counter() for _ in range(4))
                cells = set()
                reward_sum = 0.
            rgb, key = game.step(button)
            reward_sum += score.read(game, key)[0]
            actions[BUTTONS[button]] += 1
            states[screen_key(rgb)] += 1
            phases[PHASES[key[0]]] += 1
            if key[0] == 5 and game.verified_story(key) is not None:
                g, m = key[1:3]
                if 0 <= g < len(MAPS) and 0 <= m < len(MAPS[g]):
                    maps[MAPS[g][m]] += 1
                    cells.add(tuple(key[1:5]))
            if n % 8192 == 0 or n == len(tape):
                windows.append(dict(end=n, button_counts=actions, distinct_pixel_codes=len(states),
                                    most_common_pixel_count=states.most_common(1)[0][1],
                                    phases=phases, overworld_maps=maps, overworld_cells=len(cells),
                                    scalar_return=reward_sum, events_so_far=dict(score.events)))
                interim = folder / "progress.tmp"
                interim.write_text(json.dumps(windows, indent=2))
                interim.replace(folder / "progress.json")
        assert expected["events"] == score.events
        (folder / "verified.json").write_text(json.dumps(dict(whole_tape_verified=True,
            total_buttons=len(tape), events=score.events, windows=windows), indent=2))
    finally:
        game.close()


if __name__ == "__main__":
    run()
