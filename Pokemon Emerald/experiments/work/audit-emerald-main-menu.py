"""Count actual main-menu accept and cancel handlers on unmodified, completed game tapes."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import sys

work = Path(__file__).resolve().parent
project = work.parent / "outputs/EmeraldRL"
sys.path.insert(0, str(project))
from emerald import BUTTON_IDS, Emerald, REPEAT
from pixel_env import Score, game_paths

official = project / "runs/pixels-only-01/official-wsl"
arm = official / "seed-71299-h15-reward-replay"
handlers = {"Task_HandleMainMenuAPressed": "A", "Task_HandleMainMenuBPressed": "B"}


def symbols():
    names = set(handlers) | {"gTasks", "gMain", "CB2_MainMenu"}
    result = {}
    for line in (work / "emerald.sym").read_text().splitlines():
        fields = line.split()
        if fields and fields[-1] in names:
            result[fields[-1]] = int(fields[0], 16)
    assert result.keys() == names
    return result


def audit_game(audit):
    expected = json.loads(audit.read_text())
    tape = audit.with_suffix(".bin").read_bytes()
    assert len(tape) == expected["steps"]
    names = symbols()
    states = {names[k]: value for k, value in handlers.items()}
    game = Emerald(*game_paths())
    score = Score()
    previous = None
    choices = []
    try:
        game.load(game.boot)
        for n, action in enumerate(tape, 1):
            if "bedroom" in score.events:
                _, key = game.step(action)
            else:
                button = BUTTON_IDS[action]
                for frame in range(REPEAT):
                    game.buttons = (1 << button) if button is not None and frame < 14 else 0
                    game.capture = frame == REPEAT - 1
                    game.lib.retro_run()
                    cb = int.from_bytes(game.ram(names["gMain"] + 4, 4), "little") & ~1
                    current = None
                    if cb == names["CB2_MainMenu"]:
                        tasks = game.ram(names["gTasks"], 16 * 40)
                        for offset in range(0, len(tasks), 40):
                            handle = struct.unpack_from("<I", tasks, offset)[0] & ~1
                            if tasks[offset+4] and handle in states:
                                menu_type, item = struct.unpack_from("<hh", tasks, offset+8)
                                assert menu_type == 0 and item in (0, 1)
                                current = (states[handle], item)
                                if current != previous:
                                    choices.append(dict(button=states[handle],
                                                        item=("New Game","Options")[item],
                                                        decision=n, frame=frame))
                                break
                    previous = current
                key = game.descriptor()
            score.read(game, key)
        assert score.events == expected["events"] and len(score.cells) == expected["first_visit_cells"]
    finally:
        game.close()
    counts = Counter((c["button"], c["item"]) for c in choices)
    return dict(source=audit.relative_to(official).as_posix(), steps=len(tape), events=score.events,
                input_sha256=hashlib.sha256(tape).hexdigest(), checked_from_boot=True,
                handler_counts={f"{key} on {item}": val for (key,item),val in counts.items()},
                choices=choices)


if __name__ == "__main__":
    files = []
    for stage in ("eval-65536", "eval-262144"):
        paths = sorted((arm/stage/"external-audit").glob("*/*/*.json"))
        assert len(paths) == 4
        files.extend(paths[:1] if stage == "eval-65536" else paths)
    dest = official / "controls" / "frozen-main-menu"
    dest.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for path, row in zip(files, pool.map(audit_game, files)):
            target = dest / (path.parents[3].name + "-" + path.parent.name + ".json")
            target.write_text(json.dumps(row, indent=2))
            print(target.name, row["handler_counts"], flush=True)
