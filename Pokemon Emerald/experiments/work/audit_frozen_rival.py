"""Replay a frozen policy's recorded controller inputs and inspect Emerald story variables."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1] / "outputs" / "EmeraldRL"
sys.path.insert(0, str(PROJECT))
from emerald import Emerald
from experiment import location

EVAL = PROJECT / "runs/world-12h-20261124/evaluations/step-855552"


def replay(trial):
    config = json.loads((PROJECT / "local.json").read_text(encoding="utf-8-sig"))
    expected = json.loads((EVAL / "evaluation.json").read_text())["trials"][trial]
    actions = np.load(EVAL / "moments/buttons.npz")["actions"][:, trial]
    env = Emerald(config["core"], config["rom"])
    first_map, first_event = {}, {}
    before = deque(maxlen=6)
    directory = EVAL / "rival-audit"
    directory.mkdir(exist_ok=True)
    try:
        for decision, action in enumerate(actions, 1):
            _, key = env.step(int(action))
            name = location(key)
            if name:
                first_map.setdefault(name, decision)
            if key[0] != 5 or not name:
                continue
            read = lambda addr, n: int.from_bytes(env.ram(addr, n), "little")
            sb = read(0x03005D8C, 4)
            sb2 = read(0x03005D90, 4)
            if not 0x02000000 <= sb2 < 0x02040000:
                continue
            gender = read(sb2+8, 1)
            if gender not in (0, 1):
                continue
            def var(v):
                return read(sb + 0x139c + 2*v, 2)
            rival_home = ("MaysHouse" if gender == 0 else "BrendansHouse") in name
            if rival_home:
                before.append((decision, env.screen.copy()))
            events = {
                "rival_house": rival_home,
                "rival_bedroom": rival_home and name.endswith("_2F"),
                "met_rival": var(0x50) == 1 and var(0x8d) >= 3,
                "route101": name == "Route101" or var(0x60) >= 1,
                "party": 1 <= read(0x020244e9, 1) <= 6,
            }
            for event, present in events.items():
                if present and event not in first_event:
                    first_event[event] = decision
                    Image.fromarray(env.screen).save(directory / f"trial-{trial}-{event}.png")
                    if event == "met_rival" and before:
                        n, frame = before[0]
                        Image.fromarray(frame).save(directory / f"trial-{trial}-dialogue-{n}.png")
    finally:
        env.close()
    return dict(trial=trial, decisions=len(actions), first_event_decisions=first_event,
                first_map_decisions_match_original=(first_map == expected["first_map_decision"]))


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(replay, range(4)))
    report = dict(policy_sha256=json.loads((EVAL/"evaluation.json").read_text())["policy_sha256"],
                  source="Fresh boot, exact recorded buttons, no restores. Story counters sampled in overworld.",
                  trials=rows)
    print(json.dumps(report, indent=2))
    (EVAL/"rival-audit/report.json").write_text(json.dumps(report, indent=2)+"\n")
