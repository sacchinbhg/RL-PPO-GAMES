"""Replay original-ROM Emerald tapes and score two opening rewards on the same inputs."""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from emerald import BUTTONS, Emerald
from experiment import location, progress


EVENT_POINTS = {
    "bedroom": 1,
    "clock": 3,
    "sent_to_meet_rival": 1,
    "met_rival": 4,
    "route101": 4,
    "heard_birch_request": 2,
    "first_party": 10,
}
CELL_POINTS = 0.02
MAX_PAID_CELLS = 50


def verified_events(env, key):
    facts = env.verified_story(key)
    if facts is None:
        return {}
    rival, occupied = facts
    on_route = location(key) == "Route101" and rival
    rescue_state = 0
    if on_route:
        save_block = int.from_bytes(env.ram(0x03005D8C, 4), "little")
        rescue_state = int.from_bytes(env.ram(save_block + 0x139C + 2 * 0x60, 2), "little")
    return {
        "bedroom": key[5] >= 4,
        "clock": bool(key[6]),
        "sent_to_meet_rival": bool(key[6]) and key[5] >= 6,
        "met_rival": rival,
        "route101": on_route,
        "heard_birch_request": on_route and rescue_state >= 2,
        "first_party": occupied,
    }


def score(env, actions):
    env.load(env.boot)
    key = env.descriptor()
    old_best = progress(key)
    old_progress = 0
    old_novelty = 0
    old_seen = {key}
    cells = set()
    first = {}
    snapshots = {}
    bag_step = None
    for decision, button in enumerate(actions, 1):
        _, key = env.step(int(button))
        rank = progress(key)
        old_progress += max(0, rank - old_best)
        old_best = max(rank, old_best)
        old_novelty += 0.25 * (key not in old_seen)
        old_seen.add(key)
        signals = verified_events(env, key)
        if signals and key[7] and bag_step is None:
            bag_step = decision
        if signals and key[0] == 5:
            cells.add(key[1:5])
        for event, active in signals.items():
            if active and event not in first:
                first[event] = decision
        if decision in (596, 683, len(actions)):
            event_points = sum(EVENT_POINTS[name] for name in first)
            exploration = round(CELL_POINTS * min(len(cells), MAX_PAID_CELLS), 2)
            snapshots[decision] = {
                "old_progress": old_progress,
                "old_novelty": old_novelty,
                "old_total": old_progress + old_novelty,
                "proposed_events": event_points,
                "proposed_exploration": exploration,
                "proposed_total": round(event_points + exploration, 2),
            }
    return {"first_events": first, "first_verified_bag": bag_step, "snapshots": snapshots}


def main():
    diagnostics = PROJECT / "starter-diagnostics"
    evals = PROJECT / "runs/world-12h-20261124/evaluations"
    cfg = json.loads((PROJECT / "local.json").read_text(encoding="utf-8-sig"))
    env = Emerald(cfg["core"], cfg["rom"])
    results = {}
    try:
        for label, folder in (("final_policy", "final"), ("earlier_policy", "step-6009856")):
            source = evals / folder
            with np.load(source / "moments/buttons.npz") as tape:
                actions = tape["actions"][:1024, 0]
            result = score(env, actions)
            recorded = json.loads((source / "evaluation.json").read_text())["trials"][0]
            for event in ("clock", "met_rival"):
                when = recorded["verified_event_decision"].get(event)
                assert result["first_events"].get(event) == (when if when and when <= len(actions) else None)
            results[label] = result
        tape = (diagnostics / "guided-party.bin").read_bytes()
        guided = score(env, tape)
        assert guided["first_verified_bag"] == 674
        assert guided["first_events"]["first_party"] == 683
        results["agent_prefix_596_then_guided_87"] = guided
        wait = bytes([BUTTONS.index("wait")])
        padded = score(env, tape + wait * (1024 - len(tape)))
        assert padded["snapshots"][683] == guided["snapshots"][683]
        assert padded["first_events"] == guided["first_events"]
        results["same_guided_tape_then_wait_to_1024"] = padded
    finally:
        env.close()
    output = {
        "method": "Fresh mGBA boot per tape, no archive restore or goal, continuous undiscounted totals. Same inputs for both rewards. Frozen policies, trial index 0. The guided tape ends at 683. A separate control appends 341 wait decisions to compare equal budgets.",
        "event_points": EVENT_POINTS,
        "cell_points": CELL_POINTS,
        "max_paid_cells": MAX_PAID_CELLS,
        "runs": results,
    }
    destination = diagnostics / "red-reward-comparison.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
