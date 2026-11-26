"""Measure how the next screen moves after trying each learned movement button from the same position."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from collect_button_alternatives import effective
from fit_visual_motion import translation
from reward_branch_env import ForkEnv
from train_earned_dreamer import Trajectory, digest, write_json


def collect(dataset, motion, output, stride=160, skip=0):
    game = Trajectory(dataset)
    probe = json.loads(motion.read_text())
    if not probe["informative"] or probe["position_or_map_input"]:
        raise ValueError("Expected a validated motion probe learned from pixels")
    shifts = {int(k): tuple(v["most_common"]) for k, v in probe["learned_actions"].items()}
    buttons = sorted(shifts)
    if len(buttons) < 2 or stride < 1 or not 0 <= skip < len(game.tape):
        raise ValueError("Invalid sampling schedule or movement action set")
    output.mkdir(parents=True, exist_ok=True)
    plan = dict(training_tape=game.source["source_sha256"], motion_probe=digest(motion), stride=stride,
                skip_prefix=skip, buttons=buttons, attempts_per_button=3,
                trigger="first confidently scrolling observation after the interval",
                inputs="the current RGB image, candidate button, and motion learned from training observations")
    registration = output / "registration.json"
    if registration.exists() and json.loads(registration.read_text()) != plan:
        raise ValueError("Another collection owns this directory")
    write_json(registration, plan)
    if (output / "completed.json").exists():
        if digest(output / "examples.npz") != json.loads((output / "completed.json").read_text())["data_sha256"]:
            raise ValueError("Existing motion samples changed")
        return
    env = ForkEnv()
    images, targets, decisions = [], [], []
    counts = Counter()
    next_sample = max(skip + 1, stride)
    total = visits = 0
    try:
        image = env.initial
        for i, original in enumerate(game.tape, 1):
            before = image
            image, cents, done, used = env.command("step", (original, 1))
            capped, visits = effective(cents, visits)
            total += cents if game.audit["uncapped_exploration"] else capped
            if used != 1 or done and i != len(game.tape):
                raise ValueError("Unexpected end of original training tape")
            if i >= next_sample and i < len(game.tape) and original in shifts and translation(before, image) == shifts[original]:
                env.command("save", 0)
                row = []
                for action in buttons:
                    observed = env.command("restore", 0)
                    if not np.array_equal(image, observed):
                        raise ValueError("A fork changed the input screen")
                    scrolled = False
                    for _ in range(3):
                        following, _, ended, used = env.command("step", (action, 1))
                        if used != 1:
                            raise ValueError("Invalid emulation duration")
                        counts["alternative_buttons"] += 1
                        scrolled |= translation(observed, following) == shifts[action]
                        observed = following
                        if ended:
                            break
                    row.append(scrolled)
                image = env.command("restore", 0)
                images.append(image.copy())
                targets.append(row)
                decisions.append(i)
                next_sample = i + stride
            if i % 4096 == 0:
                write_json(output / "status.json", dict(cold_decisions=i, expected=len(game.tape), positions=len(images),
                           alternative_buttons=counts["alternative_buttons"]))
        if total != game.source["return_cents"]:
            raise ValueError("Cold replay differs from the source episode reward")
    finally:
        env.close()
    if not targets:
        raise ValueError("No action-conditioned motion samples")
    outcomes = np.asarray(targets, bool)
    temp = output / "examples.tmp.npz"
    np.savez_compressed(temp, rgb=np.stack(images), scroll=outcomes, decision=np.asarray(decisions), button=np.asarray(buttons))
    file = output / "examples.npz"
    temp.replace(file)
    write_json(output / "completed.json", dict(**plan, data_sha256=digest(file), positions=len(images),
               action_sensitive_positions=int((outcomes.any(1) & ~outcomes.all(1)).sum()),
               alternative_buttons=counts["alternative_buttons"], cold_buttons=len(game.tape)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "motion", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--stride", type=int, default=160)
    parser.add_argument("--skip-prefix", type=int, default=0)
    args = parser.parse_args()
    collect(args.dataset, args.motion, args.output, args.stride, args.skip_prefix)
