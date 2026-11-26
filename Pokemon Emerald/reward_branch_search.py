"""Explore from self-earned training states using only RGB, scalar rewards and button history."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from reward_branch_env import ForkEnv
from screen_explorer import ScreenExplorer


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def visual_code(pic):
    blocks = pic.reshape(4, 10, 6, 10, 3).sum(axis=(1, 3), dtype=np.uint16)
    return hashlib.blake2b((blocks // 6400).astype(np.uint8).tobytes(), digest_size=12).digest()


@dataclass
class Point:
    screen: bytes
    total: int
    length: int
    starts: int = 0
    pose: tuple | None = None
    stage: int = 0


class Archive:
    def __init__(self, env, rng, visual_slots=0):
        self.env, self.rng = env, rng
        self.points = {}
        self.recent = 0
        self.visual_slots = visual_slots
        self.keep(0, Point(visual_code(env.initial), 0, 0))

    def keep(self, slot, point):
        self.env.command("save", slot)
        self.points[slot] = point

    def offer_visual(self, point, visits, best):
        if not self.visual_slots or point.total < best - 100:
            return
        slots = [s for s in self.points if s >= 65]
        if any(self.points[s].screen == point.screen for s in slots):
            return
        if len(slots) < self.visual_slots:
            self.keep(65 + len(slots), point)
            return
        candidates = self.rng.choice(slots, min(6, len(slots)), replace=False)
        def priority(p):
            return (p.total >= best - 100, -visits[p.screen], -p.starts)
        slot = min(candidates, key=lambda s: priority(self.points[s]))
        if priority(point) >= priority(self.points[slot]):
            self.keep(int(slot), point)

    def offer(self, point, recent=False):
        if recent:
            slot = 49 + self.recent % 16
            self.recent += 1
            self.keep(slot, point)
            return
        elite = [s for s in self.points if 1 <= s <= 48]
        same = [s for s in elite if self.points[s].screen == point.screen]
        if same:
            slot = same[0]
        elif len(elite) < 48:
            self.keep(len(elite) + 1, point)
            return
        else:
            slot = min(elite, key=lambda s: (self.points[s].total, -self.points[s].length))
        prior = self.points[slot]
        if (point.total, -point.length) > (prior.total, -prior.length):
            self.keep(slot, point)

    def start(self):
        slots = sorted((s for s in self.points if s < 65), key=lambda s: (-self.points[s].total, self.points[s].length))
        draw = self.rng.random()
        high = self.points[slots[0]].total
        visual = [s for s in self.points if s >= 65 and self.points[s].total >= high - 100]
        if visual and draw < .45:
            slots = visual
        elif draw < .55:
            slots = slots[:6]
        elif draw < .85:
            slots = slots[:24]
        weights = np.array([1 / np.sqrt(1 + self.points[s].starts) for s in slots])
        chosen = int(self.rng.choice(slots, p=weights / weights.sum()))
        point = self.points[chosen]
        pic = self.env.command("restore", chosen)
        self.restored_image = pic
        if visual_code(pic) != point.screen:
            raise RuntimeError("Training fork returned a different image")
        point.starts += 1
        return point


def train(directory, seed, budget, visual_slots=0):
    directory.mkdir(parents=True, exist_ok=True)
    settings = dict(seed=seed, observation="40x60 RGB, scalar reward, episode end",
                    reward="existing story/level rewards and 0.02 per first visited cell, uncapped",
                    root="fresh original ROM boot", receives_saves_from_other_runs=False,
                    behavior="online empirical pixel transition model plus symmetric button durations",
                    training_forks=True, used_for_frozen_evaluation=False)
    if visual_slots:
        settings["unrewarded_screen_slots"] = visual_slots
    setup = directory / "settings.json"
    if setup.exists() and json.loads(setup.read_text()) != settings:
        raise ValueError("Run configuration changed")
    write_json(setup, settings)
    summary = directory / "best.json"
    old = json.loads(summary.read_text()) if summary.exists() else None
    status_path = directory / "status.json"
    prior = json.loads(status_path.read_text()) if status_path.exists() else {}
    work = max(prior.get("work", 0), old["work"] if old else 0)
    branches = max(prior.get("branches", 0), old["branches"] if old else 0)
    historical_peak = max(prior.get("historical_peak_cents", 0), prior.get("best_return_cents", 0))
    rng = np.random.default_rng([seed, work])
    model = ScreenExplorer(rng)
    env = ForkEnv(directory/"first-maps")
    started = time.monotonic()
    try:
        archive = Archive(env, rng, visual_slots)
        last_saved = best = 0
        if old:
            tape = (directory / old["audit"] / "game.bin").read_bytes()
            if hashlib.sha256(tape).hexdigest() != old["tape_sha256"]:
                raise RuntimeError("Saved training prefix checksum mismatch")
            total = 0
            state = visual_code(env.initial)
            for i, button in enumerate(tape, 1):
                pic, reward, terminal, _ = env.command("step", (button, 1))
                following = visual_code(pic)
                total += reward
                model.visits[state] += 1
                model.transitions.setdefault(state, {})
                model.observe(state, button, following, reward / 100)
                state = following
                if reward:
                    archive.offer(Point(state, total, i))
                elif visual_slots and i % 64 == 0:
                    archive.offer_visual(Point(state, total, i), model.visits, total)
            if total != old["return_cents"] or state.hex() != old["screen"]:
                raise RuntimeError("Cold boot failed to reproduce the self-earned training prefix")
            if terminal:
                return
            best = last_saved = total
        initial_work = work
        while work < budget:
            origin = archive.start()
            branches += 1
            state, total, length = origin.screen, origin.total, origin.length
            stop = work + int(rng.choice([128, 256, 512, 1024, 2048]))
            while work < min(stop, budget):
                button = model.choose(state)
                duration = min(int(rng.choice([1, 1, 2, 2, 4, 8])), budget - work, stop - work)
                pic, reward, terminal, used = env.command("step", (button, duration))
                following = visual_code(pic)
                model.observe(state, button, following, reward / 100)
                state, total, length, work = following, total + reward, length + used, work + used
                point = Point(state, total, length)
                if reward:
                    archive.offer(point)
                elif visual_slots and work // 64 != (work - used) // 64:
                    archive.offer_visual(point, model.visits, best)
                best = max(total, best)
                if (total == best and total > last_saved) or terminal:
                    relative = f"discoveries/work-{work:010d}"
                    target = directory / relative
                    env.command("audit", str(target))
                    digest = hashlib.sha256((target / "game.bin").read_bytes()).hexdigest()
                    record = dict(work=work, branches=branches, return_cents=total, length=length,
                                  screen=state.hex(), audit=relative, tape_sha256=digest,
                                  training_only=True, terminal=terminal)
                    write_json(target / "record.json", record)
                    write_json(summary, record)
                    last_saved = total
                if terminal:
                    return
            archive.offer(point, recent=True)
            write_json(status_path, dict(work=work, branches=branches,
                       best_return_cents=best, latest_path_length=length, slots=len(archive.points),
                       historical_peak_cents=max(historical_peak,best),
                       distinct_screen_codes=len(model.visits),
                       steps_per_second=(work - initial_work) / (time.monotonic() - started)))
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69731)
    parser.add_argument("--steps", type=int, default=5000000)
    parser.add_argument("--unrewarded-screen-slots", type=int, default=0,
                        help="Retain this many different RGB states even when the last action had zero reward")
    args = parser.parse_args()
    if not 0 <= args.unrewarded_screen_slots <= 1024:
        parser.error("unrewarded screen slots must be between zero and 1024")
    train(args.logdir.resolve(), args.seed, args.steps, args.unrewarded_screen_slots)
