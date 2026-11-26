"""Expand self-earned training states by testing short action sequences at each newly rewarded screen."""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from reward_branch_env import ForkEnv
from reward_branch_search import visual_code, write_json
from scene_reservoir import appearance
from screen_explorer import ScreenExplorer


@dataclass
class Node:
    screen: bytes
    depth: int
    kind: str
    born: int
    offset: int
    tries: int = 0


class Frontier:
    def __init__(self, env, rng, motions, capacity, sweep=False):
        self.env = env
        self.rng = rng
        self.capacity = capacity
        self.slots = {}
        self.visual_seen = set()
        self.stage = -1
        self.added = Counter()
        self.sweep = sweep
        self.options = [((button, count),) for button in range(9) for count in (1, 2, 4, 8)]
        other = sorted(set(range(9)).difference(motions))
        self.options += [((move, count), (button, 1)) for move in sorted(motions)
                         for count in (1, 2, 4, 8) for button in other]
        rng.shuffle(self.options)
        self.first_buttons = rng.permutation(9).tolist() if sweep else []

    def store(self, image, stage, depth, work, kind):
        if self.sweep and kind == "appearance":
            return False
        if stage < self.stage:
            return False
        if stage > self.stage:
            self.stage = stage
            self.slots.clear()
            self.visual_seen.clear()
        limit = self.capacity if kind == "reward" else max(8, self.capacity // 8)
        first = 0 if kind == "reward" else self.capacity
        if kind == "appearance":
            key = appearance(image)
            if key in self.visual_seen:
                return False
            self.visual_seen.add(key)
        cohort = [i for i in self.slots if first <= i < first + limit]
        if len(cohort) < limit:
            slot = first + len(cohort)
        else:
            sample = self.rng.choice(cohort, min(12, len(cohort)), replace=False)
            slot = max(map(int, sample), key=lambda i: (self.slots[i].tries, -self.slots[i].born))
        self.env.command("save", slot)
        self.slots[slot] = Node(visual_code(image), depth, kind, work, int(self.rng.integers(len(self.options))))
        self.added[kind] += 1
        return True

    def start(self):
        eligible = list(self.slots)
        if not eligible:
            raise RuntimeError("No rewarded training state was available")
        fresh = [i for i in eligible if self.slots[i].born > 0]
        if fresh and self.rng.random() < .5:
            eligible = fresh
        sample = self.rng.choice(eligible, min(12, len(eligible)), replace=False)
        slot = min(map(int, sample), key=lambda i: self.slots[i].tries)
        node = self.slots[slot]
        if self.sweep and node.tries < len(self.first_buttons):
            button = self.first_buttons[(node.offset + node.tries) % len(self.first_buttons)]
            option = ((button, 1),)
        else:
            option = self.options[(node.offset + node.tries) % len(self.options)]
        node.tries += 1
        frame = self.env.command("restore", slot)
        if visual_code(frame) != node.screen:
            raise RuntimeError("Restored pixels disagree with the original frontier")
        return node, frame, option


def run(folder, arm, record, probe, steps, seed, capacity, sweep=False, tile_visits=False,
        movement_triples=False, scroll_model=None):
    arm = arm.resolve()
    settings = json.loads((arm / "settings.json").read_text())
    discovery = json.loads(record.read_text())
    source = (arm / discovery["audit"] / "game.bin").resolve()
    tape = source.read_bytes()
    if (not settings.get("training_forks") or settings.get("receives_saves_from_other_runs")
            or not discovery["training_only"] or not source.is_relative_to(arm / "discoveries")
            or hashlib.sha256(tape).hexdigest() != discovery["tape_sha256"]):
        raise ValueError("Expected a verified self-earned training discovery")
    learned = json.loads(probe.read_text())
    motions = {int(b) for b in learned["learned_actions"]}
    if not learned["informative"] or learned["position_or_map_input"] or not 0 < len(motions) < 9:
        raise ValueError("Expected action-conditioned motion learned from training pixels")
    if min(steps, capacity) < 1 or not 8 <= capacity <= 2048:
        raise ValueError("Invalid simulation budget or storage capacity")
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(seed=seed, training_forks=True, receives_saves_from_other_runs=False,
                  inputs="RGB, scalar reward and termination", source_sha256=discovery["tape_sha256"],
                  motion_training_sha256=learned["training_tape_sha256"], new_simulator_decisions=steps,
                  reward_slots=capacity, appearance_slots=max(8, capacity // 8), frame_repeat=24)
    if sweep:
        config.update(reward_sweep=True, appearance_slots=0, first_pass="try every single button per reward state",
                      model_continuation="one in sixteen branches receives 32 empirical transition-model decisions")
    if tile_visits:
        config.update(visit_reward="0.005 per newly observed individual tile across the training search")
    predictor = None
    if scroll_model:
        from fit_local_motion import ScrollModel
        predictor = ScrollModel(scroll_model)
        if not movement_triples or set(predictor.buttons) != motions:
            raise ValueError("The three-decision movement options and learned motion buttons must agree")
        config["scroll_model_sha256"] = predictor.sha
    if movement_triples:
        config["movement_triples"] = "with probability one half, repeat an eligible continuation movement three times"
    setup = folder / "settings.json"
    if setup.exists() and json.loads(setup.read_text()) != config:
        raise ValueError("Another experiment already owns this directory")
    write_json(setup, config)
    if (folder / "completed.json").exists():
        return
    write_json(folder / "branch-origin.json", dict(parent_record=discovery, source_run=str(arm), seed=seed,
               inherited_training_buttons=discovery["work"], cold_replay_buttons=len(tape)))
    rng = np.random.default_rng(seed)
    behavior = ScreenExplorer(np.random.SeedSequence([seed, 1]))
    env = ForkEnv(folder / "first-maps", shared_visit_reward=True, trainer_audits=folder / "first-trainers",
                  tile_visit_reward=tile_visits)
    try:
        archive = Frontier(env, rng, motions, capacity, sweep)
        frame = env.initial
        state = visual_code(frame)
        stage = warm_total = 0
        for i, button in enumerate(tape, 1):
            frame, cents, done, _ = env.command("step", (button, 1))
            following = visual_code(frame)
            behavior.visits[state] += 1
            behavior.transitions.setdefault(state, {})
            behavior.observe(state, button, following, cents / 100)
            state = following
            stage += int(cents >= 100)
            warm_total += cents
            if cents > 0:
                archive.store(frame, stage, i, 0, "reward")
            if i % 4096 == 0:
                write_json(folder / "initialization.json", dict(cold_replay_buttons=i, expected=len(tape)))
        original_total = warm_total
        if tile_visits:
            env.command("audit", str(folder / "warmup-original"))
            audited = json.loads((folder / "warmup-original/game.json").read_text())
            original_total = audited["return_cents"]
        if done or original_total != discovery["return_cents"] or state.hex() != discovery["screen"]:
            raise ValueError("The self-earned cold-boot history could not be reproduced")
        write_json(folder / "initialization.json", dict(cold_replay_buttons=len(tape), matched_image_and_reward=True,
                   surviving_rewarded_states=len(archive.slots), source_sha256=discovery["tape_sha256"],
                   original_return_cents=original_total, training_return_cents=warm_total))
        work = branches = 0
        high_stage = stage
        best = None
        future_payments = Counter()
        option_counts = Counter()
        option_rng = np.random.default_rng([seed, 809])
        start = time.monotonic()
        while work < steps:
            node, frame, option = archive.start()
            branches += 1
            state, depth, stage = node.screen, node.depth, archive.stage
            previous_stage = high_stage
            # Longer branches let a new or unfamiliar screen unfold into later rewards.
            if sweep:
                explore = 32 if rng.random() < 1/16 else 0
            else:
                explore = 128 if node.kind == "appearance" or rng.random() < .125 else 0
            plan = [button for button, count in option for _ in range(count)]
            progress = 0
            repeated = 0
            while work < steps and (progress < len(plan) or explore > 0):
                if progress < len(plan):
                    button = plan[progress]
                    behavior.visits[state] += 1
                    behavior.transitions.setdefault(state, {})
                else:
                    if repeated:
                        button = repeat_button
                        behavior.visits[state] += 1
                        behavior.transitions.setdefault(state, {})
                        repeated -= 1
                    else:
                        button = behavior.choose(state)
                        if movement_triples and explore >= 3 and work + 3 <= steps and button in motions and option_rng.random() < .5:
                            tie_draw = option_rng.random()
                            option_counts["three_button_choices"] += 1
                            if predictor is not None:
                                predictions = predictor.predictions(frame)
                                choices = predictor.buttons[np.isclose(predictions, predictions.max(), atol=1e-6, rtol=0)]
                                if button not in choices:
                                    button = int(choices[int(tie_draw * len(choices))])
                                    option_counts["changed_first_button"] += 1
                            repeat_button = button
                            repeated = 2
                    explore -= 1
                previous = frame
                frame, cents, done, _ = env.command("step", (button, 1))
                following = visual_code(frame)
                behavior.observe(state, button, following, cents / 100)
                work += 1
                depth += 1
                progress += 1
                stage += int(cents >= 100)
                if cents > 0:
                    future_payments[cents] += 1
                    archive.store(frame, stage, depth, work, "reward")
                elif not sweep and button not in motions and following != state and appearance(previous) != appearance(frame):
                    archive.store(frame, stage, depth, work, "appearance")
                state = following
                if stage > high_stage:
                    high_stage = stage
                    explore = max(explore, 128)
                if done:
                    break
            if done or high_stage > previous_stage or branches == 1 or work // 4096 != (work-progress) // 4096 or work == steps:
                relative = f"discoveries/work-{discovery['work']+work:010d}"
                path = folder / relative
                env.command("audit", str(path))
                audit = json.loads((path / "game.json").read_text())
                saved = dict(work=discovery["work"]+work, branches=discovery["branches"]+branches,
                             return_cents=audit["return_cents"], length=depth, screen=state.hex(), audit=relative,
                             tape_sha256=hashlib.sha256((path / "game.bin").read_bytes()).hexdigest(),
                             training_only=True, terminal=bool(done),
                             distinct_cells_across_training=audit["distinct_cells_across_training"])
                write_json(path / "record.json", saved)
                if best is None or saved["return_cents"] > best["return_cents"] or done:
                    best = saved
                    write_json(folder / "best.json", best)
            if branches % 8 == 0 or done or work == steps:
                status = dict(work=discovery["work"]+work, new_buttons=work, branches=branches, latest_stage=high_stage,
                              best_return_cents=best["return_cents"], active_slots=len(archive.slots),
                              admissions=dict(archive.added), new_reward_counts=dict(future_payments),
                              steps_per_second=work / (time.monotonic()-start))
                if movement_triples:
                    status["movement_options"] = dict(option_counts)
                write_json(folder / "status.json", status)
            if done:
                break
        write_json(folder / "completed.json", dict(best=best, status=status))
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "source", "record", "motion"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--steps", type=int, default=262144)
    parser.add_argument("--capacity", type=int, default=768)
    parser.add_argument("--seed", type=int, default=173741)
    parser.add_argument("--reward-sweep", action="store_true",
                        help="Keep each newly rewarded snapshot and test all nine single buttons before longer options")
    parser.add_argument("--individual-tile-rewards", action="store_true",
                        help="During search, split each 2x2 block's potential 0.02 reward into 0.005 per visited tile")
    parser.add_argument("--movement-triples", action="store_true", help="Test three repeated movement decisions in learned continuations")
    parser.add_argument("--scroll-model", type=Path, help="Rank three-decision movement choices using a fitted local RGB model")
    args = parser.parse_args()
    run(args.output, args.source, args.record, args.motion, args.steps, args.seed, args.capacity,
        args.reward_sweep, args.individual_tile_rewards, args.movement_triples, args.scroll_model)
