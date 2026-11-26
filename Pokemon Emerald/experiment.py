"""Train Emerald policies with real PPO, an optional archive and world model, or evaluate from boot."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw

from emerald import BUTTONS, MAPS, PHASES, REPEAT, ROM_SHA1, verify_rom
from pool_gba import Pool


def location(key):
    group, num = key[1:3]
    if 0 <= group < len(MAPS) and 0 <= num < len(MAPS[group]):
        return MAPS[group][num]
    return None


def progress(k):
    return k[5] + 3 * k[6] + 8 * k[7] + 12 * (k[8] & 255).bit_count() + 50 * (k[8] >> 8)


def summarize(keys):
    keys = list(keys)
    return dict(phases=sorted({PHASES[k[0]] for k in keys}),
                maps=sorted({name for k in keys if (name := location(k))}),
                intro=max((k[5] for k in keys), default=0),
                clock=any(k[6] for k in keys), starter=any(k[7] for k in keys),
                badges=max(((k[8] & 255).bit_count() for k in keys), default=0),
                hall_of_fame=any(k[8] & 256 for k in keys))


def story_flags(key):
    if key[6]:
        yield "clock"
    if key[7]:
        yield "starter"
    if key[8] & 255:
        yield "first_badge"
    for i in range(8):
        if key[8] & (1 << i):
            yield f"badge_{i+1}"
    if key[8] & 256:
        yield "hall_of_fame"


def gallery(archive, outfile):
    # One actual emulator frame per map and pre-game phase.
    examples = {}
    for k, cell in archive.items():
        label = location(k) or PHASES[k[0]]
        if label not in examples or cell["image"].std() > examples[label].std():
            examples[label] = cell["image"]
    examples = list(examples.items())
    if not examples:
        return
    cols = min(4, len(examples))
    width, height = 216, 160
    out = Image.new("RGB", (width * cols, height * ((len(examples) + cols - 1) // cols)), "#18202d")
    draw = ImageDraw.Draw(out)
    for i, (label, data) in enumerate(examples):
        x, y = i % cols * width, i // cols * height
        out.paste(Image.fromarray(data).resize((198, 132), Image.Resampling.NEAREST), (x+8, y+4))
        draw.text((x+8, y+139), label.replace("_", " "), fill="white")
    out.save(outfile)


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def moment_notes(directory, events, policy):
    rows = ["# Fresh-game highlights", "",
            f"Input source was {policy}. Each row records the first appearance across the evaluated games. Trial numbers start at zero and match the evaluation JSON.", "",
            "| Game event or location | Trial | Decision within trial | Last button | Native frame |",
            "|---|---:|---:|---|---|"]
    for event, item in sorted(events.items(), key=lambda pair: pair[1]["decision"]):
        rows.append(f"| {event} | {item['trial']} | {item['decision']:,} | {item['button']} | [Still]({item['file']}) |")
    rows.extend(["", "The last button is the input held during the observation interval. "
                 "An earlier input may have initiated a menu or a scripted transition. "
                 "[Event metadata](moments.json) · [Recorded inputs](buttons.npz)"])
    (directory / "HIGHLIGHTS.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


class Run:
    def __init__(self, pool, cfg):
        self.pool, self.cfg = pool, cfg
        self.rng = random.Random(cfg.seed)
        self.archive = {}
        self.visited = set()
        self.events = Counter()
        self.first = {}
        self.milestones = {}
        self.obs = list(pool.initial)
        self.episodes = [self.episode(k) for _, k in self.obs]
        self.step_count = 0
        self.capture_dir = None
        self.known_events = set()
        self.parent_steps = 0

    def capture_discovery(self, i, key, ep, label):
        if self.capture_dir is None or label in self.known_events:
            return
        self.known_events.add(label)
        folder = self.capture_dir / label
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.pool.request(i, "frame")).save(folder / "frame.png")
        (folder / "buttons.bin").write_bytes(ep["path"])
        write_json(folder / "event.json", dict(event=label, map=location(key), descriptor=key,
                   source="archive-assisted training", total_training_steps=self.parent_steps+self.step_count,
                   recorded_buttons=len(ep["path"]), worker=i))

    @staticmethod
    def episode(k, path=b"", goal=None):
        return dict(path=bytearray(path), goal=goal, hit=False, seen={k}, length=0,
                    best=progress(k))

    def reset(self, i):
        root_img, root_key = self.pool.initial[i]
        img, key, state, path = root_img, root_key, None, b""
        goal = None
        if self.cfg.mode in ("archive", "world") and self.archive:
            keys = list(self.archive)
            if self.rng.random() < .5:
                weights = [(1 + min(8, progress(k))) / (1 + self.archive[k]["starts"])**.5
                           for k in keys]
                key = self.rng.choices(keys, weights=weights)[0]
                cell = self.archive[key]
                cell["starts"] += 1
                img, state, path = cell["image"], cell["state"], cell["path"]
                self.events["archive_resets"] += 1
            choices = [k for k in keys if k != key]
            if choices and self.rng.random() < .65:
                weights = []
                for k in choices:
                    c = self.archive[k]
                    p = (1 + c["hits"]) / (2 + c["trials"])
                    weights.append(.1 + p * (1 - p))
                goal = self.rng.choices(choices, weights=weights)[0]
                self.events["goals_assigned"] += 1
        if state is None:
            self.events["fresh_resets"] += 1
        if self.pool.request(i, "load", state) != key:
            raise RuntimeError("Restored checkpoint differs from archived descriptor")
        self.obs[i] = (img, key)
        self.episodes[i] = self.episode(key, path, goal)

    def observe(self, i, result, action):
        image, key = result
        ep = self.episodes[i]
        ep["path"].append(action)
        ep["length"] += 1
        self.visited.add(key)
        mile = location(key)
        if mile:
            if mile not in self.first:
                self.first[mile] = self.step_count
                self.capture_discovery(i, key, ep, mile)
        for label in story_flags(key):
            if label not in self.milestones:
                self.milestones.setdefault(label, self.step_count)
                self.capture_discovery(i, key, ep, label)
        novelty = key not in ep["seen"]
        ep["seen"].add(key)
        rank = progress(key)
        reward = .25 * novelty + max(0, rank - ep["best"])
        ep["best"] = max(rank, ep["best"])
        if self.cfg.mode in ("archive", "world"):
            old = self.archive.get(key)
            if old is None or len(ep["path"]) < len(old["path"]):
                cell = dict(state=self.pool.request(i, "save"), image=image.copy(),
                            path=bytes(ep["path"]), starts=0, hits=0, trials=0)
                if old is not None:
                    for field in ("starts", "hits", "trials"):
                        cell[field] = old[field]
                    self.events["shorter_paths"] += 1
                self.archive[key] = cell
        if ep["goal"] == key and not ep["hit"]:
            ep["hit"] = True
            reward += 2.
            self.events["goals_reached"] += 1
        self.obs[i] = result
        ended = ep["length"] >= self.cfg.episode or bool(key[8] & 256)
        if ended:
            if ep["goal"] is not None:
                cell = self.archive[ep["goal"]]
                cell["trials"] += 1
                cell["hits"] += int(ep["hit"])
            self.reset(i)
        return reward, ended


def train(cfg, pool, net, device):
    import torch
    from policy import tensors, update
    run = Run(pool, cfg)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    world = None
    if cfg.mode == "world":
        from world_model import WorldLearner
        world = WorldLearner(cfg.seed, device, cfg.imagination_update or "full-bptt")
    cfg.out.mkdir(parents=True, exist_ok=True)
    path = cfg.out / "checkpoint.pt"
    if path.exists() or (cfg.out / "metrics.jsonl").exists():
        raise FileExistsError(f"Choose a new run directory: {cfg.out}")
    core_sha = hashlib.sha256(cfg.core.read_bytes()).hexdigest()
    parent_steps = 0
    initial_policy_sha256 = None
    if cfg.weights:
        net.load_state_dict(torch.load(cfg.weights, weights_only=True, map_location=device))
        initial_policy_sha256 = hashlib.sha256(cfg.weights.read_bytes()).hexdigest()
    if cfg.restart_from:
        parent = torch.load(cfg.restart_from, map_location=device, weights_only=False)
        spec = parent["spec"]
        if (spec["rom_sha1"], spec["core_sha256"], spec["mode"], spec.get("backend", "mgba")) != (ROM_SHA1, core_sha, cfg.mode, cfg.backend):
            raise ValueError("Parent run must use the same ROM, emulator build and training mode")
        net.load_state_dict(parent["policy"])
        optimizer.load_state_dict(parent["optimizer"])
        run.archive = parent["archive"]
        parent_steps = parent["steps"]
        if world is not None:
            if cfg.imagination_update is None:
                world.actor_update = parent["world"].get("actor_update", "root")
            world.restore(parent["world"])
        del parent
    if cfg.train_seconds is not None:
        run.capture_dir = cfg.out / "discoveries"
        run.parent_steps = parent_steps
        for key in run.archive:
            run.known_events.add(location(key))
            run.known_events.update(story_flags(key))
    active_seconds, collect_seconds, optimize_seconds = 0., 0., 0.
    actor_seconds, emulation_seconds = 0., 0.
    model_seconds = 0.
    total_updates = cfg.steps // (cfg.envs * cfg.horizon)
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    next_save, next_snapshot = cfg.checkpoint_seconds, cfg.snapshot_seconds
    for u in (itertools.count(1) if cfg.train_seconds is not None else range(1, total_updates + 1)):
        tic = time.perf_counter()
        batch = [[] for _ in range(7)]
        successors, remaining = [], []
        for _ in range(cfg.horizon):
            before_actor = time.perf_counter()
            with torch.no_grad():
                img, ram = tensors(run.obs, [ep["goal"] for ep in run.episodes], device)
                dist, values = net(img, ram)
                actions = dist.sample()
                lp = dist.log_prob(actions)
                # Copying actions back to Python synchronizes this CUDA forward pass.
                ids = actions.tolist()
            before_emulator = time.perf_counter()
            result = pool.step(ids)
            if world is not None:
                # observe() can restore a different game state at a boundary.
                successors.extend(result)
                remaining.extend(cfg.episode - ep["length"] for ep in run.episodes)
            after_emulator = time.perf_counter()
            actor_seconds += before_emulator - before_actor
            emulation_seconds += after_emulator - before_emulator
            rewards, ends = [], []
            for i, (obs, a) in enumerate(zip(result, ids)):
                run.step_count += 1
                r, end = run.observe(i, obs, a)
                rewards.append(r)
                ends.append(float(end))
            for col, value in zip(batch, (img, ram, actions, lp, values,
                                          torch.tensor(rewards, device=device),
                                          torch.tensor(ends, device=device))):
                col.append(value)
        with torch.no_grad():
            _, tail = net(*tensors(run.obs, [ep["goal"] for ep in run.episodes], device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        split = time.perf_counter()
        loss = update(net, optimizer, batch, tail)
        ppo_finished = time.perf_counter()
        world_metrics = {}
        if world is not None:
            real_images, real_ram, real_actions = (torch.cat(v) for v in batch[:3])
            next_images, next_ram = tensors(successors, [None] * len(successors), device)
            recent = world.record(real_images, real_ram, real_actions, next_images, next_ram, torch.cat(batch[5]))
            model_loss = world.fit(recent, torch.stack(batch[6]))
            check_momentum = cfg.train_seconds is None and u == total_updates and world.actor_update == "root" and world.replay.size >= 2048
            if check_momentum:
                from copy import deepcopy
                from policy import actor_weights
                base_weights = actor_weights(net)
                shadow, shadow_opt = deepcopy((net, optimizer))
                for name, parameter in shadow.named_parameters():
                    parameter.grad = None if name.startswith("critic.") else torch.zeros_like(parameter)
                shadow_opt.step()
                zero_step = actor_weights(shadow) - base_weights
                del shadow, shadow_opt
            norm, kl = world.improve(net, optimizer, real_images, real_ram, torch.tensor(remaining, device=device))
            if check_momentum and norm > 0:
                actual_step = actor_weights(net) - base_weights
                write_json(cfg.out / "optimizer-control.json", dict(training_steps=parent_steps + run.step_count,
                    zero_to_actual_norm=(zero_step.norm()/actual_step.norm()).item(),
                    step_cosine=torch.nn.functional.cosine_similarity(zero_step, actual_step, dim=0).item(),
                    relative_difference=((actual_step - zero_step).norm()/actual_step.norm()).item()))
            world_metrics = dict(world_loss=model_loss, imagination_actor_grad_norm=norm,
                                 imagination_actor_kl=kl, imagined_steps=world.imagined_steps,
                                 dynamics_updates=world.model_updates, real_replay_size=world.replay.size)
        end = time.perf_counter()
        collect_seconds += split - tic
        optimize_seconds += ppo_finished - split
        model_seconds += end - ppo_finished
        active_seconds += end - tic
        row = dict(update=u, steps=run.step_count, cells=len(run.visited),
                   archive_cells=len(run.archive), ppo_loss=round(loss, 5),
                   active_steps_per_second=round(run.step_count/active_seconds, 1),
                   **world_metrics, **dict(run.events))
        elapsed = time.monotonic() - started
        if cfg.train_seconds is not None:
            row["training_elapsed_seconds"] = round(elapsed, 3)
            row["training_started_utc"] = started_utc
            row["total_training_steps"] = parent_steps + run.step_count
        done = elapsed >= cfg.train_seconds if cfg.train_seconds is not None else u == total_updates
        snapshot_due = cfg.snapshot_seconds is not None and (elapsed >= next_snapshot or done)
        save_due = elapsed >= next_save if cfg.checkpoint_seconds is not None else u % 8 == 0
        with (cfg.out / "metrics.jsonl").open("a", encoding="utf-8") as f:
            print(json.dumps(row), file=f)
        if u == 1 or save_due or snapshot_due or done:
            print(json.dumps(row), flush=True)
            spec = dict(rom_sha1=ROM_SHA1, core_sha256=core_sha, backend=cfg.backend, seed=cfg.seed,
                        parent_steps=parent_steps, total_training_steps=parent_steps + run.step_count,
                        initial_policy_sha256=initial_policy_sha256,
                        mode=cfg.mode, envs=cfg.envs, episode=cfg.episode, horizon=cfg.horizon,
                        repeat=REPEAT, gpu=torch.cuda.get_device_name() if device.type == "cuda" else "CPU")
            if world is not None:
                spec["imagination_update"] = world.actor_update
            tmp = path.with_suffix(".tmp")
            ckpt = dict(policy=net.state_dict(), optimizer=optimizer.state_dict(),
                        archive=run.archive, spec=spec, steps=parent_steps + run.step_count)
            if world is not None:
                ckpt["world"] = world.checkpoint()
            torch.save(ckpt, tmp)
            del ckpt
            tmp.replace(path)
            # A lightweight, weights-only artifact can be evaluated independently.
            tmp = cfg.out / "policy.tmp"
            torch.save(net.state_dict(), tmp)
            tmp.replace(cfg.out / "policy.pt")
            report = dict(row | spec | summarize(run.visited),
                       parameters=sum(p.numel() for p in net.parameters()),
                       first_map_step=run.first, first_milestone_step=run.milestones,
                       collect_seconds=round(collect_seconds, 2),
                       actor_call_seconds=round(actor_seconds, 2),
                       emulator_step_seconds=round(emulation_seconds, 2),
                       ppo_update_seconds=round(optimize_seconds, 2),
                       world_update_seconds=round(model_seconds, 2),
                       peak_gpu_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20, 1)
                       if device.type == "cuda" else 0)
            report["finished"] = done
            write_json(cfg.out / "training.json", report)
            if cfg.checkpoint_seconds is not None:
                next_save = (int(elapsed / cfg.checkpoint_seconds) + 1) * cfg.checkpoint_seconds
            if snapshot_due:
                snap = cfg.out / "snapshots" / f"{report['total_training_steps']:010d}"
                snap.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copyfile(cfg.out / "policy.pt", snap / "policy.pt")
                write_json(snap / "snapshot.json", report)
                next_snapshot = (int(elapsed / cfg.snapshot_seconds)+1)*cfg.snapshot_seconds
            gallery(run.archive, cfg.out / "discoveries.png")
        if done:
            break


def evaluate(cfg, pool, net, device):
    import torch
    from policy import tensors
    cfg.out.mkdir(parents=True, exist_ok=True)
    target = cfg.out / "evaluation.json"
    if target.exists():
        raise FileExistsError(f"Choose a new evaluation directory: {cfg.out}")
    if cfg.weights:
        net.load_state_dict(torch.load(cfg.weights, weights_only=True, map_location=device))
    net.eval()
    rng = np.random.default_rng(cfg.seed)
    obs = list(pool.initial)
    seen = [{k} for _, k in obs]
    first = [{} for _ in obs]
    milestones = [{} for _ in obs]
    verified_events = [{} for _ in obs]
    verified_maps = [{} for _ in obs]
    hall_step = [None] * len(obs)
    capture = cfg.capture_moments if cfg.capture_moments is not None else bool(cfg.mode == "world" and cfg.weights)
    moment_dir = cfg.out / "moments"
    moments = {}
    if capture:
        moment_dir.mkdir(exist_ok=True)
        button_log = np.empty((cfg.episode, len(obs)), dtype=np.uint8)
    for t in range(1, cfg.episode + 1):
        with torch.no_grad():
            if cfg.weights:
                dist, _ = net(*tensors(obs, [None] * len(obs), device))
                actions = dist.sample().tolist()
            else:
                actions = rng.integers(len(BUTTONS), size=len(obs))
        verified_steps = pool.step(actions, verified=True)
        obs = [pair[0] for pair in verified_steps]
        if capture:
            button_log[t - 1] = actions
        for i, (_, k) in enumerate(obs):
            if hall_step[i] is not None:
                continue
            seen[i].add(k)
            if k[8] & 256:
                hall_step[i] = t
            if name := location(k):
                first[i].setdefault(name, t)
            events = [name or PHASES[k[0]]]
            facts = verified_steps[i][1]
            if facts is not None and name:
                verified_maps[i].setdefault(name, t)
                for label, present in (("clock", k[6]), ("met_rival", facts[0]), ("party", facts[1]),
                                       ("route101", name == "Route101")):
                    if present:
                        verified_events[i].setdefault(label, t)
                        if label in ("met_rival", "party", "route101"):
                            events.append(label)
            for label in story_flags(k):
                milestones[i].setdefault(label, t)
                events.append(label)
            if capture:
                for event in events:
                    if event in moments:
                        continue
                    frame = pool.request(i, "frame")
                    Image.fromarray(frame).save(moment_dir / (event + ".png"))
                    moments[event] = dict(trial=i, decision=t, button=BUTTONS[actions[i]],
                                          map=name, descriptor=list(k), file=event + ".png")
    rows = [dict(trial=i, cells=len(s), first_map_decision=first[i], first_milestone_decision=milestones[i], hall_decision=hall_step[i],
                 verified_map_decision=verified_maps[i], verified_event_decision=verified_events[i],
                 **summarize(s)) for i, s in enumerate(seen)]
    report = dict(start="fresh boot", resets=0, backend=cfg.backend, policy="sampled frozen policy" if cfg.weights else "uniform random",
                  policy_sha256=hashlib.sha256(cfg.weights.read_bytes()).hexdigest() if cfg.weights else None,
                  seed=cfg.seed, decision_budget=cfg.episode, trials=rows)
    if capture:
        np.savez_compressed(moment_dir / "buttons.npz", actions=button_log)
        write_json(moment_dir / "moments.json", dict(start=report["start"], policy=report["policy"],
                   policy_sha256=report["policy_sha256"], seed=cfg.seed, buttons=BUTTONS,
                   frames_per_decision=REPEAT, events=moments))
        moment_notes(moment_dir, moments, report["policy"])
    write_json(target, report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "eval"])
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--backend", choices=("mgba", "native"), default="mgba")
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["archive", "ppo", "world"], default="archive")
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16384)
    parser.add_argument("--train-seconds", type=float, help="Train until this elapsed-time budget, finishing the current rollout.")
    parser.add_argument("--checkpoint-seconds", type=float, help="Time between recovery checkpoints.")
    parser.add_argument("--snapshot-seconds", type=float, help="Keep frozen policies at this interval and on completion.")
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--episode", type=int, default=512)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--weights", type=Path, help="Evaluate or initialize from policy.pt. Evaluating without weights selects uniform random.")
    parser.add_argument("--restart-from", type=Path, help="Local training checkpoint to extend in a new output directory.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--capture-moments", action=argparse.BooleanOptionalAction, default=None,
                        help="Save native frames and the complete button history during evaluation. Defaults on for a frozen world policy.")
    parser.add_argument("--imagination-update", choices=["root", "full-bptt"],
                        help="World actor update. New runs use full-bptt. Extensions inherit the parent setting.")
    cfg = parser.parse_args()
    if min(cfg.envs, cfg.episode, cfg.steps, cfg.horizon) < 1:
        parser.error("Budgets and environment count must be positive")
    if any(value is not None and (not np.isfinite(value) or value <= 0 or cfg.command != "train")
           for value in (cfg.train_seconds, cfg.checkpoint_seconds, cfg.snapshot_seconds)):
        parser.error("Time budgets require train and finite, positive seconds")
    if cfg.command == "train" and cfg.train_seconds is None and cfg.steps % (cfg.horizon * cfg.envs):
        parser.error("steps must be divisible by envs × horizon")
    if cfg.restart_from and cfg.weights:
        parser.error("Choose one of --weights or --restart-from")
    if cfg.command == "eval" and cfg.restart_from:
        parser.error("--restart-from applies to train")
    if cfg.imagination_update and (cfg.mode != "world" or cfg.command != "train"):
        parser.error("--imagination-update applies to world training")
    verify_rom(cfg.rom)
    import torch
    from policy import Policy
    torch.set_num_threads(2)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    net = Policy().to(device)
    pool = Pool(cfg.envs, cfg.core, cfg.rom, cfg.backend)
    try:
        (train if cfg.command == "train" else evaluate)(cfg, pool, net, device)
    finally:
        pool.close()


if __name__ == "__main__":
    main()
