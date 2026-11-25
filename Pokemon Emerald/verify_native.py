"""Replay the upstream control recording, then time identical eight-game worker workloads."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from experiment import location, write_json
from native_emerald import NativeEmerald
from pool_gba import Pool


def reference(cfg, trace, out):
    buttons = {name: 1 << i for i, name in enumerate(
        ["a", "b", "select", "start", "right", "left", "up", "down", "r", "l"])}
    events = defaultdict(list)
    for line in trace.read_text().splitlines():
        parts = line.partition("#")[0].split()
        if parts:
            events[int(parts[0])].append(parts[1:])
    env = NativeEmerald(cfg["native_core"], cfg["rom"])
    try:
        if not env.lib.rl_power_on():
            raise RuntimeError("Cold boot failed")
        held, next_frame = 0, 0
        rows = {}
        for frame, entries in sorted(events.items()):
            env.lib.rl_frames(held, frame - next_frame)
            for entry in entries:
                if entry[0] == "button":
                    mask = buttons[entry[1]]
                    held = held | mask if entry[2] == "on" else held & ~mask
            env.lib.rl_frames(held, 1)
            next_frame = frame + 1
            for entry in entries:
                if entry[0] != "screenshot":
                    continue
                env.lib.rl_render()
                k = env.inspect()
                rows[entry[1]] = dict(frame=frame, map=location(k), phase=k[0],
                                      clock=k[6], met_rival=k[9], route101_state=k[10], party=k[11])
                if entry[1] in {"after-clock", "outside-rival-house", "mudkip-battle"}:
                    Image.fromarray(env.screen).save(out / (entry[1] + ".png"))
        write_json(out / "scripted-replay.json", rows)
        required = (rows["after-clock"]["clock"], rows["outside-rival-house"]["met_rival"],
                    rows["mudkip-battle"]["party"] == 1, rows["mudkip-battle"]["phase"] == 6)
        if not all(required):
            raise AssertionError(f"Reference milestones failed: {required}")
        return rows["mudkip-battle"]
    finally:
        env.close()


def bench(cfg, backend, actions):
    core = cfg["native_core"] if backend == "native" else cfg["core"]
    pool = Pool(actions.shape[1], core, cfg["rom"], backend)
    try:
        clock = time.perf_counter()
        pixel_checks = set()
        for i, row in enumerate(actions):
            obs = pool.step(row)
            if i % 16 == 0:
                pixel_checks.add(hashlib.sha256(obs[0][0].tobytes()).digest())
        elapsed = time.perf_counter() - clock
        image, key = obs[0]
        saved = pool.request(0, "save")
        start = time.perf_counter()
        other = pool.request(1, "load", saved)
        restore = time.perf_counter() - start
        assert other == key and image.shape == (40, 60, 3) and len(pixel_checks) > 2
        expected, actual = [], []
        for action in (1, 4, 0, 2, 6, 5, 7, 1):
            step = pool.step([action] * actions.shape[1])
            for result, history in zip(step[:2], (expected, actual)):
                img, desc = result
                history.append((hashlib.sha256(img.tobytes()).hexdigest(), desc))
        assert expected == actual, "Restored worker diverged from the original"
        return dict(backend=backend, workers=actions.shape[1], real_decisions=int(actions.size),
                    step_seconds=elapsed, decisions_per_second=actions.size / elapsed,
                    restore_seconds=restore, cross_worker_equal_after_eight_steps=True)
    finally:
        pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("local.json"))
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=512)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    args.out.mkdir(parents=True, exist_ok=True)
    reference_result = reference(cfg, args.trace, args.out)
    actions = np.random.default_rng(72625).integers(9, size=(args.batches, 8))
    results = []
    for backend in ("native", "mgba", "mgba", "native"):
        results.append(bench(cfg, backend, actions))
        write_json(args.out / "benchmark.json", dict(scripted_reference=reference_result,
                   samples=results, unit="one action advances 24 frames and returns RGB and state",
                   workload="eight workers; fixed inputs; existing overnight training running concurrently"))
        print(results[-1], flush=True)
