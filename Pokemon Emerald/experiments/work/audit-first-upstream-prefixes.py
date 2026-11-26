"""Replay the first saved Dreamer sequences and compare their pixels and rewards."""
import json
import os
import sys
from pathlib import Path

import numpy as np

project = Path(__file__).resolve().parents[1] / "outputs" / "EmeraldRL"
sys.path.insert(0, str(project))
from emerald import Emerald
from pixel_env import Score

train = project / "runs/pixels-only-01/official-wsl/seed-71299-h15/train"
out = train.parent / "highlights/first-training-prefixes"
out.mkdir(parents=True, exist_ok=True)
paths = sorted((train / "replay").glob("*.npz"))
by_id = {p.stem.split("-")[1]: p for p in paths}

for index, root in enumerate(paths[:4]):
    path = root
    chain = []
    rewards = []
    while path:
        chain.append(path)
        with np.load(path) as chunk:
            if len(chain) == 1:
                assert chunk["is_first"][0]
            rewards.extend(chunk["reward"].tolist())
        path = by_id.get(path.stem.split("-")[2])
    milestones = np.flatnonzero(np.asarray(rewards) >= 0.5)
    if not len(milestones):
        continue
    stop = int(milestones[-1])
    g = Emerald(os.environ["EMERALD_CORE"], os.environ["EMERALD_ROM"])
    score, buttons = Score(), bytearray()
    action = None
    try:
        g.load(g.boot)
        for path in chain:
            with np.load(path) as chunk:
                actions, images = chunk["action"], chunk["image"]
                for t, next_action in enumerate(actions):
                    if action is None:
                        pic, reward = g.boot_image, 0.
                    else:
                        pic, key = g.step(action)
                        buttons.append(action)
                        reward, _ = score.read(g, key)
                    assert np.array_equal(pic, images[t, 12:52, 2:62]), (index, len(buttons), "pixels")
                    assert np.float32(reward) == rewards[len(buttons)], (index, len(buttons), "reward")
                    action = int(next_action)
                    if len(buttons) == stop:
                        break
            if len(buttons) == stop:
                break
        label = out / f"stream-{index}"
        label.with_suffix(".bin").write_bytes(buttons)
        evidence = dict(events=score.events, steps=len(buttons), source="training replay prefix",
                        root_chunk=root.name, first_visit_cells=len(score.cells),
                        compared_rgb_frames=len(buttons) + 1)
        label.with_suffix(".json").write_text(json.dumps(evidence, indent=2))
        print(json.dumps(evidence), flush=True)
    finally:
        g.close()
