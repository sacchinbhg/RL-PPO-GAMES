"""Collect contiguous, labeled original-ROM experience from the project's autonomous policies."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from emerald import Emerald

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "stone-badge-01"


def collect(job):
    folder, name, trial, budget = job
    dest = RUN / "experience" / (name + f"-{trial}.npz")
    if dest.exists():
        return
    cfg = json.loads((ROOT / "local.json").read_text(encoding="utf-8-sig"))
    env = Emerald(cfg["core"], cfg["rom"])
    with np.load(folder / "moments/buttons.npz") as source:
        actions = source["actions"][:budget, trial].astype(np.uint8)
    images = [env.boot_image]
    flags = [np.zeros(3, dtype=np.uint8)]
    try:
        for button in actions:
            frame, key = env.step(int(button))
            images.append(frame)
            facts = env.verified_story(key)
            achieved = flags[-1].copy()
            if facts is not None:
                achieved |= np.array([key[6], *facts], dtype=np.uint8)
            flags.append(achieved)
        np.savez_compressed(dest, image=np.asarray(images), action=actions, flags=np.asarray(flags))
        print(name, trial, len(actions), flags[-1].tolist(), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    (RUN / "experience").mkdir(parents=True, exist_ok=True)
    groups = [
        ("runs/world-12h-20261124/evaluations/step-6009856", "train-rival", 4, 1050),
        ("runs/native-night-20261124/pilot-transfer", "train-pilot", 4, 1580),
        ("runs/world-12h-20261124/evaluations/final", "train-stalled", 4, 650),
        ("runs/world-12h-20261124/evaluations/step-9171456", "test-rival", 4, 1280),
        ("runs/native-night-20261124/transfer", "test-stalled", 4, 650),
    ]
    jobs = [(ROOT / folder, name, t, budget) for folder, name, count, budget in groups for t in range(count)]
    with ProcessPoolExecutor(max_workers=5) as pool:
        list(pool.map(collect, jobs))
