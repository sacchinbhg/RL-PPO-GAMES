"""Benchmark the official sequential and parallel drivers with identical button tapes."""
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

project = Path(__file__).resolve().parents[1] / "outputs" / "EmeraldRL"
sys.path[:0] = [str(project), "/opt/emerald-rl/source"]
import elements
import embodied
from pixel_env import PixelPool


class Game:
    def __init__(self):
        self.pool = PixelPool(1)

    @property
    def act_space(self):
        return dict(action=elements.Space(np.int32,(),0,9), reset=elements.Space(bool))

    def step(self, act):
        reset = bool(act["reset"])
        if reset:
            image = self.pool.command(0,"reset")
            reward, done = 0.,False
        else:
            images, rewards, ends = self.pool.step([int(act["action"])])
            image, reward, done = images[0], rewards[0], ends[0]
        padded = np.zeros((64,64,3),np.uint8)
        padded[12:52,2:62] = image
        return dict(image=padded,reward=np.float32(reward),is_first=np.bool_(reset),
                    is_last=np.bool_(done),is_terminal=np.bool_(done))

    def close(self):
        self.pool.close()


def bench(parallel, seed):
    envs = 4
    rng = np.random.default_rng(seed)
    def policy(carry, obs):
        return carry, dict(action=rng.integers(0,9,envs,dtype=np.int32)), {}
    h = [hashlib.sha256() for _ in range(envs)]
    def check(tran, index):
        for key in ("image","reward","is_first","is_last","is_terminal","action"):
            h[index].update(tran[key].tobytes())
    start = time.perf_counter()
    driver = embodied.Driver([Game] * envs, parallel=parallel)
    try:
        driver.on_step(check)
        driver(policy, steps=256)
        warm = time.perf_counter()
        driver(policy, steps=4096)
        end = time.perf_counter()
    finally:
        driver.close()
    return dict(parallel=parallel, seed=seed, measured_transitions=4096,
                steps_per_second=4096/(end-warm), startup_and_warmup_seconds=warm-start,
                worker_hashes=[x.hexdigest() for x in h])


if __name__ == "__main__":
    rows = []
    for seed, order in ((4297,(False,True)), (9817,(True,False))):
        pair = []
        for mode in order:
            r = bench(mode, seed)
            pair.append(r)
            print(seed, mode, r["steps_per_second"], flush=True)
        assert pair[0]["worker_hashes"] == pair[1]["worker_hashes"], "Driver changed gameplay"
        rows.extend(pair)
    output = project / "runs/pixels-only-01/official-wsl/controls/driver-throughput.json"
    output.write_text(json.dumps(rows, indent=2))
