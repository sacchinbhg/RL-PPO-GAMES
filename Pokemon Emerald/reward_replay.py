"""Optional sampling experiment on a policy's own upstream Dreamer replay."""
from contextvars import ContextVar
from threading import Lock

import numpy as np


class RewardWindowSampler:
    def __init__(self, replay, context, fraction, seed):
        from embodied.core.selectors import SampleTree
        self.replay = replay
        self.uniform = replay.sampler
        self.weighted = SampleTree(seed=seed)
        self.context = context
        self.fraction = fraction
        self.rng = np.random.default_rng(seed)
        self.lock = Lock()
        self.training = ContextVar("sample_training_replay", default=False)
        self.weighted_draws = 0
        self.uniform_draws = 0

    def __len__(self):
        return len(self.uniform)

    def __setitem__(self, item, stepids):
        self.uniform[item] = stepids
        chunk, index = self.replay.items[item]
        data = self.replay._getseq(chunk, index, keys=["reward"])
        magnitude = float(np.abs(data["reward"][self.context:]).max())
        if magnitude > 0:
            with self.lock:
                self.weighted.insert(item, magnitude)

    def __delitem__(self, item):
        del self.uniform[item]
        with self.lock:
            if item in self.weighted.entries:
                self.weighted.remove(item)

    def __call__(self):
        if self.training.get():
            with self.lock:
                if len(self.weighted) and self.rng.random() < self.fraction:
                    self.weighted_draws += 1
                    return self.weighted.sample()
                self.uniform_draws += 1
        return self.uniform()


def install_reward_sampling(replay, context, fraction, seed):
    """Upstream continues to handle fresh online samples, storage, and eviction."""
    selector = RewardWindowSampler(replay, context, fraction, seed)
    original_sample = replay.sample
    original_stats = replay.stats
    replay.sampler = selector

    def sample(batch, mode="train"):
        token = selector.training.set(mode == "train")
        try:
            return original_sample(batch, mode)
        finally:
            selector.training.reset(token)

    def stats():
        values = original_stats()
        with selector.lock:
            values.update(reward_windows=len(selector.weighted),
                          weighted_draws=selector.weighted_draws, uniform_draws=selector.uniform_draws)
            selector.weighted_draws = selector.uniform_draws = 0
        return values

    replay.sample = sample
    replay.stats = stats
    return replay
