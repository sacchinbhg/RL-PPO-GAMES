"""Optional sampling experiment on a policy's own upstream Dreamer replay."""
from contextvars import ContextVar
import hashlib
from threading import Lock

import numpy as np


class RetainedRewardWindows:
    """Keep up to eight own sequences per power-of-two scalar reward band."""

    def __init__(self, seed):
        self.seed = str(seed).encode()
        self.bands = {}
        self.rng = np.random.default_rng(seed + 1)

    def add(self, data):
        magnitude = abs(float(data["reward"][-1]))
        if magnitude < 1:
            return
        band = min(7, int(np.floor(np.log2(magnitude))))
        windows = self.bands.setdefault(band, {})
        stepid = data["stepid"][-1].tobytes()
        rank = hashlib.blake2b(self.seed + stepid, digest_size=16).digest()
        if rank in windows or (len(windows) == 8 and rank >= max(windows)):
            return
        if len(windows) == 8:
            del windows[max(windows)]
        windows[rank] = {k: np.array(v, copy=True) for k, v in data.items()}

    def windows(self):
        return [window for band in sorted(self.bands)
                for _, window in sorted(self.bands[band].items())]

    def sample(self):
        windows = self.windows()
        weights = np.array([abs(float(x["reward"][-1])) for x in windows])
        chosen = windows[self.rng.choice(len(windows), p=weights / weights.sum())]
        return {k: [v] for k, v in chosen.items()}


class RewardWindowSampler:
    def __init__(self, replay, context, fraction, seed, retain=False):
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
        self.memory = RetainedRewardWindows(seed) if retain else None
        self.large_items = set()
        self.memory_choice = ContextVar("retained_reward_choice", default=None)
        self.retained_draws = 0

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
                if self.memory is not None and magnitude >= 1:
                    self.large_items.add(item)
                    if abs(float(data["reward"][-1])) >= 1:
                        self.memory.add(self.replay._getseq(chunk, index))

    def __delitem__(self, item):
        del self.uniform[item]
        with self.lock:
            self.large_items.discard(item)
            if item in self.weighted.entries:
                self.weighted.remove(item)

    def __call__(self):
        if self.training.get():
            with self.lock:
                use_memory = self.memory is not None and bool(self.memory.bands) and not self.large_items
                if self.memory is not None:
                    self.memory_choice.set(None)
                if (len(self.weighted) or use_memory) and self.rng.random() < self.fraction:
                    self.weighted_draws += 1
                    if use_memory:
                        self.memory_choice.set(self.memory.sample())
                        return self.uniform()
                    return self.weighted.sample()
                self.uniform_draws += 1
        return self.uniform()


def install_reward_sampling(replay, context, fraction, seed, retain=False):
    """Upstream continues to handle fresh online samples, storage, and eviction."""
    selector = RewardWindowSampler(replay, context, fraction, seed, retain)
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
            if selector.memory is not None:
                values.update(retained_windows=len(selector.memory.windows()),
                              live_large_windows=len(selector.large_items), retained_draws=selector.retained_draws)
                selector.retained_draws = 0
            selector.weighted_draws = selector.uniform_draws = 0
        return values

    replay.sample = sample
    replay.stats = stats
    if retain:
        _install_retained_windows(replay, selector)
    return replay


def _install_retained_windows(replay, selector):
    original_one, original_save, original_load = replay._sample, replay.save, replay.load

    def one(mode):
        token = selector.memory_choice.set(None)
        try:
            seq, online = original_one(mode)
            chosen = selector.memory_choice.get()
            if mode == "train" and not online and chosen is not None:
                with selector.lock:
                    selector.retained_draws += 1
                return chosen, False
            return seq, online
        finally:
            selector.memory_choice.reset(token)

    def save():
        upstream = original_save()
        with selector.lock:
            return dict(upstream=upstream, retained_v1=selector.memory.windows(),
                        retained_rng=selector.memory.rng.bit_generator.state)

    def load(data=None, *args, **kwargs):
        if isinstance(data, dict) and "retained_v1" in data:
            with selector.lock:
                selector.memory.bands.clear()
                for window in data["retained_v1"]:
                    selector.memory.add(window)
                selector.memory.rng.bit_generator.state = data["retained_rng"]
            data = data["upstream"]
        return original_load(data, *args, **kwargs)

    replay._sample, replay.save, replay.load = one, save, load
