"""Collect original-game experience from pixels, buttons, and scalar rewards."""
from collections import Counter
import hashlib
import math

import numpy as np


def screen_key(rgb):
    if rgb.shape != (40, 60, 3):
        raise ValueError(f"Expected a 40 by 60 RGB frame, got {rgb.shape}")
    blocks = rgb.reshape(8, 5, 12, 5, 3).sum(axis=(1, 3), dtype=np.uint16)
    quantized = (blocks // (25 * 32)).astype(np.uint8)
    return hashlib.blake2b(quantized.tobytes(), digest_size=12).digest()


class ScreenExplorer:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.visits = Counter()
        self.transitions = {}
        self.reward_sum = Counter()

    def choose(self, state):
        self.visits[state] += 1
        choices = self.transitions.setdefault(state, {})
        unknown = [button for button in range(9) if button not in choices]
        if unknown:
            return unknown[int(self.rng.integers(len(unknown)))]
        scores = []
        for button in range(9):
            outcomes = choices[button]
            count = sum(outcomes.values())
            novelty = sum(n / math.sqrt(1 + self.visits[key]) for key, n in outcomes.items()) / count
            bonus = 0.25 * math.sqrt(math.log1p(self.visits[state]) / count)
            scores.append(novelty + self.reward_sum[state, button] / count + bonus)
        best = max(scores)
        ties = [button for button, score in enumerate(scores) if score == best]
        return ties[int(self.rng.integers(len(ties)))]

    def observe(self, state, button, following, reward):
        self.transitions[state].setdefault(button, Counter())[following] += 1
        self.reward_sum[state, button] += math.tanh(reward)


def install_screen_collection(agent, count, seed):
    """Change behavior on the first count training workers, including RSSM action history."""
    import jax

    original = agent.policy
    explorers = [None] * count
    previous = [None] * count
    device_buttons = None

    def policy(carry, obs, mode="train"):
        nonlocal device_buttons
        carry, acts, outs = original(carry, obs, mode=mode)
        if mode != "train":
            return carry, acts, outs
        buttons = acts["action"].copy()
        action_history = carry[-1]["action"]
        if device_buttons is None:
            template = action_history[0]
            device_buttons = [jax.device_put(np.asarray(i, dtype=template.dtype), template.sharding)
                              for i in range(9)]
        for worker in range(count):
            screen = screen_key(obs["image"][worker, 12:52, 2:62])
            if obs["is_first"][worker]:
                # The upstream policy counter is checkpointed, including across process restarts.
                explorers[worker] = ScreenExplorer(np.random.SeedSequence([seed, worker, int(agent.n_actions)]))
                previous[worker] = None
            if previous[worker] is not None:
                state, button = previous[worker]
                explorers[worker].observe(state, button, screen, float(obs["reward"][worker]))
            if obs["is_last"][worker]:
                previous[worker] = None
                buttons[worker] = 0
            else:
                buttons[worker] = explorers[worker].choose(screen)
                previous[worker] = (screen, int(buttons[worker]))
            action_history[worker] = device_buttons[int(buttons[worker])]
        return carry, {**acts, "action": buttons}, outs

    agent.policy = policy
    return agent
