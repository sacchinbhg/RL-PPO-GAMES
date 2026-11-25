"""Small RGB and RAM actor-critic suitable for a laptop."""

import numpy as np
import torch
from torch import nn

from emerald import BUTTONS, PHASES


def encode(key):
    stage, group, num, x, y, intro, clock, starter, flags = key
    return [float(stage == i) for i in range(len(PHASES))] + [
        (group + 1) / 40, (num + 1) / 100, x / 64, y / 64,
        min(1., intro / 6), clock, starter,
    ] + [float(flags >> bit & 1) for bit in range(9)]


N_FEATURES = len(encode((0,) * 9))


def actor_weights(policy):
    return torch.cat([p.detach().flatten() for name, p in policy.named_parameters()
                      if not name.startswith("critic.")])


def tensors(obs, goals, device):
    images, keys = zip(*obs)
    ram = [encode(k) + (encode(g) if g is not None else [0.] * N_FEATURES)
           for k, g in zip(keys, goals)]
    return (torch.as_tensor(np.stack(images), device=device).permute(0, 3, 1, 2),
            torch.tensor(ram, device=device))


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(3, 16, 5, 3), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2), nn.ReLU(), nn.Flatten(),
        )
        with torch.no_grad():
            width = self.vision(torch.zeros(1, 3, 40, 60)).shape[1]
        self.trunk = nn.Sequential(nn.Linear(width + 2 * N_FEATURES, 128), nn.ReLU())
        self.actor = nn.Linear(128, len(BUTTONS))
        self.critic = nn.Linear(128, 1)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

    def forward(self, image, features):
        visual = self.vision(image.contiguous().float() / 255.)
        hid = self.trunk(torch.cat((visual, features), -1))
        return torch.distributions.Categorical(logits=self.actor(hid)), self.critic(hid)[:, 0]


def update(net, optimizer, batch, tail, epochs=3):
    image, ram, actions, lp, values, rewards, ends = [torch.stack(v) for v in batch]
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(tail)
    for t in reversed(range(len(rewards))):
        cont = 1. - ends[t]
        delta = rewards[t] + .99 * cont * tail - values[t]
        gae = delta + .99 * .95 * cont * gae
        adv[t] = gae
        tail = values[t]
    targets = (adv + values).flatten()
    adv = adv.flatten()
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    image, ram = image.flatten(0, 1), ram.flatten(0, 1)
    actions, lp = actions.flatten(), lp.flatten()
    losses = []
    for _ in range(epochs):
        for ids in torch.randperm(len(adv), device=adv.device).split(128):
            dist, v = net(image[ids], ram[ids])
            ratio = (dist.log_prob(actions[ids]) - lp[ids]).exp()
            clipped = torch.clamp(ratio, .8, 1.2)
            policy_loss = -torch.minimum(ratio * adv[ids], clipped * adv[ids]).mean()
            value_loss = (v - targets[ids]).square().mean()
            loss = policy_loss + .5 * value_loss - .015 * dist.entropy().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), .5)
            optimizer.step()
            losses.append(loss.detach())
    return float(torch.stack(losses).mean().item())
