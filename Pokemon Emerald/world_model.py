"""Action-conditioned Emerald dynamics and short, differentiable policy rollouts."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call

from emerald import BUTTONS
from policy import N_FEATURES


PIXELS = 3 * 8 * 12
STATE = PIXELS + N_FEATURES
ACTIONS = len(BUTTONS)
HORIZON = 3


def pack(image, ram):
    """Stable targets shared across training updates. Only current RAM is part of the state."""
    pixels = F.avg_pool2d(image.float() / 255., 5).flatten(1)
    return torch.cat((pixels, ram[:, :N_FEATURES]), dim=-1)


def observation(state, goal):
    image = state[:, :PIXELS].reshape(-1, 3, 8, 12)
    image = F.interpolate(image, size=(40, 60), mode="bilinear", align_corners=False)
    return image * 255., torch.cat((state[:, PIXELS:], goal), dim=-1)


def bound_state(state):
    # RGB, phases and progress bits are bounded. Map indices and coordinates
    # use scale factors and may extend past the unit interval on larger maps.
    return torch.cat((state[:, :PIXELS+8].clamp(0,1),
                      state[:, PIXELS+8:PIXELS+12],
                      state[:, PIXELS+12:].clamp(0,1)), dim=-1)


class Dynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.state = nn.Sequential(nn.Linear(STATE, 192), nn.LayerNorm(192), nn.SiLU())
        self.button = nn.Linear(ACTIONS, 32, bias=False)
        self.joint = nn.Sequential(nn.Linear(224, 192), nn.SiLU(), nn.Linear(192, 128), nn.SiLU())
        self.delta = nn.Linear(128, STATE)
        self.reward = nn.Sequential(nn.Linear(128 + N_FEATURES, 96), nn.SiLU(), nn.Linear(96, 2))
        nn.init.normal_(self.delta.weight, std=.002)
        nn.init.zeros_(self.delta.bias)
        with torch.no_grad():
            self.reward[-1].bias.copy_(torch.tensor([-3., -1.]))
        # Put a two-tile move on a useful numerical scale in the RAM regression.
        self.register_buffer("ram_scale", torch.tensor([1.]*8 + [4., 6., 12., 12., 1., 1., 1.] + [1.]*9))

    def forward(self, state, action, goal):
        joint = self.joint(torch.cat((self.state(state), self.button(action)), dim=-1))
        nxt = state + self.delta(joint)
        event, amount = self.reward(torch.cat((joint, goal), dim=-1)).unbind(-1)
        return nxt, event, F.softplus(amount)

    def step(self, state, action, goal):
        nxt, event, amount = self(state, action, goal)
        # Clip the decoded reward at an opening-game scale during imagination only.
        reward = event.sigmoid() * amount.clamp(max=2.2).expm1()
        return bound_state(nxt), reward

    def losses(self, state, action, goal, target, reward):
        return self.prediction_loss(*self(state, action, goal), target, reward)

    def prediction_loss(self, nxt, event, amount, target, reward):
        image_loss = F.mse_loss(nxt[:, :PIXELS], target[:, :PIXELS])
        ram_loss = ((nxt[:, PIXELS:] - target[:, PIXELS:]) * self.ram_scale).square().mean()
        positive = (reward > 0).float()
        event_loss = F.binary_cross_entropy_with_logits(event, positive)
        magnitude = ((amount - reward.log1p()).square() * positive).sum() / positive.sum().clamp_min(1)
        return 4 * image_loss + 2 * ram_loss + .15 * (event_loss + magnitude)


class Replay:
    """Fixed-size circular buffer of actual pre-reset transitions, independent of the archive."""
    def __init__(self, device, capacity=32768):
        self.data = torch.empty((capacity, 2*STATE + N_FEATURES + 2), dtype=torch.float16, device=device)
        self.size = self.cursor = 0

    def add(self, state, action, goal, target, reward):
        rows = torch.cat((state, action[:, None], goal, target, reward[:, None]), 1)
        capacity = len(self.data)
        rows = rows[-capacity:]
        ids = (torch.arange(len(rows), device=rows.device) + self.cursor) % capacity
        self.data[ids] = rows.to(self.data.dtype)
        self.size = min(capacity, self.size + len(rows))
        self.cursor = (self.cursor + len(rows)) % capacity

    def sample(self, generator, n):
        ids = torch.randint(self.size, (n,), generator=generator, device=self.data.device)
        state, action, goal, target, reward = self.data[ids].float().split((STATE, 1, N_FEATURES, STATE, 1), dim=1)
        return state, F.one_hot(action[:, 0].long(), ACTIONS).float(), goal, target, reward[:, 0]


def draw_button(logits, generator):
    u = torch.rand(logits.shape, generator=generator, device=logits.device).clamp_(1e-6, 1-1e-6)
    soft = (logits - (-u.log()).log()).softmax(-1)
    hard = F.one_hot(soft.argmax(-1), ACTIONS).to(soft.dtype)
    return hard + (soft - soft.detach())


def window_starts(ends):
    """Flat time-major starts whose observed future stays within one worker episode."""
    if len(ends) < HORIZON:
        return torch.empty(0, device=ends.device, dtype=torch.long)
    windows = ends.unfold(0, HORIZON, 1)
    return torch.where(~windows[:, :, :-1].bool().any(-1).flatten())[0]


class WorldLearner:
    def __init__(self, seed, device, actor_update="root"):
        # The extra model must not consume the actor's exploration RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 43)
            self.model = Dynamics().to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001)
        self.replay = Replay(device)
        self.rng = torch.Generator(device=device).manual_seed(seed + 1301)
        self.imagined_steps = 0
        self.model_updates = 0
        self.actor_update = actor_update
        self.imagined_optimizer = None
        self._saved_actor_optimizer = None

    def record(self, images, ram, actions, next_images, next_ram, rewards):
        state, target = pack(images, ram), pack(next_images, next_ram)
        goal = ram[:, N_FEATURES:]
        self.replay.add(state, actions, goal, target, rewards)
        return state, F.one_hot(actions, ACTIONS).float(), goal, target, rewards

    def fit(self, recent=None, ends=None, updates=32):
        self.model.requires_grad_(True)
        losses = []
        starts = window_starts(ends) if recent is not None else ()
        for j in range(updates):
            if len(starts) and j % 4 == 3:
                states, actions, goals, targets, rewards = recent
                ids = starts[torch.randint(len(starts), (128,), generator=self.rng, device=starts.device)]
                state, goal = states[ids], goals[ids]
                loss = 0.
                for t in range(HORIZON):
                    at = ids + t * ends.shape[1]
                    nxt, event, amount = self.model(state, actions[at], goal)
                    loss = loss + self.model.prediction_loss(nxt, event, amount, targets[at], rewards[at])
                    state = bound_state(nxt)
                loss = loss / HORIZON
            else:
                loss = self.model.losses(*self.replay.sample(self.rng, 192))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 2., error_if_nonfinite=True)
            self.optimizer.step()
            losses.append(loss.detach())
        self.model_updates += updates
        self.optimizer.zero_grad(set_to_none=True)
        self.model.requires_grad_(False)
        return torch.stack(losses).mean().item()

    def improve(self, policy, optimizer, image, ram, remaining):
        """Optimize button choices through three model transitions and a fixed critic.

        Stop at the known episode deadline. The root variant freezes future policy
        parameters. Full BPTT trains every action with an independent optimizer.
        Both keep bootstrap parameters fixed while taking derivatives through states.
        """
        eligible = torch.where(remaining >= HORIZON)[0]
        if not len(eligible) or self.replay.size < 2048:
            return 0., 0.
        n = min(128, len(eligible))
        ids = eligible[torch.randint(len(eligible), (n,), generator=self.rng, device=image.device)]
        image, ram = image[ids], ram[ids]
        goal = ram[:, N_FEATURES:]
        state = pack(image, ram)
        dist, _ = policy(image, ram)
        root_logits = dist.logits
        fixed = {name: p.detach() for name, p in policy.named_parameters()}
        if self.actor_update == "full-bptt":
            if self.imagined_optimizer is None:
                actor_params = [p for name, p in policy.named_parameters() if not name.startswith("critic.")]
                self.imagined_optimizer = torch.optim.Adam(actor_params, lr=2e-4, betas=(.9, .95))
                if self._saved_actor_optimizer is not None:
                    self.imagined_optimizer.load_state_dict(self._saved_actor_optimizer)
                    self._saved_actor_optimizer = None
            optimizer = self.imagined_optimizer
        survival = torch.ones(n, device=image.device)
        imagined_return = torch.zeros_like(survival)
        for t in range(HORIZON):
            action = draw_button(dist.logits, self.rng)
            state, reward = self.model.step(state, action, goal)
            imagined_return = imagined_return + (.99**t) * survival * reward
            survival = survival * (1. - state[:, -1])  # Hall of Fame bit
            future = observation(state, goal)
            if self.actor_update == "full-bptt" and t + 1 < HORIZON:
                dist, _ = policy(*future)
            else:
                dist, value = functional_call(policy, fixed, future)
        bootstrap = (remaining[ids] > HORIZON) * (5 * torch.tanh(value / 5))
        imagined_return = imagined_return + .99**HORIZON * survival * bootstrap
        loss = -imagined_return.mean()
        policy.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(policy.parameters(), .25, error_if_nonfinite=True)
        optimizer.step()
        self.imagined_steps += n * HORIZON
        with torch.no_grad():
            after, _ = policy(image, ram)
            kl = (root_logits.detach().exp() * (root_logits.detach() - after.logits)).sum(-1).mean()
        return norm.item(), kl.item()

    def checkpoint(self):
        actor_optimizer = (self.imagined_optimizer.state_dict() if self.imagined_optimizer is not None
                           else self._saved_actor_optimizer)
        return dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                    actor_update=self.actor_update, actor_optimizer=actor_optimizer,
                    replay=self.replay.data[:self.replay.size].clone(), cursor=self.replay.cursor,
                    rng=self.rng.get_state(), imagined_steps=self.imagined_steps, model_updates=self.model_updates)

    def restore(self, saved):
        self.model.load_state_dict(saved["model"])
        self.optimizer.load_state_dict(saved["optimizer"])
        rows = saved["replay"]
        self.replay.data[:len(rows)].copy_(rows)
        self.replay.size, self.replay.cursor = len(rows), saved["cursor"]
        self.rng.set_state(saved["rng"].cpu())
        self.imagined_steps, self.model_updates = saved["imagined_steps"], saved["model_updates"]
        if self.actor_update == saved.get("actor_update", "root"):
            self._saved_actor_optimizer = saved.get("actor_optimizer")
