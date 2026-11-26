"""Small Dreamer-inspired sequence learner for original-ROM trajectories and discrete buttons."""
import argparse
import json
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
from gym_data import RUN

GROUPS, CLASSES, MEMORY = 8, 8, 128
LATENT = GROUPS * CLASSES
WIDTH = MEMORY + LATENT
WEIGHTS = (3., 4., 10.)


class Dream(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(20*30*3, 192), nn.LayerNorm(192), nn.SiLU())
        self.recur = nn.GRUCell(LATENT + 9, MEMORY)
        self.prior = nn.Linear(MEMORY, LATENT)
        self.posterior = nn.Linear(MEMORY + 192, LATENT)
        self.decoder = nn.Linear(WIDTH, 10*15*3)
        self.reward = nn.Linear(WIDTH, 4)
        self.events = nn.Linear(WIDTH, 3)
        self.actor = nn.Sequential(nn.Linear(WIDTH, 128), nn.SiLU(), nn.Linear(128, 9))
        self.value = nn.Sequential(nn.Linear(WIDTH, 128), nn.SiLU(), nn.Linear(128, 1))

    def latent(self, logits, sample):
        probs = logits.reshape(-1, GROUPS, CLASSES).softmax(-1)
        choice = Categorical(probs).sample() if sample else probs.argmax(-1)
        hard = F.one_hot(choice, CLASSES).float()
        return (hard + probs - probs.detach()).flatten(1)

    def observed(self, pictures, actions, sample):
        size, length = pictures.shape[:2]
        embeds = self.encoder(pictures.flatten(2))
        h = embeds.new_zeros(size, MEMORY)
        prior_states, post_states, priors, posts = [], [], [], []
        for t in range(length):
            prior = self.prior(h)
            post = self.posterior(torch.cat([h, embeds[:, t]], -1))
            z = self.latent(post, sample)
            post_states.append(torch.cat([h, z], -1))
            prior_states.append(torch.cat([h, self.latent(prior, sample)], -1))
            posts.append(post)
            priors.append(prior)
            if t < length - 1:
                h = self.recur(torch.cat([z, F.one_hot(actions[:, t], 9).float()], -1), h)
        return tuple(torch.stack(values, 1) for values in (post_states, prior_states, posts, priors))

    def imagine(self, start, steps=10):
        state = start.detach()
        logps, values, rewards, entropy = [], [], [], []
        for _ in range(steps):
            dist = Categorical(logits=self.actor(state))
            a = dist.sample()
            logps.append(dist.log_prob(a))
            entropy.append(dist.entropy())
            values.append(self.value(state).squeeze(-1))
            with torch.no_grad():
                h = self.recur(torch.cat([state[:, MEMORY:], F.one_hot(a, 9).float()], -1), state[:, :MEMORY])
                state = torch.cat([h, self.latent(self.prior(h), True)], -1)
                rewards.append((self.reward(state).softmax(-1) * state.new_tensor((0., *WEIGHTS))).sum(-1))
        target = self.value(state).squeeze(-1).detach()
        targets = []
        for i in reversed(range(steps)):
            boot = values[i+1].detach() if i+1 < steps else target
            target = rewards[i] + .99 * (.05 * boot + .95 * target)
            targets.append(target)
        targets = torch.stack(targets[::-1])
        v = torch.stack(values)
        adv = targets - v.detach()
        return (-(torch.stack(logps)*adv / adv.std().clamp_min(.5)).mean()
                - .001*torch.stack(entropy).mean() + F.smooth_l1_loss(v, targets))


def load_runs():
    groups = {"train": [], "test": []}
    for f in sorted((RUN / "experience").glob("*.npz")):
        with np.load(f) as row:
            x = np.ascontiguousarray(row["image"][:, ::2, ::2, :])
            act = row["action"].astype(np.int64)
            events = np.maximum(0, np.diff(row["flags"].astype(np.int16), axis=0))
        groups[f.name.split("-")[0]].append((x, act, events, f.name))
    return groups


def get_batch(rows, starts, length):
    cols = [[], [], []]
    for index, pos in starts:
        x, a, events, _ = rows[index]
        for bucket, item in zip(cols, (x[pos:pos+length+1], a[pos:pos+length], events[pos:pos+length])):
            bucket.append(item)
    x, a, e = [torch.as_tensor(np.stack(v), device="cuda") for v in cols]
    return x.float() / 255., a, e.float()


def logits_kl(x, y):
    x = x.reshape(*x.shape[:2], GROUPS, CLASSES).log_softmax(-1)
    y = y.reshape_as(x).log_softmax(-1)
    return (x.exp()*(x-y)).sum(-1).sum(-1).mean()


@torch.inference_mode()
def test(model, rows, length, auxiliary):
    scores, truth = [], []
    for i, row in enumerate(rows):
        windows = [(i, j) for j in range(0, len(row[1])-length+1, length)]
        for k in range(0, len(windows), 32):
            x, a, e = get_batch(rows, windows[k:k+32], length)
            _, future, _, _ = model.observed(x, a, False)
            z = future[:, 1:]
            prob = model.events(z).sigmoid()[:,:,0] if auxiliary else model.reward(z).softmax(-1)[:,:,1]
            scores.extend(prob.flatten().tolist())
            truth.extend(e[:,:,0].flatten().tolist())
    scores, truth = np.array(scores), np.asarray(truth, dtype=bool)
    order = np.argsort(-scores)
    precision = np.cumsum(truth[order]) / np.arange(1, len(order)+1)
    return dict(clock_events=int(truth.sum()), decisions=len(truth),
                clock_average_precision=float(precision[truth[order]].sum()/max(1,truth.sum())),
                clock_base_rate=float(truth.mean()),
                clock_detected_at_half=int((truth & (scores >= .5)).sum()),
                false_alarms_at_half=int((~truth & (scores >= .5)).sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=700)
    args = p.parse_args()
    data = load_runs()
    rows = data["train"]
    length = 16
    rare = [(i, pos) for i, row in enumerate(rows) for pos in np.flatnonzero(row[2].any(axis=1))]
    if not rare or not data["test"]:
        raise RuntimeError("Need both autonomous training events and untouched evaluation games")
    output = RUN / "dream"
    output.mkdir(exist_ok=True)
    report = {"train_files":[r[3] for r in rows], "test_files":[r[3] for r in data["test"]], "runs":{}}
    for auxiliary in (False, True):
        torch.manual_seed(71299)
        rng = np.random.default_rng(71299)
        model = Dream().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.00045)
        name = "event-aware" if auxiliary else "standard"
        started = time.monotonic()
        for step in range(1, args.updates+1):
            starts = []
            for i in range(32):
                if i < 8:
                    idx, event = rare[rng.integers(len(rare))]
                    pos = np.clip(event - rng.integers(1, length), 0, len(rows[idx][1])-length)
                else:
                    idx = rng.integers(len(rows))
                    pos = rng.integers(len(rows[idx][1])-length+1)
                starts.append((idx, pos))
            x, a, e = get_batch(rows, starts, length)
            state, future, post, prior = model.observed(x, a, True)
            target = (e * e.new_tensor([1,2,3])).amax(-1).long()
            reconstruction = F.mse_loss(model.decoder(state).sigmoid(), x[:,:,::2,::2].flatten(2))
            predicted = future[:,1:]
            rew = F.cross_entropy(model.reward(predicted).flatten(0,1), target.flatten())
            loss = 8*reconstruction + rew + .1*logits_kl(post.detach(), prior) + .025*logits_kl(post, prior.detach())
            if auxiliary:
                loss += F.binary_cross_entropy_with_logits(model.events(predicted), e, pos_weight=e.new_tensor([16,24,24]))
            if step > 100:
                loss += .25 * model.imagine(state[:, -1])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            if step % 100 == 0:
                print(name, step, float(loss.detach()), flush=True)
        result = test(model, data["test"], length, auxiliary)
        result.update(updates=step, wall_seconds=round(time.monotonic()-started, 2))
        torch.save(model.state_dict(), output / (name+".pt"))
        report["runs"][name] = result
        (output / "pilot.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(name, result, flush=True)


if __name__ == "__main__":
    main()
