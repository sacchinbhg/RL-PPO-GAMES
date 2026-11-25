"""Test saved world models on independent random Emerald games and same-state button forks."""

import argparse
import json
from pathlib import Path

import numpy as np

from emerald import BUTTONS
from experiment import progress, write_json
from pool_gba import Pool


def encode_obs(items):
    from policy import encode
    images, keys = zip(*items)
    # Independent implementation of the learner's 5-by-5 RGB averaging.
    p = np.stack(images).reshape(len(items), 8,5,12,5,3).mean(axis=(2,4), dtype=np.float32)
    p = p.transpose(0,3,1,2).reshape(len(items), -1) / 255.
    return np.concatenate((p, np.array([encode(k) for k in keys], np.float32)), axis=-1)


def collect(pool, steps, seed):
    obs = list(pool.initial)
    n = len(obs)
    rng = np.random.default_rng(seed)
    seen = [{k} for _,k in obs]
    best = [progress(k) for _,k in obs]
    states = [encode_obs(obs)]
    actions, rewards, fork_starts, fork_results = [], [], [], []
    reproduced = 0
    for t in range(steps):
        fork = t % 64 == 32
        if fork:
            saved = [pool.request(i, "save") for i in range(n)]
            fork_starts.append(states[-1])
            targets = []
            for action in range(len(BUTTONS)):
                for i, snap in enumerate(saved):
                    if pool.request(i, "load", snap) != obs[i][1]:
                        raise AssertionError("Descriptor changed across state restore")
                targets.append(encode_obs(pool.step([action]*n)))
            fork_results.append(np.stack(targets, axis=1))
            for i, snap in enumerate(saved):
                pool.request(i, "load", snap)
        a = rng.integers(len(BUTTONS), size=n)
        obs = pool.step(a)
        packed = encode_obs(obs)
        if fork:
            np.testing.assert_array_equal(packed, fork_results[-1][np.arange(n), a])
            reproduced += n
        rs = []
        for i, (_,k) in enumerate(obs):
            score = progress(k)
            rs.append(.25 * (k not in seen[i]) + max(0, score - best[i]))
            best[i] = max(best[i], score)
            seen[i].add(k)
        actions.append(a)
        rewards.append(rs)
        states.append(packed)
    return [np.stack(states), np.stack(actions), np.array(rewards, np.float32),
            np.concatenate(fork_starts), np.concatenate(fork_results), reproduced]


def audit(checkpoint, arrays, device):
    import torch
    from torch.nn import functional as F
    from policy import N_FEATURES
    from world_model import ACTIONS, PIXELS, Dynamics
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = Dynamics().to(device)
    model.load_state_dict(ckpt["world"]["model"])
    model.eval()
    mean_reward = ckpt["world"]["replay"][:, -1].float().mean().item()
    del ckpt
    s, a, r, starts, targets = [torch.as_tensor(t, device=device) for t in arrays[:5]]
    def advance(x, action):
        goal = torch.zeros(len(x), N_FEATURES, device=device)
        return model.step(x, F.one_hot(action.long(), ACTIONS).float(), goal)
    def err(x, y):
        return (x-y).square().mean().sqrt().item()
    scores = {}
    with torch.inference_mode():
        for h in (1,3):
            count = len(a)-h+1
            state = s[:count].flatten(0,1)
            target = s[h:].flatten(0,1)
            pred = state
            for j in range(h):
                out, rews = zip(*(advance(x, act) for x, act in zip(
                    pred.split(512), a[j:j+count].flatten().split(512))))
                pred = torch.cat(out)
            scores[f"horizon_{h}"] = dict(rgb_rmse=err(pred[:,:PIXELS],target[:,:PIXELS]),
                copy_rgb_rmse=err(state[:,:PIXELS],target[:,:PIXELS]),
                ram_rmse=err(pred[:,PIXELS:],target[:,PIXELS:]),
                copy_ram_rmse=err(state[:,PIXELS:],target[:,PIXELS:]))
            if h == 1:
                scores["reward"] = dict(rmse=err(torch.cat(rews), r.flatten()),
                    train_mean_rmse=err(torch.full_like(r, mean_reward), r),
                    positive_outcomes=int((r>0).sum().item()))
        preds = torch.stack([advance(starts, torch.full((len(starts),), i, device=device))[0]
                             for i in range(ACTIONS)], dim=1)
        diff = targets - targets.mean(1,keepdim=True)
        predicted = preds - preds.mean(1,keepdim=True)
        # An action-blind forecast obtains exactly zero, even if it predicts typical screen changes.
        def explained(actual, est):
            return 1. - ((actual-est).square().sum()/actual.square().sum().clamp_min(1e-12)).item()
        scores["button_forks"] = dict(starts=len(starts), outcomes=len(starts)*ACTIONS,
            informative_starts=int(((targets != targets[:,:1]).reshape(len(starts),-1).any(1)).sum().item()),
            rgb_action_variance_explained=explained(diff[:,:,:PIXELS],predicted[:,:,:PIXELS]),
            ram_action_variance_explained=explained(diff[:,:,PIXELS:],predicted[:,:,PIXELS:]),
            correct_action_rmse=err(preds,targets), rotated_action_rmse=err(preds.roll(1,1),targets))
        # Hindsight supplies only the destination. Pick a button using predicted outcomes,
        # then score it against the real fork. Exclude warps and stationary destinations.
        where = slice(PIXELS+10, PIXELS+12)
        map_id = slice(PIXELS+8, PIXELS+10)
        branch_hits, expected = [], []
        for origin, real, predicted_states in zip(starts.cpu(), targets.cpu(), preds.cpu()):
            if origin[PIXELS+5] != 1 or not torch.all(real[:,map_id] == origin[map_id]):
                continue
            for goal in real[:,where].unique(dim=0):
                if torch.equal(goal,origin[where]):
                    continue
                good = (real[:,where] == goal).all(-1)
                choice = (predicted_states[:,where] - goal).square().sum(-1).argmin()
                branch_hits.append(int(good[choice]))
                expected.append(good.float().mean().item())
        scores["one_button_goal_test"] = dict(targets=len(expected), hits=sum(branch_hits),
            model_rate=sum(branch_hits)/len(expected) if expected else None,
            uniform_expected_rate=sum(expected)/len(expected) if expected else None)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=82631)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu","cuda"))
    args = parser.parse_args()
    if args.steps < 64 or args.envs < 1:
        parser.error("Need at least 64 decisions per worker and one worker")
    if args.out.exists():
        parser.error("Choose a new output file")
    import torch
    torch.set_num_threads(2)
    pool = Pool(args.envs, args.core, args.rom)
    try:
        data = collect(pool,args.steps,args.seed)
    finally:
        pool.close()
    checks = {str(p.parent.name):audit(p,data,args.device) for p in args.checkpoints}
    report = dict(behavior="uniform random from boot", seed=args.seed, trials=args.envs,
                  raw_decisions=args.steps*args.envs, replay_matches=data[-1], models=checks)
    write_json(args.out, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
