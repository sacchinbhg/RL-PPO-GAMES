"""Measure the contribution of inherited PPO momentum to an imagined actor step."""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from experiment import write_json
from policy import Policy, actor_weights, tensors
from world_model import WorldLearner


def audit(checkpoint, device, seed):
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if data["world"].get("actor_update", "root") != "root":
        raise ValueError("The shared-optimizer control requires a root-update checkpoint")
    rng = np.random.default_rng(seed)
    keys = list(data["archive"])
    selected = rng.choice(len(keys), size=128, replace=True)
    obs = [(data["archive"][keys[i]]["image"], keys[i]) for i in selected]
    image, ram = tensors(obs, [None] * len(obs), device)
    deltas = []
    grads = []
    for zero in (False, True):
        policy = Policy().to(device)
        policy.load_state_dict(data["policy"])
        opt = torch.optim.Adam(policy.parameters(), lr=.0003)
        opt.load_state_dict(deepcopy(data["optimizer"]))
        world = WorldLearner(seed, device)
        world.restore(deepcopy(data["world"]))
        world.model.requires_grad_(False)
        if zero:
            for p in policy.parameters():
                p.register_hook(torch.zeros_like)
        before = actor_weights(policy)
        norm, _ = world.improve(policy, opt, image, ram, torch.full((len(obs),), 512, device=device))
        deltas.append(actor_weights(policy) - before)
        grads.append(norm)
    actual, zero = deltas
    if grads[1] != 0 or grads[0] <= 0:
        raise AssertionError("Gradient control did not isolate the current imagined gradient")
    return dict(observations=len(obs), real_gradient_norm=grads[0], zero_gradient_norm=grads[1],
                actual_step_norm=actual.norm().item(), zero_gradient_step_norm=zero.norm().item(),
                zero_to_actual_norm=(zero.norm()/actual.norm()).item(),
                step_cosine=torch.nn.functional.cosine_similarity(actual, zero, dim=0).item(),
                relative_difference=((actual-zero).norm()/actual.norm()).item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu","cuda"), default="cuda")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output file")
    torch.set_num_threads(2)
    result = {p.parent.name: audit(p, torch.device(args.device), 75253) for p in args.checkpoints}
    write_json(args.out, result)
    print(result)


if __name__ == "__main__":
    main()
