"""Test whether a frozen Emerald policy responds to a requested archived destination."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from emerald import BUTTONS
from experiment import location, write_json
from policy import Policy, tensors
from pool_gba import Pool


def pairs_from_paths(archive, rng, limit):
    candidates = []
    for target, endpoint in archive.items():
        if target[0] != 5:
            continue
        end_path = endpoint["path"]
        for start, origin in archive.items():
            if start[:3] != target[:3] or start[3:5] == target[3:5]:
                continue
            prefix = origin["path"]
            gap = len(end_path) - len(prefix)
            if 3 <= gap <= 24 and end_path.startswith(prefix):
                candidates.append((start, target, end_path[len(prefix):]))
    rng.shuffle(candidates)
    selected, used = [], set()
    for start, target, path in candidates:
        if target not in used:
            selected.append((start, target, path))
            used.add(target)
        if len(selected) == limit:
            break
    return selected


def audit(checkpoint, core, rom, seed, limit, horizon):
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if data["spec"]["core_sha256"] != hashlib.sha256(core.read_bytes()).hexdigest():
        raise ValueError("The checkpoint requires its original emulator build")
    rng = np.random.default_rng(seed)
    archive = data["archive"]
    pairs = pairs_from_paths(archive, rng, limit)
    if not pairs:
        return dict(pairs=[], hits={}, reason="No eligible same-map path pairs")
    policy = Policy()
    policy.load_state_dict(data["policy"])
    policy.eval()
    # Reuse the same uniform variates for all three arms at each decision.
    uniform = rng.random((horizon, len(pairs)))
    pool = Pool(len(pairs), core, rom)
    rows = [dict(start=start, target=target, map=location(target), recorded_decisions=len(path),
                 first_hit={}) for start, target, path in pairs]
    try:
        for arm in ("recorded", "goal", "empty", "random"):
            obs, goals = [], []
            for i, (start, target, _) in enumerate(pairs):
                k = pool.request(i, "load", archive[start]["state"])
                if k != start:
                    raise AssertionError("Archive restore missed the start descriptor")
                obs.append((archive[start]["image"], k))
                goals.append(target if arm == "goal" else None)
            for t in range(horizon if arm != "recorded" else max(len(p[2]) for p in pairs)):
                if arm == "recorded":
                    actions = [path[t] if t < len(path) else 0 for _, _, path in pairs]
                elif arm == "random":
                    actions = (uniform[t] * len(BUTTONS)).astype(int)
                else:
                    with torch.inference_mode():
                        dist, _ = policy(*tensors(obs, goals, "cpu"))
                    cdf = np.cumsum(dist.probs.numpy(), axis=1)
                    actions = (uniform[t, :, None] > cdf).sum(axis=1).clip(max=len(BUTTONS)-1)
                obs = pool.step(actions)
                for i, (_, key) in enumerate(obs):
                    _, target, path = pairs[i]
                    if arm == "recorded" and t + 1 == len(path):
                        if key != target:
                            raise AssertionError("Stored continuation missed its recorded target")
                    if key == target:
                        rows[i]["first_hit"].setdefault(arm, t + 1)
    finally:
        pool.close()
    return dict(pairs=rows, hits={arm:sum(arm in r["first_hit"] for r in rows)
                                 for arm in ("recorded", "goal", "empty", "random")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("core", "rom", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=94027)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=96)
    args = parser.parse_args()
    if args.out.exists() or min(args.pairs, args.horizon) < 1:
        parser.error("Choose a new output and a positive budget")
    torch.set_num_threads(2)
    reports = {p.parent.name: audit(p, args.core, args.rom, args.seed, args.pairs, args.horizon)
               for p in args.checkpoints}
    write_json(args.out, dict(seed=args.seed, horizon=args.horizon, runs=reports))
    for name, report in reports.items():
        print(name, report["hits"], flush=True)


if __name__ == "__main__":
    main()
