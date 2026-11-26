"""Fresh mGBA rollouts with independent sampling seeds and an autonomous earlier policy."""
import json
import numpy as np
import torch
from gym_data import ROOT, RUN
from pool_gba import Pool
from policy import Policy, tensors


def main():
    cfg = json.loads((ROOT / "local.json").read_text(encoding="utf-8-sig"))
    checkpoint = ROOT / "runs/native-night-20261124/long/train-01/snapshots/0010986496/policy.pt"
    actor = Policy().cuda().eval()
    actor.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    pool = Pool(4, cfg["core"], cfg["rom"])
    dest = RUN / "experience"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        for label, seed in (("train-fresh", 2006271), ("test-fresh", 4907131), ("train-fresh-b", 337181)):
            generator = torch.Generator(device="cuda").manual_seed(seed)
            obs = list(pool.initial)
            for i in range(4):
                pool.request(i, "load", None)
            frames = [[o[0]] for o in obs]
            flags = [[np.zeros(3, dtype=np.uint8)] for i in range(4)]
            actions = [[] for i in range(4)]
            with torch.inference_mode():
                for _ in range(1100):
                    dist, _ = actor(*tensors(obs, [None] * 4, "cuda"))
                    buttons = torch.multinomial(dist.probs, 1, generator=generator).flatten().tolist()
                    pairs = pool.step(buttons, verified=True)
                    obs = [pair[0] for pair in pairs]
                    for i, ((image, key), facts) in enumerate(pairs):
                        frames[i].append(image)
                        actions[i].append(buttons[i])
                        current = flags[i][-1].copy()
                        if facts is not None:
                            current |= np.array([key[6], *facts], dtype=np.uint8)
                        flags[i].append(current)
            for i in range(4):
                np.savez_compressed(dest / f"{label}-{i}.npz", image=np.asarray(frames[i]),
                                    flags=np.asarray(flags[i]), action=np.array(actions[i], dtype=np.uint8))
            print(label, seed, [f[-1].tolist() for f in flags], flush=True)
    finally:
        pool.close()


if __name__ == "__main__":
    main()
