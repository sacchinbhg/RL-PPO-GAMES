"""A fixed, uniform controller for interpreting fresh-game milestone rates."""
import json
from pathlib import Path
import sys
import numpy as np

project = Path(__file__).resolve().parents[1] / "outputs" / "EmeraldRL"
sys.path.insert(0, str(project))
from pixel_env import PixelPool


def main():
    seed, horizon = 271340, 8192
    out = project / "runs/pixels-only-01/official-wsl/controls/uniform-seed-271340"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for first in (0, 2):
        pool = PixelPool(2)
        try:
            rngs = [np.random.default_rng(np.random.SeedSequence([seed, first+i])) for i in range(2)]
            finished = set()
            for i in range(2):
                pool.command(i, "reset")
            for _ in range(horizon):
                for i, rng in enumerate(rngs):
                    if i not in finished:
                        pool.pipes[i].send(("act", int(rng.integers(9))))
                for i in range(2):
                    if i not in finished and pool.recv(i)[2]:
                        finished.add(i)
                if len(finished) == 2:
                    break
            for i in range(2):
                folder = out / f"game-{first+i}"
                pool.command(i, "audit", str(folder))
                audit, = folder.glob("*.json")
                records.append(json.loads(audit.read_text()))
                print(first+i, records[-1], flush=True)
        finally:
            pool.close()
    result = dict(controller="independent uniform draw over 9 buttons", seed=seed, budget=horizon,
                  seeds="NumPy SeedSequence([seed, game index])", games=records)
    (out / "summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
