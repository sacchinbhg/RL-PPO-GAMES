"""Measure alternative button orders in independent original-ROM replays for diagnostics."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import numpy as np

project = Path(__file__).resolve().parents[1] / "outputs" / "EmeraldRL"
sys.path.insert(0,str(project))
from emerald import Emerald
from pixel_env import game_paths
official = project / "runs/pixels-only-01/official-wsl"
out = official / "controls/heldout-world-model"


def play(job):
    path, start, i = job
    tape = path.with_suffix(".bin").read_bytes()
    cut = start + 16
    with np.load(out/"evaluation-windows.npz") as data:
        expected = data["image"][i,16,12:52,2:62]
        recorded = data["action"][i,16:32]
    assert list(recorded) == list(tape[cut:cut+16])
    g = Emerald(*game_paths())
    try:
        g.load(g.boot)
        for action in tape[:cut]:
            image, _ = g.step(action)
        assert np.array_equal(image, expected)
        return np.stack([g.step(int(action))[0] for action in recorded[::-1]])
    finally:
        g.close()


if __name__ == "__main__":
    paths=sorted((official/"seed-71299-h15/eval-65536/external-audit").glob("*/*/*.json"),
                 key=lambda p:p.parent.name)
    jobs=[(path,start,2*i+j) for i,path in enumerate(paths) for j,start in enumerate((2048,6144))]
    with ProcessPoolExecutor(max_workers=2) as pool:
        counterfactual=np.stack(list(pool.map(play,jobs)))
    with np.load(out/"evaluation-windows.npz") as data:
        original=data["image"][:,17:33,12:52,2:62]
    np.savez_compressed(out/"alternative-futures.npz", reversed=counterfactual, recorded=original)
    difference = (np.abs(original.astype(np.int16)-counterfactual)>=8).any(-1)
    print("Pixels changed by reversing the next 16 buttons:", difference.mean(), flush=True)
