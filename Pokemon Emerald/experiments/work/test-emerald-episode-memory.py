"""Prove which episode metrics change and how many batched image buffers stay alive."""
import contextlib
import gc
from pathlib import Path
import sys
import weakref
import elements
import numpy as np

project = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL"
sys.path.insert(0, str(project))
from upstream_emerald import skip_json_episode_video


def trial(omit):
    buffers = []
    length = 128
    with skip_json_episode_video(elements) if omit else contextlib.nullcontext():
        metrics = elements.Agg()
        for step in range(length):
            batch = np.full((4, 64, 64, 3), step, np.uint8)
            buffers.append(weakref.ref(batch))
            metrics.add("policy_image", batch[0], agg="stack")
            metrics.add("score", 0.25, agg="sum")
            metrics.add("length", 1, agg="sum")
            metrics.add("rewards", 0.25, agg="stack")
        del batch
        gc.collect()
        held = sum(x() is not None for x in buffers)
        report = metrics.result()
        if not omit:
            assert report.pop("policy_image").shape == (length, 64, 64, 3)
        # The separate upstream prediction report continues to use its normal reducer.
        metrics.add({"openloop/image": np.zeros((2, 4, 4, 3), np.uint8)})
        assert metrics.result()["openloop/image"].shape == (2, 4, 4, 3)
        return report, held


ordinary, before = trial(False)
lean, after = trial(True)
assert ordinary.keys() == lean.keys()
assert all(np.array_equal(ordinary[k], lean[k]) for k in ordinary)
assert (before, after) == (128, 0)
print("Identical episode scalars and reward history; retained four-worker image batches: 128 -> 0.")
