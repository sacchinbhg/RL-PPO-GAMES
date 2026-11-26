"""Locate the exact real frames used by the first open-loop prediction report."""
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL/runs/pixels-only-01/official-wsl/seed-71299-h1"
report = root / "train/world-model-predictions/step-000000042252.npz"


def fingerprint(images):
    return b"".join(hashlib.blake2b(im.tobytes(), digest_size=16).digest() for im in images)


with np.load(report) as data:
    video = data["frames"]
valid = np.any(video[:,0,0] != 0, axis=-1)
true = video[valid].reshape(-1,196,6,68,3).transpose(2,0,1,3,4)[:,:,2:66,2:66]
needles = [fingerprint(seq) for seq in true]
paths = sorted((root / "train/replay").glob("*.npz"))
chunks = {path.stem.split("-")[1]:path for path in paths}
successors = {path.stem.split("-")[2] for path in paths}
roots = [p for p in paths if p.stem.split("-")[1] not in successors]
matches = [[] for _ in needles]
for i, p in enumerate(roots):
    parts = []
    while p:
        with np.load(p) as data:
            parts.append(fingerprint(data["image"]))
        p = chunks.get(p.stem.split("-")[2])
    history = b"".join(parts)
    for s, needle in enumerate(needles):
        pos = history.find(needle)
        while pos >= 0:
            if pos % 16 == 0:
                matches[s].append(dict(stream=i, first_observation=pos//16, length=true.shape[1]))
            pos = history.find(needle, pos+16)
assert all(matches), matches
result = dict(report=report.name, exact_matching_sequences=matches)
(root / "highlights/prediction-42252/replay-origins.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
