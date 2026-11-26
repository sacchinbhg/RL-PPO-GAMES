"""Reconstruct the official uniform sampler's end-of-block sequence starts from real worker order."""
import json
from pathlib import Path
import sys
import numpy as np
from ruamel.yaml import YAML

sys.path.insert(0, "/opt/emerald-rl/source")
from embodied.core.replay import Replay

root = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
arm = root / "seed-71299-h15-screen-collection"
train = arm / "train"
config = YAML(typ="safe").load((train / "config.yaml").read_text())
proof = json.loads((arm / "verified-training-experience.json").read_text())
length = config["consec_train"] * config["batch_length"] + config["replay_context"]
capacity = int(config["replay"]["size"])
workers = len(proof["games"])
assert proof["new_transitions"] % workers == 0 and config["replay"]["fracs"]["uniform"] == 1.
# Measurements use upstream insertion and eviction law. Images and actions do not affect uniform eligibility.
replay = Replay(length, capacity, chunksize=config["replay"]["chunksize"], online=False)
for seq in range(proof["new_transitions"] // workers):
    for i in range(workers):
        replay.add({"seq":np.int32(seq), "worker":np.int8(i)}, i)
coverage = [[] for _ in range(workers)]
for cid, index in replay.items.values():
    chunk = replay.chunks[cid].data
    coverage[int(chunk["worker"][index])].append(int(chunk["seq"][index]))
rows = []
for game in proof["games"]:
    i = game["worker"]
    starts = np.sort(coverage[i])
    assert len(starts) == capacity // workers and np.all(np.diff(starts)==1)
    rows.append(dict(worker=i, first_eligible_start=int(starts[0]), last_eligible_start=int(starts[-1]),
        windows=len(starts), windows_containing_each_reward={name:int(((starts<=pos) & (pos<starts+length)).sum())
            for name,pos in game["events"].items()}))
result = dict(real_transitions=proof["new_transitions"], uniform_capacity=capacity, window_length=length,
    online_fast_path="Recent online samples remain a separate path. This counts uniform replay eligibility at the end.",
    games=rows)
(arm/"opening-replay-coverage.json").write_text(json.dumps(result,indent=2))
print(json.dumps(rows),flush=True)
