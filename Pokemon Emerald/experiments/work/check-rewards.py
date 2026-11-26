from collections import defaultdict
import json
from pathlib import Path
import torch

base = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL/runs/world-comparison"
report = {}
for seed in (19,37,71):
    d = torch.load(base/f"world-{seed}/checkpoint.pt",map_location="cpu",weights_only=False)
    rows = d["world"]["replay"].numpy()
    # Layout from Replay.add: packed state, scalar action, goal, next state, scalar reward.
    state_dim = (rows.shape[1] - 24 - 2) // 2
    outcomes = defaultdict(set)
    counts = defaultdict(int)
    positives = defaultdict(int)
    for row in rows:
        key = row[:state_dim+1+24].tobytes()
        r = float(row[-1])
        outcomes[key].add(r)
        counts[key] += 1
        positives[key] += r > 0
    ambiguous = [k for k, rs in outcomes.items() if len(rs)>1]
    report[f"world-{seed}"] = dict(rows=len(rows), unique_inputs=len(outcomes),
        ambiguous_inputs=len(ambiguous), ambiguous_rows=sum(counts[k] for k in ambiguous),
        positive_rows=sum(positives.values()), ambiguous_positive_rows=sum(positives[k] for k in ambiguous))
    del d, rows
(base/"reward-aliasing.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(report)
