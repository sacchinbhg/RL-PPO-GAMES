"""Recover the last live FIFO windows and cross-check their counts against training logs."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
train=root/"seed-71299-h15-reward-replay/train"
files=list((train/"replay").glob("*.npz"))
chunks={f.stem.split("-")[1]:f for f in files}
successors={f.stem.split("-")[2] for f in files}
roots=sorted(f for k,f in chunks.items() if k not in successors)
chains=[]
for path in roots:
    parts=[]
    while path:
        with np.load(path) as chunk:
            parts.append(chunk["reward"])
        path=chunks.get(path.stem.split("-")[2])
    chains.append(np.concatenate(parts))
assert [len(c) for c in chains]==[16386]*4+[49152]*4
capacity=131072
start=65544
logged=[json.loads(line) for line in (train/"metrics.jsonl").read_text().splitlines()]
records={r["step"]:r for r in logged if "replay/reward_windows" in r and r["step"]>=start+capacity+128}
rows=[]
for step in sorted(set(records)|{262152}):
    end=(step-start)//4
    scores=[]
    for stream in chains[-4:]:
        data=stream[end-capacity//4-32:end]
        assert len(data)==capacity//4+32
        scores.append(np.lib.stride_tricks.sliding_window_view(data,33)[:,1:].max(1))
    peaks=np.concatenate(scores)
    positives=int(np.count_nonzero(peaks))
    # Prefetch does not change the number of active, eligible reward windows.
    if step in records:
        assert positives == records[step]["replay/reward_windows"], (step,positives,records[step])
    rows.append(dict(step=step,windows=len(peaks),positive_reward_windows=positives,
                     reward_at_least_one_windows=int((peaks>=1).sum()),largest_reward=float(peaks.max())))
summary=dict(capacity=capacity,context=1,training_length=32,verified_logged_counts=len(records),
             story_sized_observations_recorded=sum(int((r>=1).sum()) for r in chains),
             rows=rows)
(root/"controls/replay-eviction-262152.json").write_text(json.dumps(summary,indent=2))
print("Exact log checks",len(records),"final",rows[-1],"first logged empty",next(r["step"] for r in rows if r["reward_at_least_one_windows"]==0))
