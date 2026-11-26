"""Compare the first new online checkpoint with the frozen official parent before its first update."""
import json
from pathlib import Path
import pickle
import sys
import numpy as np

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from train_earned_dreamer import digest,write_json

root=project/"runs/pixels-only-01/official-wsl"
source=root/"seed-71299-h15/eval-65536/frozen-policy/agent.pkl"
for arm in sys.argv[1:]:
    run=root/"earned-reset-online-comparison"/arm/"train"
    candidates=sorted((run/"ckpt").glob("*/step.pkl"))
    zero=next(p for p in candidates if int(pickle.loads(p.read_bytes()))==0)
    target=zero.with_name("agent.pkl")
    a=pickle.loads(source.read_bytes())
    b=pickle.loads(target.read_bytes())
    assert set(a["params"])==set(b["params"])
    assert all(np.array_equal(a["params"][k],b["params"][k]) for k in a["params"])
    expected={**a["counters"],"batches":a["counters"]["updates"]}
    assert expected==b["counters"]
    write_json(root/"controls/dreamer-earned-resets"/(arm+"-initial-weights.json"),
        dict(fresh_log_step=0,matched_parameters=len(a["params"]),parameter_bytes=sum(v.nbytes for v in a["params"].values()),
             all_values_equal=True,source_counters=a["counters"],restored_counters=b["counters"],
             prefetched_batches_rewound_to_completed_updates=True,source_sha256=digest(source),initial_saved_sha256=digest(target)))
