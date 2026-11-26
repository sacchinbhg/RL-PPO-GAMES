"""Locate component-level rendering differences across four replays of shared inputs."""
import json
from pathlib import Path
import numpy as np

base=Path(__file__).resolve().parents[1]
root=base/"outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
cache=base/"work/emerald-render-traces"
first=root/"paired-self-earned-frontiers/visual-slots-0/discoveries/work-0000139234/game.bin"
other,=(root/"reward-only-training-forks/verified-training-tapes/24479561ff0b9fe07b3932aeb611e70d3b7b7f4b42a9f6fe428cea84e19fa412/audit").glob("*.bin")
a,b=first.read_bytes(),other.read_bytes()
common=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
roots={"first":cache/"frames-36058-2","second":cache/"frames-16573-2/24479561ff0b"}
data={}
for sample,where in roots.items():
    for index,variant in enumerate(("stock","cadence")):
        with np.load(where/f"run-{index}-{variant}/trace.npz") as f:
            data[sample,variant]=f["obs"].tolist()
def differences(s,t,n):
    x,y=data[s],data[t]
    return dict(policy=[i+1 for i in range(n) if x[i][:32]!=y[i][:32]],
                native=[i+1 for i in range(n) if x[i][32:]!=y[i][32:]])
observed=dict(common_inputs=common,first_stock_vs_second_stock=differences(("first","stock"),("second","stock"),common),
    first_cadence_vs_second_cadence=differences(("first","cadence"),("second","cadence"),common),
    second_stock_vs_cadence=differences(("second","stock"),("second","cadence"),len(data["second","stock"])))
dest=root/"controls/last-frame-render/cross-tape-equivalence.json"
dest.write_text(json.dumps(observed,indent=2))
print(json.dumps(observed))
