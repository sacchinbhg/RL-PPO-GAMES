"""Inspect independently exported RGB arrays at the three differing frames of a shared opening."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

base=Path(__file__).resolve().parents[1]
root=base/"outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
out=root/"controls/last-frame-render/visual-differences"
out.mkdir(exist_ok=True)
sources=[
    next((root/"reward-only-training-forks/verified-training-tapes").glob("24479561ff0b*")),
    next((root/"paired-self-earned-frontiers/visual-slots-0/verified-training-tapes").glob("e30822c294f0*")),
    next((root/"earned-frontier-curriculum/verified-training-tapes").glob("01f5c681*")),
]
prefix=(next((sources[0]/"audit").glob("*.bin"))).read_bytes()[:10534]
cache=base/"work/emerald-render-traces"
with np.load(cache/"frames-16573-2/24479561ff0b/run-0-stock/trace.npz") as f:
    alt=f["obs"]
with np.load(cache/"frames-36058-2/run-0-stock/trace.npz") as f:
    reference=f["obs"]
rows={}
for dataset in sources:
    tape=next((dataset/"audit").glob("*.bin")).read_bytes()
    if tape[:len(prefix)]!=prefix:
        continue
    rows[dataset.name]={}
    for at in (10518,10533,10534):
        offset=(at-1)//256*256
        with np.load(dataset/f"chunk-{offset:09d}.npz") as f:
            frame=f["image"][at-offset]
        digest=hashlib.blake2b(frame,digest_size=16).hexdigest()
        Image.fromarray(frame).resize((480,320),Image.Resampling.NEAREST).save(out/f"{at}-{digest}.png")
        rows[dataset.name][at]=dict(policy_digest=digest,first_stock=digest==reference[at-1][:32],second_stock=digest==alt[at-1][:32])
(out/"comparison.json").write_text(json.dumps(rows,indent=2))
print(json.dumps(rows))
