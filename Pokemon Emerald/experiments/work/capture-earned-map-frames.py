"""Extract the actual logged policy input near an independently replayed map entry."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

def take(data,place,dest):
    manifest=json.loads((data/"manifest.json").read_text())
    audit,=(data/"audit").glob("*.json")
    record=json.loads(audit.read_text())
    entry=record["maps_first_reached"][place]
    assert manifest["replay_restores_during_game"]==0 and not manifest["frozen_evaluation_data_used"]
    dest.mkdir(parents=True,exist_ok=True)
    snapshots=[]
    for at in (entry,entry+4,entry+8,entry+16):
        if at>record["steps"]:
            break
        size=manifest["chunk_length"]
        offset=(at-1)//size*size
        with np.load(data/f"chunk-{offset:09d}.npz") as rows:
            frame=Image.fromarray(rows["image"][at-offset])
        name=f"decision-{at}.png"
        frame.resize((480,320),Image.Resampling.NEAREST).save(dest/name)
        snapshots.append(dict(after_real_button=at,file=name))
    (dest/"frames.json").write_text(json.dumps(dict(tape_sha256=manifest["source_sha256"],place=place,
        first_overworld_decision=entry,source="original exported 60x40 RGB policy observations",
        display_scale=8,resize="nearest neighbor",frames=snapshots),indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("dataset",type=Path)
    p.add_argument("map")
    p.add_argument("output",type=Path)
    a=p.parse_args()
    take(a.dataset,a.map,a.output)
