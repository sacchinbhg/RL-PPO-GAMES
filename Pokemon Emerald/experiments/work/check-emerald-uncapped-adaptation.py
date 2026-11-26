"""Verify that the proposed training ablation changes only the already-earned visit component."""
from pathlib import Path
import sys
import numpy as np

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from train_earned_dreamer import Trajectory,write_json

root=project/"runs/pixels-only-01/official-wsl"
source_dirs=(root/"reward-only-training-forks/verified-training-tapes",
             root/"paired-self-earned-frontiers/visual-slots-0/verified-training-tapes")
sha=("24479561ff0b9fe07b3932aeb611e70d3b7b7f4b42a9f6fe428cea84e19fa412",
     "e30822c294f0d5469126f3e250c9ad4ac83b232534b37d6f54b24b56078e37c5")
out=root/"controls/earned-recurrent-stream/uncapped-adaptation.json"
rows=[]
for directory,source in zip(source_dirs,sha,strict=True):
    game=Trajectory(directory/source)
    extra=visits=adjusted=raw=0
    for left,right in zip(game.episode(False),game.episode(True),strict=True):
        assert np.array_equal(left[0],right[0]) and left[1]==right[1] and left[3:]==right[3:]
        capped=round(left[2]*100)
        uncapped=round(right[2]*100)
        visit=uncapped%25==2
        visits+=int(visit)
        assert uncapped-capped==(2 if visit and visits>50 else 0)
        extra+=int(uncapped!=capped)
        raw+=uncapped
        adjusted+=capped
    assert visits==game.audit["first_visit_cells"] and raw==game.source["return_cents"]
    rows.append(dict(tape_sha256=source,actions=len(game.tape),visits=visits,additional_nonzero_visit_terms=extra,
                     capped_return_cents=adjusted,uncapped_return_cents=raw,all_other_inputs_identical=True))
write_json(out,dict(episodes=rows))
print(rows)
