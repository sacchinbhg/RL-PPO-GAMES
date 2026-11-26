"""Exercise loss timing on a three-transition fixture and verify the real smoke-test stream."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import numpy as np

base=Path(__file__).resolve().parents[1]
root=base/"outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
spec=importlib.util.spec_from_file_location("compare_whiteout",Path(__file__).with_name("compare-emerald-whiteout-cost.py"))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from train_earned_dreamer import Trajectory,write_json

with tempfile.TemporaryDirectory() as temp:
    # A fixture, kept outside real training data. Its middle and final transitions are marked.
    folder=Path(temp)
    np.savez(folder/"chunk-000000000.npz",image=np.zeros((4,40,60,3),np.uint8),
             action=np.zeros(3,np.uint8),reward=np.zeros(3,np.float32),terminal=np.zeros(3,bool))
    game=object.__new__(Trajectory)
    game.folder=folder
    game.tape=bytes(3)
    losses=[{"decision":2,"outcome":2},{"decision":3,"outcome":2}]
    game.audit=dict(uncapped_exploration=True,first_visit_cells=0,events={},whiteouts=losses)
    game.source=dict(return_cents=0,verified_whiteouts=losses)
    assert [r[2] for r in game.episode(False,50)]==[0.,0.,-.5,-.5]
    assert [r[2] for r in game.episode(False)]==[0.,0.,0.,0.]
    game.source.pop("verified_whiteouts")
    try:
        list(game.episode(False,50))
    except ValueError:
        pass
    else:
        raise AssertionError("Unverified outcomes were accepted")
control=root/"controls/outcome-collection-smoke"
real=next((control/"verified-training-tapes").glob("*/manifest.json")).parent
checked=module.check_stream(real,50)
assert checked["changed_observations"]==[] and checked["recorded_buttons"]==2804
write_json(control/"checked-loss-alignment.json",dict(real_smoke=checked,fixture_changes_only_at_successor_observations=[2,3],
           last_observation_cost_verified=True,unverified_outcomes_rejected=True))
