import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

project = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL"
sys.path.insert(0, str(project))
from emerald import BUTTONS, Emerald
from experiment import moment_notes

base = project / "runs/world-comparison"
def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

policies = {hashlib.sha256(p.read_bytes()).hexdigest() for p in base.rglob("policy.pt")}
reports = list(base.rglob("evaluation.json"))
captures = images = 0
for path in reports:
    report = read(path)
    assert report["start"]=="fresh boot" and report["resets"]==0 and len(report["trials"])==8
    if report["policy"]=="sampled frozen policy":
        assert report["policy_sha256"] in policies
    m=path.parent/"moments"
    if not m.is_dir():
        continue
    captures += 1
    meta=read(m/"moments.json")
    actions=np.load(m/"buttons.npz",allow_pickle=False)["actions"]
    assert actions.shape==(report["decision_budget"],8)
    assert actions.max()<len(BUTTONS) and meta["policy_sha256"]==report["policy_sha256"]
    for event, entry in meta["events"].items():
        with Image.open(m/entry["file"]) as pic:
            assert pic.size==(240,160) and pic.mode=="RGB"
        i,t=entry["trial"],entry["decision"]
        assert BUTTONS[actions[t-1,i]]==entry["button"]
        if event in ("clock","starter","first_badge"):
            assert report["trials"][i]["first_milestone_decision"][event]==t
        images+=1
    moment_notes(m,meta["events"],report["policy"])

original=read(base/"world-19-eval/evaluation.json")
assert original==read(base/"world-19-capture/evaluation.json")
directory=base/"world-37-eval/moments"
moment=read(directory/"moments.json")["events"]["clock"]
actions=np.load(directory/"buttons.npz")["actions"]
cfg=read(project/"local.json")
env=Emerald(cfg["core"],cfg["rom"])
try:
    for a in actions[:moment["decision"],moment["trial"]]:
        _,key=env.step(int(a))
    assert tuple(moment["descriptor"])==key
    with Image.open(directory/moment["file"]) as pic:
        np.testing.assert_array_equal(env.screen,np.asarray(pic))
finally:
    env.close()

result=dict(evaluations=len(reports),unique_policies=len(policies),capture_runs=captures,
            verified_stills=images,seed19_capture_reproduced_primary=True,
            independently_replayed_native_frame=dict(training_seed=37,trial=moment["trial"],
                decision=moment["decision"],descriptor_and_all_rgb_pixels_match=True))
(base/"artifact-check.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(result)
