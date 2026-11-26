from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
from collections import Counter

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from earned_memory import Memory,encode
from visual_route_recovery import VisualRouteRecovery,appearance
from train_earned_dreamer import Trajectory,write_json
root=project/"runs/pixels-only-01/official-wsl"

images=[np.full((40,60,3),i*24,np.uint8) for i in range(8)]
trace=dict(features=np.stack([encode(im)[1] for im in images]),buttons=np.array([4,3,1,7,2,5,6,0]),total=np.array(1.))
fixture=VisualRouteRecovery(SimpleNamespace(traces=[trace]))
for _ in range(255):
    assert fixture.choose(images[0],0) is None
for i in range(6):
    assert fixture.choose(images[i],0)==trace["buttons"][i]
assert fixture.choose(images[6],0) is None and fixture.stats["arrived_at_known_late_screen"]==1

memory=Memory(root/"earned-screen-memory/model",197461)
model=VisualRouteRecovery(memory)
data=root/"earned-online-datasets/c43fbf0200a1e10d70e0cf3f93bdc64547005e3c7547372619a0041dc3146df5"
game=Trajectory(data)
assert game.source["source_sha256"] not in memory.identity["self_earned_tapes"]
counts=Counter()
states=set()
matching=set()
previous=None
for image,button,reward,first,last,terminal in game.episode():
    _,feature=encode(image)
    state=appearance(feature)
    if previous is not None:
        old,action=previous
        options=model.choices.get(old,())
        counts["heldout_transitions"]+=1
        if options:
            states.add(old)
            counts["positions_with_a_route"]+=1
            suggested=min(options)[1]
            label="matched_advice" if action==suggested else "other_recorded_action"
            counts[label]+=1
            if label=="matched_advice":
                matching.add(old)
            if model.distance.get(state,float("inf"))<model.distance[old]:
                counts[label+"_followed_by_lower_distance"]+=1
    previous=state,button
out=root/"controls/visual-route-recovery"
out.mkdir(exist_ok=True)
write_json(out/"verification.json",dict(fit=model.fit,source_tapes=memory.identity["self_earned_tapes"],
           heldout_tape=game.source["source_sha256"],heldout=dict(counts),
           distinct_heldout_route_screens=len(states),distinct_screens_with_matching_advice=len(matching),
           fixture_reaches_target=True,holdoff_checked=True))
