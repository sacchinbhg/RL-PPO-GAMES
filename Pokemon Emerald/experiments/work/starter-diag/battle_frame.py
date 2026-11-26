"""Replay the diagnostic route and identify the player's battler on the original ROM."""
import json,sys
from pathlib import Path
from PIL import Image
project=Path(__file__).resolve().parents[2]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from emerald import Emerald
from experiment import location
cfg=json.loads((project/"local.json").read_text())
out=project/"starter-diagnostics"
e=Emerald(cfg["core"],cfg["rom"])
names={277:"Treecko",280:"Torchic",283:"Mudkip",288:"Zigzagoon"}
def sp(offset):return int.from_bytes(e.ram(0x02024084+offset,2),"little")
try:
    actions=(out/"guided-party.bin").read_bytes()
    for action in actions:e.step(action)
    assert e.verified_story(e.descriptor())==(True,True)
    selected=None
    for n in range(1,110):
        im,k=e.step(0)
        pair=sp(0),sp(0x58)
        if pair[0] in (277,280,283) and pair[1]==288 and k[0]==6 and im.std()>35:
            selected=(n,pair)
    if selected:
        n,pair=selected
        e.load(e.boot)
        for action in actions:e.step(action)
        for _ in range(n):e.step(0)
        assert (sp(0),sp(0x58))==pair
        e.step(1)
        for _ in range(65):e.step(0)
        assert (sp(0),sp(0x58))==pair
        Image.fromarray(e.screen).save(out/"starter-battle.png")
        result=dict(recorded_buttons=len(actions),capture_suffix=f"{n} waits, one A, 65 waits",party_count=e.ram(0x020244e9,1)[0],
                    map=location(e.descriptor()),player_species=names[pair[0]],opponent_species=names[pair[1]])
        (out/"battle-frame.json").write_text(json.dumps(result,indent=2)+"\n")
        print(result)
    else:raise RuntimeError("Could not identify both battlers")
finally:e.close()
