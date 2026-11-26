"""Allocate the next training block from equal-length fresh-game evidence."""
import json
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
cutoff=8192
names=("seed-71299-h15","seed-71299-h15-reward-replay","seed-71299-h15-retained-rewards")
rows=[]
for name in names:
    evaluation=root/name/"eval-262144"
    prov=json.loads((evaluation/"provenance.json").read_text())
    files=sorted((evaluation/"external-audit").glob("*/*/*episode-00001.json"))
    assert len(files)==4
    games=[json.loads(f.read_text()) for f in files]
    assert all(g["steps"]>=cutoff for g in games) and prov["training_step"]==262152
    count=sum(g["events"].get("sent_to_meet_rival",cutoff+1)<=cutoff for g in games)
    rows.append(dict(arm=name,weights=prov["agent_sha256"],fresh_games=4,tv_by_8192=count))
selected=rows[0]["tv_by_8192"]>=2 and rows[0]["tv_by_8192"]>max(r["tv_by_8192"] for r in rows[1:])
record=dict(cutoff=cutoff,training_step=262152,criteria="Ordinary policy reaches TV in at least two "
            "of four fresh games by 8192 buttons and more often than both reward-sampling policies",
            comparison=rows,selected=names[0] if selected else None,next_budget=524288 if selected else None)
(root/"controls/ordinary-524288-priority.json").write_text(json.dumps(record,indent=2))
print(record,flush=True)
sys.exit(0 if selected else 1)
