"""Plot actual starter times for the four uncapped-length fresh games, keeping the original labels."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

root=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
parent=root/"seed-71299-h15"
out=parent/"highlights/long-game-starters"
out.mkdir(parents=True,exist_ok=True)
trial,=(parent/"unbounded-seed-271341").glob("fresh-*")
identity=json.loads((trial/"provenance.json").read_text())
assert identity["seed"]==271341 and identity["buttons_per_game"] is None and identity["in_game_restores"]==0
rows=[]
for i in range(4):
    samples=[]
    for p in (trial/"games"/f"env-{i}").glob("step-*/*episode-00001.json"):
        j=json.loads(p.read_text())
        if "lab_handoff" in j["events"]:
            samples.append((j["steps"],j,p))
    if not samples:
        raise ValueError(f"Missing starter handoff for worker {i}")
    _,j,p=min(samples,key=lambda s:s[0])
    assert j["events"]["starter"]<=j["events"]["lab_handoff"]<=j["steps"]
    rows.append(dict(worker=i,starter=j["events"]["starter"],lab_handoff=j["events"]["lab_handoff"],
                     audit=str(p.relative_to(root))))
rows.sort(key=lambda r:r["starter"])
budget=65536
data=dict(seed=identity["seed"],weight_sha256=identity["weight_sha256"],from_boot=True,restores=0,
          comparison_checkpoint=budget,started_by_checkpoint=sum(r["starter"]<=budget for r in rows),
          latest_starter=max(r["starter"] for r in rows),games=rows)
(out/"times.json").write_text(json.dumps(data,indent=2))
fig,ax=plt.subplots(figsize=(11.8,6.2),dpi=125)
fig.subplots_adjust(left=.13,right=.95,top=.74,bottom=.22)
fig.patch.set_facecolor("#faf9f5")
ax.set_facecolor("#faf9f5")
fig.text(.075,.9,"How long did the same policy take to find a starter?",size=21,weight="bold",color="#192d27")
fig.text(.075,.83,"Four fresh Emerald games with the same frozen Dreamer checkpoint",size=13,color="#435b53")
ax.axvline(budget,ls="--",lw=1.6,color="#a65a28")
label_box=dict(facecolor="#faf9f5",edgecolor="none",pad=1.5)
ax.text(budget,3.55,"65,536-decision checkpoint",ha="center",va="bottom",fontsize=10,color="#88451e",bbox=label_box)
for y,row in enumerate(reversed(rows)):
    t=row["starter"]
    ax.hlines(y,0,t,lw=4,color="#adcbba")
    ax.scatter([t],[y],s=90,color="#147957",zorder=3)
    ax.text(t+2600,y,f"{t:,}",va="center",fontsize=11,weight="bold",color="#264638",bbox=label_box)
ax.set(yticks=range(4),yticklabels=[f"Game {r['worker']+1}" for r in reversed(rows)],xlim=(0,205000),ylim=(-.45,3.9))
ax.set_xticks([0,50000,100000,150000,200000])
ax.xaxis.set_major_formatter(FuncFormatter(lambda n,_:f"{n:,.0f}"))
ax.set_xlabel("Real controller decisions from boot",labelpad=9)
ax.grid(axis="x",color="#dce4df",lw=.65)
ax.tick_params(length=0,pad=7,colors="#435b53")
for spine in ax.spines.values(): spine.set_visible(False)
fig.text(.075,.065,"One game had a starter at the checkpoint. All four eventually had one. Each decision advances 24 GBA frames.",
         fontsize=11,color="#435b53")
fig.savefig(out/"starter-times.png",facecolor=fig.get_facecolor())
