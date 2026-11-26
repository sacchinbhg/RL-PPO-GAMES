from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

out=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL/runs/pixels-only-01/official-wsl/controls"
data=json.loads((out/"replay-eviction-262152.json").read_text())
rows=data["rows"]
x=[r["step"] for r in rows]
y=[r["reward_at_least_one_windows"] for r in rows]
first=next(r["step"] for r in rows if r["reward_at_least_one_windows"]==0)
fig,ax=plt.subplots(figsize=(12.8,7.2),dpi=110,facecolor="#f7f8f4")
fig.subplots_adjust(left=.11,right=.96,top=.78,bottom=.22)
ax.set_facecolor("#f7f8f4")
fig.text(.075,.91,"Story rewards disappeared from active replay",size=26,weight="bold",color="#122e29")
fig.text(.077,.85,"Pokémon Emerald  ·  Reward-weighted DreamerV3  ·  Training seed 71,299",size=14,color="#425c55")
ax.plot(x,y,color="#17785e",linewidth=2.5,marker="o",markersize=5)
ax.axvline(first,color="#8b9e97",lw=1,ls=(0,(3,4)))
ax.annotate("First measured zero\n216,852 steps",(first,0),xytext=(first+3500,115),
            color="#135741",size=13,arrowprops=dict(arrowstyle="-",color="#567c6a"))
ax.annotate("0 of 131,072 sequences",(x[-1],0),xytext=(x[-1]-500,50),ha="right",color="#135741",size=13)
ax.set_xlim(197000,264000)
ax.set_ylim(-16,max(y)*1.12)
ax.set_xticks([200000,215000,230000,245000,260000])
ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f"{v/1000:.0f}k"))
ax.set_xlabel("Training steps",size=13,labelpad=13,color="#263e35")
ax.set_ylabel("Sequences with reward ≥ 1",size=13,labelpad=12,color="#263e35")
ax.set_axisbelow(True)
ax.grid(axis="y",color="#d6ded7",lw=.8)
for spine in ax.spines.values(): spine.set_visible(False)
ax.tick_params(colors="#425c55",labelsize=11,length=0,pad=6)
fig.text(.075,.067,"Each dot uses the saved four-worker experience. Small exploration rewards of 0.02 still appeared.",
         size=12,color="#425c55")
fig.savefig(out/"story-rewards-in-replay.png")
plt.close(fig)
