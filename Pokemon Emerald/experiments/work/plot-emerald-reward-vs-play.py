from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = Path(__file__).resolve().parents[1]/"outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
controls = base/"controls"
arms = [("reward-replay","65536"), ("reward-replay","262144"), ("retained-rewards","262144")]
predictions, bedrooms = [], []
for arm, phase in arms:
    name = f"seed-71299-h15-{arm}"
    probe = json.loads((controls/"heldout-rewards"/f"{name}-eval-{phase}.json").read_text())
    events = [v for v in probe["windows"] if v["offset"] == 0]
    assert len(events) == 13
    predictions.append(probe["prior"]["mean_on_story"])
    games = [json.loads(p.read_text()) for p in (base/name/f"eval-{phase}/external-audit").glob("*/*/*.json")]
    assert len(games) == 4 and all(g["steps"] >= 8192 for g in games)
    bedrooms.append(sum(g["events"].get("bedroom",8193) <= 8192 for g in games))

bg, ink, muted = "#f7f8f4","#122e29","#425c55"
colors = ["#718a7c","#718a7c","#11735b"]
fig = plt.figure(figsize=(12.8,7.2),dpi=125,facecolor=bg)
fig.text(.06,.91,"Reward forecasts improved. Fresh games still stalled.",fontsize=25,weight="bold",color=ink)
fig.text(.061,.85,"Pokémon Emerald  ·  Official DreamerV3 learner  ·  One training seed",fontsize=14,color=muted)
ax = fig.add_axes([.27,.31,.33,.39],facecolor=bg)
ax.set_title("Mean predicted reward",fontsize=15,weight="bold",loc="left",pad=17,color=ink)
ax.set_yticks([2,1,0], ["65k  ·  weighted replay","262k  ·  weighted replay","262k  ·  retained examples"])
ax.set_xlim(0,.68)
ax.set_ylim(-.55,2.55)
ax.set_xticks([0,.2,.4,.6])
ax.grid(axis="x",color="#d6ded7",linewidth=.8,zorder=0)
for y,value,color in zip([2,1,0],predictions,colors):
    ax.plot([0,value],[y,y],lw=3,color=color,zorder=2)
    ax.scatter(value,y,s=72,color=color,zorder=3,clip_on=False)
    ax.annotate(f"{value:.4f}" if value<.01 else f"{value:.3f}", (value,y),
                xytext=(8,0),textcoords="offset points",va="center",fontsize=12,color=ink)
ax.tick_params(axis="y",length=0,pad=12,labelsize=13,labelcolor=ink)
ax.tick_params(axis="x",length=0,pad=8,labelsize=11,labelcolor=muted)
for spine in ax.spines.values():
    spine.set_visible(False)
right = fig.add_axes([.69,.31,.24,.39],facecolor=bg)
right.set_title("Games reaching the bedroom",fontsize=15,weight="bold",loc="left",pad=17,color=ink)
right.set_xlim(-.4,5.15)
right.set_ylim(ax.get_ylim())
right.axis("off")
for y,count,color in zip([2,1,0],bedrooms,colors):
    for i in range(4):
        right.scatter(i,y,s=160,marker="o",facecolor=color if i<count else bg,edgecolor=color,lw=1.6)
    right.text(3.65,y,f"{count} / 4",va="center",fontsize=15,weight="bold",color=ink)
fig.text(.06,.17,"Forecasts use the same 13 held-out story events. Their mean actual reward was 1.85.",size=13,color=muted)
fig.text(.06,.115,"Every game started from boot. Each checkpoint is scored at the same 8,192-button cutoff.",size=13,color=muted)
fig.savefig(controls/"reward-prediction-and-fresh-games.png")
plt.close(fig)
