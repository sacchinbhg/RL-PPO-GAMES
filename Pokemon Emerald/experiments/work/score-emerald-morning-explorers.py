"""Compare real discovery work and coverage from sealed original-ROM explorer records."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
family=root/"paired-self-earned-frontiers"
out=root/"controls/exploration-comparison"
read=lambda p:json.loads(p.read_text())

groups=[
    ("systematic-reward-frontiers","Earlier explorer",173741),
    ("sweep-reward-frontiers","Try every button",173741),
    ("sweep-individual-tiles","Try every button with tile rewards",173741),
    ("sweep-unrewarded-scroll","Save unrewarded camera moves",173741),
    ("systematic-reward-frontiers-seed-173742","Earlier explorer",173742),
    ("sweep-reward-frontiers-seed-173742","Try every button",173742),
    ("sweep-movement-ordinary","Three-button movements",173746),
    ("sweep-movement-learned","Three-button movements ranked by model",173746),
    ("bank-movement-ordinary","Matched earned starts, ordinary movement",173746),
    ("bank-movement-learned","Matched earned starts, model-ranked movement",173746),
    ("bank-movement-ordinary-restart1","Matched earned starts, ordinary movement after restart",173746),
    ("bank-movement-learned-restart1","Matched earned starts, model-ranked movement after restart",173746),
]


def run():
    source=read(root/"controls/reward-frontier-expansion/plan.json")["selected"]
    warm=read(root/"controls/trainer-contacts/highest-return-review/measured.json")
    assert source["tape_sha256"]==warm["input_sha256"]
    initial=warm["first_visit_cells"]
    rows=[]
    for folder,label,seed in groups:
        directory=family/folder
        if not (directory/"settings.json").exists():
            continue
        setting=read(directory/"settings.json")
        assert setting["seed"]==seed and setting["source_sha256"]==source["tape_sha256"] and setting["frame_repeat"]==24
        init=directory/"initialization.json"
        starting_cells=read(init).get("initial_coarse_cells",initial) if init.exists() else initial
        series=[]
        for file in sorted(directory.glob("discoveries/work-*/record.json")):
            row=read(file)
            log=read(file.with_name("game.json"))
            assert row["distinct_cells_across_training"]==log["distinct_cells_across_training"]
            series.append((row["work"]-source["work"],row["distinct_cells_across_training"]-starting_cells))
        if not series:
            continue
        series.sort()
        assert all(x[1]<=y[1] for x,y in zip(series,series[1:]))
        checks={}
        for budget in (16384,32768,65536,131072,262144):
            first=next((x for x in series if x[0]>=budget),None)
            if first:
                checks[budget]=dict(actual_emulator_decisions=first[0],additional_coarse_cells=first[1])
        rows.append(dict(folder=folder,display=label,seed=seed,source=source["tape_sha256"],checks=checks,starting_cells=starting_cells,
                         last_sealed=dict(real_decisions=series[-1][0],additional_cells=series[-1][1])))
    out.mkdir(exist_ok=True)
    temp=out/"matched.tmp"
    temp.write_text(json.dumps(dict(source_buttons=source["length"],
        inherited_global_decisions=source["work"],initial_coarse_cells=initial,frames_per_decision=24,
        threshold="first sealed sample at or beyond nominal number of new decisions",arms=rows),indent=2))
    temp.replace(out/"matched.json")
    present=[x for x in rows if x["seed"]==173741 and x["checks"]]
    fig,ax=plt.subplots(figsize=(8,4.6),layout="constrained")
    colors=["#627580","#19724b","#285d9b","#ae5425"]
    for color,arm in zip(colors,present):
        data=list(arm["checks"].values())
        ax.plot([0]+[x["actual_emulator_decisions"] for x in data],[0]+[x["additional_coarse_cells"] for x in data],
                marker="o",lw=2.3,label=arm["display"],color=color)
    ax.set_title("How much new ground did the explorer cover?",loc="left",weight="bold",fontsize=15)
    ax.set_xlabel("New 24-frame decisions after the shared training history")
    ax.set_ylabel("Additional 2 × 2 map cells")
    ax.ticklabel_format(style="plain",axis="x")
    ax.xaxis.set_major_formatter(lambda x,pos:f"{int(x/1000)}k" if x else "0")
    ax.spines[["right","top"]].set_visible(False)
    ax.grid(axis="y",alpha=.2)
    ax.legend(loc="lower right",frameon=True,facecolor="white",edgecolor="none",fontsize=9)
    fig.text(.01,-.015,f"Original Pokémon Emerald · same sampling seed · {initial} cells already visited at the start. "
             "Coverage counts discoveries during training.",fontsize=8)
    fig.savefig(out/"coverage.png",dpi=160,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    run()
