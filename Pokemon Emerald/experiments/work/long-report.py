import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL"
base = project / "runs/world-comparison"
long = base / "long"

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def result(run):
    s = read(run/"training.json")
    records = read(run.with_name(run.name+"-eval")/"evaluation.json")
    assert s["total_training_steps"] == s["steps"] + s["parent_steps"]
    assert records["resets"] == 0 and len(records["trials"]) == 8
    return dict(steps=s["total_training_steps"],wall=read(run/"elapsed.json")["wall_seconds"],
                clocks=sum(t["clock"] for t in records["trials"]),
                starters=sum(t["starter"] for t in records["trials"]),
                badges=sum(t["badges"]>0 for t in records["trials"]),
                intro=max(t["intro"] for t in records["trials"]),
                maps=sorted({m for t in records["trials"] for m in t["maps"]}),
                discoveries=s["first_milestone_step"], archive_cells=s["archive_cells"])

series = {}
labels = {"archive":"Archive-PPO", "world":"World model, original update", "full":"World model, full actor gradient"}
budgets = [32768,65536,98304]
for mode in labels:
    original = "world" if mode == "full" else mode
    series[mode] = [result(base / f"{original}-19")] + [result(long / f"{mode}-{b}") for b in budgets[1:]]
confirm_ready = all((long/f"{m}-{b}-confirm/evaluation.json").exists() for m in series for b in budgets[1:])

fig, (ax, right) = plt.subplots(1,2,figsize=(11.6,5),gridspec_kw={"width_ratios":[1,1.35]})
colors = {"archive":"#275C84", "world":"#CB592F", "full":"#148068"}
first = {m:[result(base/f"{m}-{s}")["clocks"] for s in (19,37,71)] for m in ("archive","world")}
fig.subplots_adjust(left=.075,right=.97,bottom=.26,top=.82,wspace=.28)
pos = np.arange(3)
for j,mode in enumerate(first):
    bars=ax.bar(pos+(j-.5)*.31,first[mode],.3,color=colors[mode])
    ax.bar_label(bars,padding=2,fontsize=10)
ax.set_xticks(pos,["19","37","71"])
ax.set_xlabel("Training seed")
ax.set_title("After 32,768 real decisions",loc="left",fontsize=13)
for mode, vals in series.items():
    xs=np.arange(3)
    ys=[v["clocks"]/8 for v in vals]
    if confirm_ready:
        for j,b in enumerate(budgets[1:], start=1):
            records=read(long/f"{mode}-{b}-confirm"/"evaluation.json")["trials"]
            ys[j] = (vals[j]["clocks"]+sum(t["clock"] for t in records))/16
    style={"archive":("o","-"),"world":("s","--"),"full":("^","-.")}[mode]
    right.plot(xs,ys,marker=style[0],linestyle=style[1],lw=2,markersize=7,
               color=colors[mode],label=labels[mode],alpha=.95)
right.set_xticks(range(3),["32,768","65,536","98,304"])
right.set_xlabel("Real training decisions")
right.set_title("Longer training from seed 19",loc="left",fontsize=13)
ax.set_ylabel("Fresh games that set the clock (out of 8)")
for axis in (ax,right):
    axis.grid(axis="y",alpha=.2)
    axis.set_axisbelow(True)
    axis.spines[["top","right"]].set_visible(False)
ax.set_ylim(-.3,8.55)
ax.set_yticks([0,2,4,6,8])
right.set_ylim(-.035,1.07)
right.set_yticks([0,.25,.5,.75,1],["0%","25%","50%","75%","100%"])
fig.suptitle("Can the Emerald agent set its bedroom clock?",x=.075,y=.97,ha="left",size=19,weight="bold")
fig.legend(*right.get_legend_handles_labels(),loc="lower left",bbox_to_anchor=(.065,.055),ncol=3,frameon=False,fontsize=10)
caption = ("Eight games per policy at 32,768 decisions and 16 at each later checkpoint."
           if confirm_ready else "Eight games per policy and 2,048 button decisions per game.")
fig.text(.075,.035,caption+" The full-gradient branch starts from the original world checkpoint.",fontsize=9,color="#434a50")
fig.savefig(project/"learning-curve.png",dpi=165,facecolor="white")
plt.close(fig)

report = """# Longer Emerald training and actor-gradient check

The pilot extended seed 19 from 32,768 to 98,304 real decisions. The original archive-PPO and world algorithms continued from their own saved weights, optimizer and archive. A third branch started from the same world checkpoint and switched to full actor backpropagation with an independent Adam optimizer. All three used identical new process seeds (17019 and 27019) for the two extension blocks, eight workers and the original three-step model horizon. Both extension boundaries began new worker episodes. The world learner also carried its real replay and model optimizer between blocks.

| Total real decisions | Learner | Fresh clock / 8 | Fresh starter / 8 | Fresh badge / 8 | Cumulative training wall time |
|---:|---|---:|---:|---:|---:|
"""
for i,budget in enumerate(budgets):
    for mode, rows in series.items():
        if mode=="full" and i==0: continue
        r=rows[i]
        assert r["steps"]==budget
        wall=sum(a["wall"] for a in rows[:i+1])
        report+=f"| {budget:,} | {labels[mode]} | {r['clocks']} | {r['starters']} | {r['badges']} | {wall/60:.2f} min |\n"
report+="""
The denominator is eight fresh game boots at each point. Evaluation used the same seed and 2,048 decisions per boot throughout. Each group of eight shares one set of trained weights. Separate seeds 37 and 71 were evaluated only at the initial training budget. Wall time includes training startup and checkpoint writes and excludes evaluation. Cumulative wall time for both world branches includes their shared initial training. The observed milestone is the bedroom clock. A starter and a badge are also tracked.

![Learning curves](learning-curve.png)

## What the controller learns from imagination

The stored-checkpoint control zeroed the imagined gradient and measured how far Adam still moved the actor. The live control repeats that comparison immediately after a real PPO update, using the actual rollout at the final update of each measured block. A cosine of one means the two parameter changes point in the same direction. The relative difference measures how far apart those changes are, divided by the size of the normal update.

| Cumulative real decisions | Size of zero-gradient update / normal | Direction cosine | Relative difference |
|---:|---:|---:|---:|
"""
for budget in budgets[1:]:
    path=long/f"world-{budget}"/"optimizer-control.json"
    if path.exists():
        d=read(path)
        report+=f"| {budget:,} | {d['zero_to_actual_norm']:.2%} | {d['step_cosine']:.5f} | {d['relative_difference']:.2%} |\n"
report+="""
The new branch gives direct credit to all three predicted actions. It also switches the optimizer and its learning rate and moments, so the comparison tests those changes together. The critic bootstrap remains frozen during the imagined update. The predictor sees an 8 × 12 RGB summary, current game variables, button and goal. The episode's visited cells, best prior progress and previous goal hits affect the training reward but are hidden from that predictor. Its estimate of those rewards must average over indistinguishable histories. The clock requires many decisions and the reward is sparse. A correct one-step movement prediction alone does not teach that sequence.

The [short-run report](WORLD_MODEL_RESULTS.md) covers the three independent seeds, model prediction, the exact archived-goal test and replay verification. The [YouTube diary](YOUTUBE_NOTES.md) links to native frames and controller recordings.
"""
report+="\nThe Emerald pilot used 128 starts × 3 imagined buttons per 512 real decisions after warmup. SuperMimic's published learner configuration defaults to 4,096 starts and 32 model steps, with a dedicated policy optimizer and actor gradients at every step. Those are software defaults for a continuous-control robotics system. Predicting nine discrete buttons, sparse story events and emulator state from a small screen entails a different prediction problem. Each of the Emerald pilot's arms still collected the same full real PPO rollout. Any elapsed-time advantage has to come from reaching the target in fewer such rollouts. [Original SuperMimic learner configuration](https://github.com/HybridRobotics/supermimic/blob/5fb92297ab4fe3d86629d9f0c635334791bb0d5f/src/supermimic/learners/policy_learner.py)\n"
first_step = [json.loads((long/f"{mode}-65536"/"metrics.jsonl").read_text().splitlines()[0]) for mode in ("world","full")]
r,f=first_step
assert r["ppo_loss"] == f["ppo_loss"] and abs(r["world_loss"]-f["world_loss"]) < 1e-6
report+=f"\nBoth world branches began from identical checkpoint weights. The first 512-decision rollout produced matching logged PPO loss ({r['ppo_loss']}) and predictor loss ({r['world_loss']:.6f}). At this update, the mean policy KL was {r['imagination_actor_kl']:.8f} for the original imagined update and {f['imagination_actor_kl']:.8f} for the full-gradient update. This measures how much the update changes action probabilities. Its gameplay value is measured by the fresh-boot trials.\n"
if all((long/f"full-{b}-confirm/evaluation.json").exists() for b in budgets[1:]):
    report+="\n## Replication on a second evaluation seed\n\nThe six saved extension policies each played eight additional games with evaluation seed 92741. These test games used the same 2,048-decision limit, zero restores and an empty goal input. The two evaluation batches provide 16 trials per frozen policy, all from the same training seed.\n\n| Real training decisions | Learner | Clocks on original 8 | Clocks on new 8 | Combined / 16 | Combined starters / 16 |\n|---:|---|---:|---:|---:|---:|\n"
    for j,b in enumerate(budgets[1:],start=1):
        for mode in series:
            extra=read(long/f"{mode}-{b}-confirm"/"evaluation.json")
            assert extra["seed"]==92741 and extra["resets"]==0 and len(extra["trials"])==8
            rr=series[mode][j]
            clocks=sum(t["clock"] for t in extra["trials"])
            starters=sum(t["starter"] for t in extra["trials"])
            report+=f"| {b:,} | {labels[mode]} | {rr['clocks']} | {clocks} | {rr['clocks']+clocks} | {rr['starters']+starters} |\n"
if (base/"reward-aliasing.json").exists():
    report+="\n## Identical inputs with different rewards\n\nThe 32,768-row replay buffers contained exactly identical encoded state, goal and button combinations associated with more than one real reward. Partial observability and differences in episode history can both contribute. These counts are a lower bound on ambiguity in the model's stored inputs.\n\n"
    for mode,v in read(base/"reward-aliasing.json").items():
        report+=f"- {mode} contained {v['ambiguous_inputs']} such inputs, accounting for {v['ambiguous_rows']:,} rows and {v['ambiguous_positive_rows']} of {v['positive_rows']} positive reward rows.\n"
if confirm_ready:
    def combined(mode, index):
        v=series[mode][index]
        extra=read(long/f"{mode}-{v['steps']}-confirm"/"evaluation.json")["trials"]
        return v["clocks"]+sum(t["clock"] for t in extra)
    conclusion=(f"The revised actor set the bedroom clock in **{combined('full',1)}/16** fresh games at "
                f"65,536 real training decisions and **{combined('full',2)}/16** at 98,304. "
                f"The original world update scored {combined('world',1)}/16 and {combined('world',2)}/16. "
                f"Archive-PPO scored {combined('archive',1)}/16 and {combined('archive',2)}/16. "
                "The win at both later checkpoints repeated across two evaluation seeds. "
                "One training seed supplied this comparison. Archive-PPO had already reached 6/8 on its initial checkpoint, "
                "and performance varied as training continued. Establishing a convergence-speed multiplier requires more independent training seeds and a sustained success threshold.\n\n")
    report=report.replace("The pilot extended", conclusion+"The pilot extended",1)
(project/"LONG_RUN_RESULTS.md").write_text(report,encoding="utf-8")
print(json.dumps(series,indent=1))
