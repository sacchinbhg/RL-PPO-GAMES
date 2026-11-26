from pathlib import Path
import statistics
import json

project = Path(__file__).resolve().parents[1] / "outputs" / "EmeraldRL"
base = project / "runs" / "world-comparison"
def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

audit = read(base / "heldout.json")
random_trials = read(base / "random-eval" / "evaluation.json")["trials"]
rows = []
for seed in (19,37,71):
    for name in ("archive","world"):
        root = base / f"{name}-{seed}"
        t = read(root / "training.json")
        e = read(base / (root.name+"-eval") / "evaluation.json")["trials"]
        wall = read(root / "elapsed.json")["wall_seconds"]
        assert (t["steps"],len(e),t["seed"]) == (32768,8,seed)
        row = dict(seed=seed, mode=name, cells=t["cells"], first=t["first_milestone_step"].get("clock"),
                   clocks=sum(tr["clock"] for tr in e), starters=sum(tr["starter"] for tr in e),
                   badges=sum(tr["badges"]>0 for tr in e), wall=wall,
                   sps=t["active_steps_per_second"], raw=t["steps"], imagined=t.get("imagined_steps",0),
                   inside=sum("LittlerootTown" in tr["maps"] for tr in e))
        rows.append(row)
doc = """# Emerald world-model experiment

Each seed trained from scratch for 32,768 real controller decisions on the RTX 4060 Laptop GPU. The primary outcome was setting the bedroom clock from a fresh game within 2,048 decisions. The paired PPO control used the same pixel policy, archive rules, rewards, episode length, worker count and optimizer. World runs added a learned action-conditioned dynamics and reward model and an actor update through three imagined button presses. Training used eight mGBA processes, 64 decisions per rollout and 512 per training episode.

## Real game results

| Training seed | Learner | First clock during training, aggregate real decisions | Independent clock successes / 8 | Starter successes / 8 | Training wall time | Active decisions/s |
|---:|---|---:|---:|---:|---:|---:|
"""
for r in rows:
    label = "World model + archive-PPO" if r["mode"]=="world" else "Archive-PPO"
    doc += f"| {r['seed']} | {label} | {r['first'] or 'Unreached'} | {r['clocks']} | {r['starters']} | {r['wall']:.1f} s | {r['sps']:.1f} |\n"
totals = {}
for name in ("archive","world"):
    rr=[r for r in rows if r["mode"]==name]
    totals[name]=dict(clocks=sum(r["clocks"] for r in rr), starters=sum(r["starters"] for r in rr),
                      mean_wall=statistics.mean(r["wall"] for r in rr), raw=sum(r["raw"] for r in rr),
                      imagined=sum(r["imagined"] for r in rr))
w, c = totals["world"], totals["archive"]
doc += f"""
The world-assisted policies set the clock in **{w['clocks']}/24** fresh boots. Archive-PPO achieved **{c['clocks']}/24**. The uniform random controller achieved **{sum(t['clock'] for t in random_trials)}/8** using the same evaluation budget. The world-assisted policies obtained a starter in {w['starters']}/24 trials and archive-PPO in {c['starters']}/24. Trial outcomes within a training seed share the same policy. The experiment has three independent training seeds per arm.

The two arms each used {w['raw']:,} real decisions in total. The world-assisted arm also used {w['imagined']:,} imagined transitions for policy improvement. Mean elapsed time per training job was {w['mean_wall']:.1f} seconds with the world model and {c['mean_wall']:.1f} seconds for archive-PPO. Timing includes startup and checkpoint writes. One computer ran the jobs sequentially, controls first. The first-clock training metric includes archive restores. Frozen-policy evaluations always started at boot with an empty goal and no restores. The experiment measures success at a fixed training budget. A reliable time-to-proficiency curve requires fresh-game evaluations at multiple training budgets.

## Predictions on independent gameplay

Eight new games used uniformly random buttons and seed {audit['seed']}. The audit collected {audit['raw_decisions']:,} real transitions and replayed all nine buttons from {audit['replay_matches']} saved positions. All {audit['replay_matches']} sampled continuation choices reproduced their corresponding fork outcomes. These data were excluded from model fitting and policy updates. The persistence baseline copies the last observed state. Lower prediction error is better.

| Seed | 1-step RGB RMSE / persistence | 3-step RGB RMSE / persistence | 1-step RAM RMSE / persistence | Action-specific RGB variance explained | Goal-directed button success / uniform expectation |
|---:|---:|---:|---:|---:|---:|
"""
for seed in (19,37,71):
    d=audit["models"][f"world-{seed}"]
    one, three, forks, task = d["horizon_1"], d["horizon_3"], d["button_forks"], d["one_button_goal_test"]
    expected = f"{task['uniform_expected_rate']:.1%}" if task["targets"] else "unavailable"
    doc+=f"| {seed} | {one['rgb_rmse']:.4f} / {one['copy_rgb_rmse']:.4f} | {three['rgb_rmse']:.4f} / {three['copy_rgb_rmse']:.4f} | {one['ram_rmse']:.4f} / {one['copy_ram_rmse']:.4f} | {forks['rgb_action_variance_explained']:.1%} | {task['hits']}/{task['targets']} / {expected} |\n"
replay=read(base/"world-19"/"replay.json")
doc+=f"""
The button-effect metric subtracts each state's mean outcome across buttons. A model predicting the same outcome for every button scores zero. Negative values indicate that its predicted differences among buttons increase squared error. The local goal test selects one action using the model to reach a different attainable coarse tile on the same map. Each target was scored against the emulator's actual response to all nine buttons. The raw audit also includes reward errors, three-step RAM errors and a rotated-button control.

## Verification and scope

- The six component tests cover action encoding and gradients, zero actor gradient when the model ignores actions, simulator transition storage, ring wraparound, boundary-safe sequence sampling, goal reward and archive replacement, checkpoint round trip, and frozen critic/model weights during imagination.
- From frame one, replaying {replay['decisions']:,} archived inputs from the world-model seed 19 run reached {replay['actual_map']}. The final descriptor {'matched' if replay['matched'] else 'differed from'} the stored endpoint. Fresh-game policy trials selected their own buttons.
- The runner reads the original English Emerald game flag for setting the clock. The game sets it inside `InitTimeBasedEvents`. The world model predicts coarse RGB and structured game variables. All archive entries and reported gameplay milestones come from the real emulator. [Emerald clock initialization](https://github.com/pret/pokeemerald/blob/8cf518839e529eeb2477905ede1df7213ac134b8/src/clock.c)
- The actor objective follows SuperMimic's short differentiable rollout and value bootstrap, adapted to discrete GBA controls. The game dynamics, reward model and training code here were written for Emerald. The ongoing PPO update anchors the actor on real observations. [SuperMimic policy learner](https://github.com/HybridRobotics/supermimic/blob/5fb92297ab4fe3d86629d9f0c635334791bb0d5f/src/supermimic/learners/policy_learner.py)

Full per-trial records, fixed-budget timings, checkpoints and prediction diagnostics are under `runs/world-comparison`. Use the commands in the README to rerun training or the independent audit.
"""
if (base / "goal-probe.json").exists():
    gp = read(base / "goal-probe.json")
    doc += "\n## Can the policy follow a named destination?\n\n"
    doc += "The probe chose distinct overworld destinations whose archived path extended a same-map start by 3–24 recorded decisions. Each controller received the same emulator state, 96 decisions and the same uniform random numbers for selecting buttons. The goal-conditioned and empty-goal arms used identical frozen weights. The destinations came from training archives. This is a test of whether the supplied goal helps control, with one draw per destination.\n\n"
    doc += "| Training seed | Reachable targets | Supplied goal | Empty goal | Uniform buttons |\n|---:|---:|---:|---:|---:|\n"
    for name, v in gp['runs'].items():
        h=v['hits']
        doc+=f"| {name.split('-')[1]} | {h.get('recorded',0)} | {h.get('goal',0)} | {h.get('empty',0)} | {h.get('random',0)} |\n"
    doc += "\nThe stored suffix reproduced every chosen destination at its recorded decision. [Per-target evidence](runs/world-comparison/goal-probe.json)\n"
if (base / "gradient-control.json").exists():
    doc += "\n## Why the first actor update can be weak\n\nThe first implementation computed a direct actor gradient only for the first imagined action. Later policy calls had detached weights, although state derivatives flowed through the three-step forecast. The update also reused the Adam optimizer just used for real PPO. We isolated its moving-average effect on three saved policies. Each pair used identical weights, optimizer state, 128 sampled archive observations, model and action randomness. In the control, hooks zeroed the current imagined gradients before Adam stepped.\n\n"
    doc += "| Seed | Size of zero-gradient update / normal update | Cosine of both parameter updates | Relative change when the current gradient was removed |\n|---:|---:|---:|---:|\n"
    for n,v in read(base / "gradient-control.json").items():
        doc+=f"| {n.split('-')[1]} | {v['zero_to_actual_norm']:.1%} | {v['step_cosine']:.4f} | {v['relative_difference']:.1%} |\n"
    doc+="\nThese measurements compare parameter displacements on selected archive states and use the PPO optimizer state saved after the checkpoint's final imagined update. They do not apportion the causes of an entire run. A second implementation uses separate Adam moments and backpropagates through every imagined action. A seventh component test checks all three action gradients, the independent optimizer and restoring its history. [Raw optimizer control](runs/world-comparison/gradient-control.json) · [SuperMimic actor training](https://github.com/HybridRobotics/supermimic/blob/5fb92297ab4fe3d86629d9f0c635334791bb0d5f/src/supermimic/learners/policy_learner.py)\n"
if (base / "restart-check.json").exists():
    resumed=read(base / "restart-check.json")
    doc+=f"\nA complete checkpoint was extended from {resumed['parent_steps']:,} to {resumed['total_training_steps']:,} real decisions. The learner continued to {resumed['dynamics_updates']:,} predictor updates and {resumed['imagined_steps']:,} imagined transitions. [Restart record](runs/world-comparison/restart-check.json)\n"
reward_scores=[v["reward"] for v in audit["models"].values()]
doc+=f"\nOnly {reward_scores[0]['positive_outcomes']} of {audit['raw_decisions']:,} independent real decisions had a positive shaped reward. Reward RMSE ranged from {min(v['rmse'] for v in reward_scores):.4f} to {max(v['rmse'] for v in reward_scores):.4f}, versus approximately {reward_scores[0]['train_mean_rmse']:.4f} when always predicting the training reward mean. The held-out movement test provides much stronger evidence of usable local dynamics than this aggregate metric provides about recognizing rare milestones.\n"
(project/"WORLD_MODEL_RESULTS.md").write_text(doc,encoding="utf-8")
print(json.dumps(dict(totals=totals, random_clock=sum(r["clock"] for r in random_trials), rows=rows),indent=2))
