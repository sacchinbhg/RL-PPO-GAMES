"""Compare Dreamer and uniform buttons after the same frozen screen memory runs out of progress."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from earned_memory import Memory
from pixel_env import PixelPool
from play_frozen_dreamer import load_actor, write_json
from train_earned_dreamer import digest


def play(model, checkpoint, parent, upstream, output, limit, seed, patience, recover=False):
    if min(limit, patience) < 1:
        raise ValueError("The budget and memory patience must be positive")
    output.mkdir(parents=True, exist_ok=True)
    memory = Memory(model, seed)
    experiment = dict(arms=["dreamer", "uniform"], weight_sha256=digest(checkpoint/"agent.pkl"),
                      memory_sha256=memory.identity["model_sha256"], memory_sources=memory.identity["self_earned_tapes"],
                      seed=seed, decisions_per_game=limit, training_updates=0, in_game_restores=0,
                      memory_patience=patience, handoff="end of known trajectory or sustained failure to advance in any trace")
    recovery=None
    first_changed=False
    if recover:
        from visual_route_recovery import VisualRouteRecovery
        recovery=VisualRouteRecovery(memory)
        experiment["arms"]=["dreamer_with_visual_route_recovery","uniform"]
        experiment["recovery_fit"]=recovery.fit
    with (output/"provenance.json").open("x") as f:
        json.dump(experiment, f, indent=2)
    actor = load_actor(checkpoint, parent, seed, upstream, batch_size=2)
    rng = np.random.default_rng([seed, 781])
    pool = PixelPool(2, json.loads((parent/"provenance.json").read_text()).get("uncapped_exploration",False),
                     output/"first-maps",log_whiteouts=recover)
    try:
        frames = np.stack([pool.command(i,"reset") for i in range(2)])
        rewards = np.zeros(2, np.float32)
        highwater = [-1] * len(memory.traces)
        advanced = 0
        handoff = None
        buttons = None
        for step in range(1, limit+1):
            neural = actor(frames, rewards, step==1, previous_buttons=buttons)
            if handoff is None:
                if not np.array_equal(frames[0],frames[1]) or rewards[0]!=rewards[1]:
                    raise RuntimeError("Matched memory games diverged before handing over control")
                proposed = memory.choose(frames[0],rewards[0])
                if memory.current is not None:
                    j,t = memory.current
                    if t>highwater[j]:
                        highwater[j]=t
                        advanced=step
                ends = memory.expected is not None and memory.expected[1] >= len(memory.traces[memory.expected[0]]["buttons"])-1
                if ends or step-advanced>=patience:
                    handoff=step
                    write_json(output/"handoff.json",dict(first_new_action=step,shared_buttons=step-1,
                               reason="end of recorded trajectory" if ends else "no advance in known visual sequence",
                               furthest_training_position=highwater,decisions_since_advance=step-advanced))
                else:
                    buttons=np.full(2,proposed,np.int32)
                    memory.executed(proposed)
            if handoff is not None:
                buttons=np.array([neural[0],rng.integers(9)],np.int32)
                if recovery is not None:
                    alternative=recovery.choose(frames[0],rewards[0])
                    if alternative is not None:
                        if alternative!=buttons[0] and not first_changed:
                            first_changed=True
                            write_json(output/"first-recovery-change.json",dict(decision=step,neural_button=int(buttons[0]),
                                       route_button=alternative))
                        buttons[0]=alternative
            frames,rewards,done=pool.step(buttons)
            if step%8192==0 or step==limit or done.any() or step==handoff or (rewards>=1).any():
                for i in range(2):
                    dest=output/"games"/f"env-{i}"/f"step-{step:07d}"
                    pool.command(i,"audit",str(dest))
                    if done[i]:
                        record,=dest.glob("*.json")
                        subprocess.run([sys.executable,str(Path(__file__).with_name("pixel_replay.py")),str(record),
                                        str(dest/"badge-replay"),"--event","stone_badge","--video"],check=True)
            if step%1024==0 or step==limit or step==handoff or done.any():
                status=dict(decisions_per_game=step,handoff=handoff,memory_actions=dict(memory.usage),
                            actual_previous_actions_observed_by_world_model=max(0,step-1))
                if recovery is not None:
                    status["recovery_actions"]=dict(recovery.stats)
                write_json(output/"status.json",status)
            if done.any():
                break
        write_json(output/"result.json",dict(**status,badge_per_arm=list(map(bool,done))))
    finally:
        pool.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("model","checkpoint","parent","upstream","output"):
        parser.add_argument("--"+name,required=True,type=Path)
    parser.add_argument("--limit",type=int,default=65536)
    parser.add_argument("--seed",type=int,default=197461)
    parser.add_argument("--patience",type=int,default=512)
    parser.add_argument("--visual-route-recovery",action="store_true",
                        help="After the handoff, use learned screen transitions to revisit late training screens from familiar places")
    a=parser.parse_args()
    play(a.model,a.checkpoint,a.parent,a.upstream,a.output,a.limit,a.seed,a.patience,a.visual_route_recovery)
