"""Check real zero-copy Dreamer carry restoration and original-ROM transitions in a training fork."""
from pathlib import Path
import sys
import numpy as np

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from fork_policy import ProposalEnv
from play_frozen_dreamer import load_actor
from reward_branch_env import ForkEnv
from reward_branch_search import write_json

def run():
    import jax
    root=project/"runs/pixels-only-01/official-wsl"
    fit=root/"earned-official-updates/adaptation/frozen-policy"
    actor=load_actor(fit,root/"seed-71299-h15/eval-65536",465721,Path("/opt/emerald-rl/source"),batch_size=1,with_forks=True)
    env=ProposalEnv(ForkEnv(),actor,np.random.default_rng(519))
    tape=(root/"paired-self-earned-frontiers/visual-slots-0/discoveries/work-0000139234/game.bin").read_bytes()
    try:
        for b in tape[:256]:
            env.command("step",(b,1))
        original=[np.asarray(v).copy() for v in jax.tree.leaves(actor[1]())]
        start=env.image.copy(),env.previous,env.incoming
        env.command("save",0)
        transitions=[]
        for b in tape[256:384]:
            env.propose()
            pic,rew,done,used=env.command("step",(b,1))
            transitions.append((pic.copy(),rew,done,used))
        env.command("restore",0)
        restored=jax.tree.leaves(actor[1]())
        assert len(original)==len(restored)>0 and all(np.array_equal(a,b) for a,b in zip(original,restored))
        assert np.array_equal(start[0],env.image) and (env.previous,env.incoming)==start[1:]
        for b,(expected,cents,terminal,units) in zip(tape[256:384],transitions):
            action=env.propose()
            assert env.propose()==action
            pic,rew,done,used=env.command("step",(b,1))
            assert np.array_equal(pic,expected) and (rew,done,used)==(cents,terminal,units)
        assert env.restores==1 and env.observed==512
        out=root/"controls/dreamer-training-forks"
        out.mkdir(parents=True,exist_ok=True)
        write_json(out/"verification.json",dict(real_decisions=512,matched_repeat_transitions=128,
                   exact_carry_arrays=len(original),exact_previous_button=env.saved[0][1],
                   single_model_observation_per_real_decision=True,policy_sampling_clock="continues across restores",
                   training_only=True,policy_seed=465721))
    finally:
        env.close()

if __name__=="__main__":
    run()
