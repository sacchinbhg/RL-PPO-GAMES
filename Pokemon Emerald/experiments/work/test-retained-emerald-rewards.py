from pathlib import Path
import pickle
import sys
import tempfile
import numpy as np

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path[:0]=[str(project),"/opt/emerald-rl/source"]
from embodied.core.replay import Replay
from reward_replay import install_reward_sampling, RetainedRewardWindows


def transition(i, reward):
    return dict(reward=np.float32(reward), is_first=np.bool_(i==0), is_last=np.bool_(False),
                action=np.int32(i%9), image=np.array([i],np.int32))


with tempfile.TemporaryDirectory() as directory:
    def build(online=False):
        r=Replay(length=5,capacity=19,directory=directory,chunksize=7,online=online,save_wait=True)
        return install_reward_sampling(r,1,1.,71299,retain=True)
    r=build()
    positives={9:1.,17:4.,29:3.}
    for i in range(110):
        r.add(transition(i,positives.get(i,.02 if i%8==0 else 0.)),0)
    report=r.stats()
    assert report["retained_windows"]==3 and report["live_large_windows"]==0
    sampled=r.sample(96)
    for ims,rew,acts in zip(sampled["image"][:,:,0],sampled["reward"],sampled["action"]):
        end=int(ims[-1])
        assert rew[-1]==positives[end] and np.array_equal(ims,np.arange(end-4,end+1))
        assert np.array_equal(acts,ims%9)
    assert r.stats()["retained_draws"]==96
    assert np.all(r.sample(100,mode="report")["reward"]<1) and np.all(r.sample(100,mode="eval")["reward"]<1)
    state=pickle.loads(pickle.dumps(r.save()))
    fresh=build(online=True)
    fresh.load(state)
    assert fresh.stats()["retained_windows"]==3
    assert np.all(fresh.sample(100)["reward"][:,-1]>=1)
    fresh.stats()
    # A real online row bypasses all offline selectors.
    fresh.queue.append(fresh.items[max(fresh.items)])
    batch=fresh.sample(2)["reward"]
    assert batch[0].max()<1 and batch[1,-1]>=1 and fresh.stats()["retained_draws"]==1
    for i in range(110,121):
        fresh.add(transition(i,4. if i==116 else 0.),0)
    fresh.queue.clear()
    assert fresh.stats()["live_large_windows"]>0
    fresh.sample(40)
    assert fresh.stats()["retained_draws"]==0
    assert fresh.stats()["retained_windows"]==4

def candidate(i):
    return dict(reward=np.array([1.],np.float32),
                stepid=np.frombuffer(i.to_bytes(20,"big"),np.uint8).reshape(1,20))
first,reverse=RetainedRewardWindows(71299),RetainedRewardWindows(71299)
for i in range(100): first.add(candidate(i))
for i in reversed(range(100)): reverse.add(candidate(i))
assert len(first.windows())==8 and first.bands[0].keys()==reverse.bands[0].keys()
print("Validated authentic sequences, FIFO eviction, capped retention, checkpoint reload, report separation, online priority and renewed live rewards.")
