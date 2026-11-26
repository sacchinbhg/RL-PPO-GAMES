import json
from pathlib import Path
import sys
import tempfile
import numpy as np

project = Path(__file__).resolve().parents[1] / "outputs" / "EmeraldRL"
sys.path[:0] = [str(project), "/opt/emerald-rl/source"]
from embodied.core.replay import Replay
from pixel_env import Score
from reward_replay import install_reward_sampling


def transition(reward, first=False):
    return dict(reward=np.float32(reward), is_first=np.bool_(first), is_last=np.bool_(False))


def validate_boundaries():
    from types import SimpleNamespace
    env = SimpleNamespace(verified_story=lambda k: (False,False), ram=lambda a,n: bytes(n))
    scores = [Score(), Score(True)]
    for i in range(53):
        key = (5,0,2,i,2,1,0,0,0)
        r = [s.read(env,key)[0] for s in scores]
        assert r == ([.02,.02] if i < 50 else [0.,.02])
        assert [s.read(env,key)[0] for s in scores] == [0.,0.]
    with tempfile.TemporaryDirectory() as d:
        def build():
            replay = Replay(length=5, capacity=19, directory=d, chunksize=7, save_wait=True)
            return install_reward_sampling(replay, 1, 1., 172)
        r = build()
        for i in range(90):
            r.add(transition(float(i==73)), 0)
        assert r.stats()["reward_windows"] == 4
        assert (r.sample(10)["reward"][:,1:].max(1) == 1).all()
        r.save()
        restored = build()
        restored.load()
        assert restored.stats()["reward_windows"] == 4
        assert (restored.sample(10)["reward"][:,1:].max(1) == 1).all()
        for i in range(40):
            restored.add(transition(0.), 0)
        assert restored.stats()["reward_windows"] == 0
        assert not restored.sample(20)["reward"].any()


def main():
    validate_boundaries()
    official = project / "runs/pixels-only-01/official-wsl"
    paths = sorted((official / "seed-71299-h15/train/replay").glob("*.npz"))
    by_id = {p.stem.split("-")[1]:p for p in paths}
    successors = {p.stem.split("-")[2] for p in paths}
    roots = [p for p in paths if p.stem.split("-")[1] not in successors]
    assert len(roots) == 4
    replay = Replay(length=33, capacity=131072, online=False)
    install_reward_sampling(replay, 1, .25, 71299)
    peaks = []
    transitions = []
    for i, root in enumerate(roots):
        rewards = []
        p = root
        while p:
            with np.load(p) as data:
                if not rewards:
                    assert data["is_first"][0]
                rewards.extend(data["reward"].tolist())
            p = by_id.get(p.stem.split("-")[2])
        for index,reward in enumerate(rewards):
            replay.add(transition(reward,index == 0), i)
        transitions.extend(rewards)
        peaks.extend(np.lib.stride_tricks.sliding_window_view(rewards, 33)[:,1:].max(axis=1).tolist())
    peaks, transitions = np.asarray(peaks), np.asarray(transitions)
    uniform = float(np.mean(peaks >= 1))
    positive = float(peaks[peaks >= 1].sum()/peaks.sum())
    expected = .75*uniform + .25*positive
    trials = {}
    for mode in ("report","train"):
        sequence_counts, batch_counts = [], []
        for _ in range(1000):
            x = replay.sample(8, mode)["reward"][:,1:]
            sequence_counts.extend((x.max(axis=1)>=1).tolist())
            batch_counts.append(bool(x.max()>=1))
        trials[mode] = dict(sequences=len(sequence_counts), empirical_fraction=float(np.mean(sequence_counts)),
                            minibatches_with_story=float(np.mean(batch_counts)))
    assert abs(trials["train"]["empirical_fraction"]-expected)<.025
    assert abs(trials["report"]["empirical_fraction"]-uniform)<.01
    result=dict(source_checkpoint_step=len(transitions), positive_transitions=int((transitions>0).sum()),
                at_least_one_reward=int((transitions>=1).sum()), eligible_windows=len(peaks),
                story_windows=int((peaks>=1).sum()), exact_uniform_fraction=uniform,
                exact_reward_mixture_fraction=expected, actual_offline_draws=trials,
                queue="The online queue stays upstream-managed in actual training.")
    (official / "controls/reward-replay-sampling.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
