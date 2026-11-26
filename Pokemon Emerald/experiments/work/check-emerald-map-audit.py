"""Exercise a real emulator worker and compare its audit to the separate original-ROM replay."""
import json
from pathlib import Path
import sys
import numpy as np

project = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL"
sys.path.insert(0,str(project))
from pixel_env import PixelPool
from pixel_replay import replay
from train_earned_dreamer import write_json


def test():
    root = project / "runs/pixels-only-01/official-wsl"
    original = root / "reward-only-training-forks/verified-training-tapes/99e5799c1361793e8ab04ee204fe5c5e32a68b739ed100ed37fb0985d8196b80"
    target = root / "controls/human-map-audits"
    prior, = (original / "audit").glob("*.json")
    tape = prior.with_suffix(".bin").read_bytes()
    old = json.loads(prior.read_text())
    expected_rewards = []
    for chunk in sorted(original.glob("chunk-*.npz")):
        with np.load(chunk) as ds:
            expected_rewards.extend(ds["reward"])
    assert len(expected_rewards)==len(tape)
    env = PixelPool(1,uncapped_exploration=True)
    try:
        initial = env.command(0,"reset")
        assert initial.shape == (40,60,3)
        for button, expected in zip(tape,expected_rewards):
            rgb, reward, done = env.command(0,"act",button)
            assert rgb.shape == (40,60,3) and isinstance(reward,float) and type(done) is bool
            assert np.float32(reward)==expected
        env.command(0,"audit",str(target/"audit"))
        env.command(0,"audit_summary",str(target/"latest"))
    finally:
        env.close()
    audit, = (target / "audit").glob("*.json")
    data = json.loads(audit.read_text())
    assert audit.with_suffix(".bin").read_bytes() == tape and {k:data[k] for k in old} == old
    latest, = (target/"latest").glob("*.json")
    assert json.loads(latest.read_text()) == data and not list(latest.parent.glob("*.bin"))
    replay(audit,target/"replay","clock",False)
    result = json.loads((target/"replay/verification.json").read_text())
    maps = result["maps_first_reached"]
    assert maps == data["maps_first_reached"] and maps["LittlerootTown_BrendansHouse_2F"] == data["events"]["bedroom"]
    write_json(target/"verification.json",dict(buttons=len(tape), events=data["events"], maps_first_reached=maps,
        controller_observation="RGB, scalar reward and done", original_tape_and_rewards_preserved=True))


if __name__ == "__main__":
    test()
