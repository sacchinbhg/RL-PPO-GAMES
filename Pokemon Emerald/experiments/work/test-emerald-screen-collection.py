"""Check upstream replay, emulator tapes, and actual previous actions with a real CUDA learner."""
from collections import Counter
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

project = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL"
sys.path.insert(0, str(project))
import screen_explorer
import upstream_emerald


def run():
    import jax
    root = project / "runs/pixels-only-01/official-wsl/controls/screen-collection-smoke/checked"
    install = screen_explorer.install_screen_collection
    checks = Counter()
    def checked_install(agent, count, seed):
        original = agent.policy
        def remember(carry, obs, mode="train"):
            result = original(carry, obs, mode=mode)
            checks["stock_batches"] += 1
            remember.actions = result[1]["action"].copy()
            return result
        agent.policy = remember
        install(agent, count, seed)
        wrapped = agent.policy
        def checked(carry, obs, mode="train"):
            carry, acts, outs = wrapped(carry, obs, mode=mode)
            current = acts["action"]
            np.testing.assert_array_equal(jax.device_get(carry[-1]["action"]), current)
            np.testing.assert_array_equal(current[count:], remember.actions[count:])
            checks["explorer_choices_different"] += int((current[:count] != remember.actions[:count]).sum())
            return carry, acts, outs
        agent.policy = checked
        return agent
    sys.argv = ["upstream_emerald.py", "--upstream", "/opt/emerald-rl/source", "--logdir", str(root),
                "--platform", "cuda", "--envs", "4", "--screen-exploration-envs", "2",
                "--steps", "396", "--batch", "4", "--length", "8", "--ratio", "8",
                "--seed", "49187", "--episode", "32", "--parallel-envs"]
    with patch.object(screen_explorer, "install_screen_collection", checked_install):
        upstream_emerald.main()
    audits = Counter(p.read_bytes() for p in (root / "external-audit").rglob("*.bin"))
    files = {p.stem.split("-")[1]: p for p in (root / "replay").glob("*.npz")}
    successors = {p.stem.split("-")[2] for p in files.values()}
    sequences = Counter()
    for key in files.keys() - successors:
        sequence = []
        while key in files:
            path = files[key]
            with np.load(path) as data:
                for action, first, last in zip(data["action"], data["is_first"], data["is_last"]):
                    if first:
                        sequence = []
                    if last:
                        sequences[bytes(sequence)] += 1
                    else:
                        sequence.append(int(action))
            key = path.stem.split("-")[2]
    assert sum(audits.values()) >= 8 and sequences == audits and checks["explorer_choices_different"] > 0
    record = dict(checks=checks, episodes=sum(audits.values()), buttons_per_episode=32,
                  replay_matches_emulator=True, training_platform="cuda")
    (root / "verification.json").write_text(json.dumps(record, indent=2))
    print(record, flush=True)


if __name__ == "__main__":
    run()
