"""Collect held-out story events and nearby controls for reward-model calibration."""
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

project = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL"
sys.path.insert(0, str(project))
from emerald import Emerald
from pixel_env import Score, game_paths

root = project / "runs/pixels-only-01/official-wsl"
out = root / "controls/heldout-rewards"
out.mkdir(exist_ok=True)


def collect(audit):
    evidence = json.loads(audit.read_text())
    tape = audit.with_suffix(".bin").read_bytes()
    assert len(tape) == evidence["steps"]
    samples = {}
    for event, at in evidence["events"].items():
        for offset in (-64, 0, 64):
            target = at + offset
            assert 32 <= target < len(tape)
            samples.setdefault(target, []).append((event, offset))
    images, rewards, actions = deque(maxlen=33), deque(maxlen=33), deque(maxlen=33)
    info, frames, buttons, targets = [], [], [], []
    game = Emerald(*game_paths())
    scorer = Score()
    try:
        game.load(game.boot)
        for n in range(max(samples)+1):
            if n:
                image, state = game.step(tape[n-1])
                reward, _ = scorer.read(game, state)
            else:
                image, reward = game.boot_image, 0.
            images.append(image)
            rewards.append(reward)
            actions.append(tape[n])
            for event, offset in samples.get(n, []):
                if offset and reward != 0:
                    continue
                assert len(images) == 33 and (offset or reward >= 1.)
                frame = np.zeros((33,64,64,3), np.uint8)
                frame[:,12:52,2:62] = np.stack(images)
                info.append(dict(event=event, offset=offset, decision=n, reward=reward))
                frames.append(frame)
                buttons.append(list(actions))
                targets.append(list(rewards))
        assert scorer.events == evidence["events"]
    finally:
        game.close()
    name = audit.parent.name
    np.savez_compressed(out / f"{name}.npz", image=frames, action=np.array(buttons,np.int32),
                        reward=np.array(targets,np.float32))
    return dict(env=name, source=audit.relative_to(root).as_posix(),
                tape_sha256=hashlib.sha256(tape).hexdigest(), windows=info)


if __name__ == "__main__":
    files = sorted((root/"seed-71299-h15/eval-65536-long/external-audit").glob("*/*/*episode-00001.json"),
                   key=lambda p: p.parent.name)
    assert len(files) == 4
    rows = [collect(path) for path in files]
    (out/"sources.json").write_text(json.dumps(rows,indent=2))
    print("Collected reward windows", [len(row["windows"]) for row in rows], flush=True)
