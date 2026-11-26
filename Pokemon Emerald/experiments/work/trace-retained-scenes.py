"""Match the historical reward windows to the original-ROM scenes that produced them."""
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np

project = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL"
sys.path.insert(0, str(project))
from emerald import Emerald, PHASES
from pixel_env import Score, game_paths

root = project / "runs/pixels-only-01/official-wsl"
branch = root / "seed-71299-h15-retained-rewards/train"


def digest(pixels, buttons, rewards):
    h = hashlib.sha256()
    for data in (np.asarray(pixels, np.uint8), np.asarray(buttons, np.int32),
                 np.asarray(rewards, np.float32)):
        h.update(data.tobytes(order="C"))
    return h.hexdigest()


def find_matches(path, wanted):
    evidence = json.loads(path.read_text())
    tape = path.with_suffix(".bin").read_bytes()
    assert len(tape) == evidence["steps"]
    game = Emerald(*game_paths())
    score = Score()
    images, phases, rewards = (deque(maxlen=33) for _ in range(3))
    latest_menu = None
    try:
        game.load(game.boot)
        for n in range(1, max(evidence["events"].values()) + 1):
            pic, key = game.step(tape[n-1])
            reward, _ = score.read(game, key)
            phase = PHASES[key[0]]
            if phase in ("title", "main-menu"):
                latest_menu = n
            images.append(pic)
            phases.append(phase)
            rewards.append(reward)
            if reward >= 1 and len(images) == 33:
                buttons = np.frombuffer(tape[n-32:n+1], np.uint8)
                keyhash = digest(np.stack(images), buttons, rewards)
                if keyhash in wanted:
                    assert wanted[keyhash] is None
                    wanted[keyhash] = dict(source=path.relative_to(root).as_posix(), end_decision=n,
                        events=[name for name, at in score.events.items() if at == n],
                        reward=reward, phases=Counter(phases), last_title_or_menu=latest_menu)
        assert score.events == evidence["events"]
    finally:
        game.close()


if __name__ == "__main__":
    cp = branch/"ckpt"
    checkpoint = cp/(cp/"latest").read_text().strip()
    assert (checkpoint/"done").is_file()
    with (checkpoint/"replay.pkl").open("rb") as f:
        examples = pickle.load(f)["retained_v1"]
    wanted = {digest(x["image"][:,12:52,2:62], x["action"], x["reward"]): None for x in examples}
    assert len(wanted) == len(examples)
    for source, length in ((root/"seed-71299-h15-reward-replay/train",16384), (branch,32768)):
        tapes = sorted((source/"external-audit").glob("*/*/*.json"))
        batch = [p for p in tapes if json.loads(p.read_text())["steps"] == length]
        for p in batch:
            if json.loads(p.read_text())["events"]:
                find_matches(p,wanted)
    assert all(wanted.values()), f"Matched {sum(bool(v) for v in wanted.values())}/{len(wanted)}"
    counts = Counter()
    events = Counter()
    for row in wanted.values():
        counts.update(row["phases"])
        events.update(row["events"])
    output = dict(checkpoint=checkpoint.name, windows=len(wanted), frames=sum(counts.values()),
                  observed_phases=counts, ending_events=events,
                  matched_by="SHA-256 of all 33 recorded RGB images, outgoing actions and scalar rewards",
                  examples=wanted)
    dest = root/"controls"/f"retained-scene-coverage-{len(wanted)}.json"
    dest.write_text(json.dumps(output, indent=2))
    print(dest.name, counts, events, flush=True)
