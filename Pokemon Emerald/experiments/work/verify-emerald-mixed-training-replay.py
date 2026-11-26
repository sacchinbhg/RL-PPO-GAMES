"""Match each real training episode to its exact experience in the official learner replay."""
import hashlib
import json
from pathlib import Path
import numpy as np

base = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
arm = base / "seed-71299-h15-screen-collection"


def run():
    train = arm / "train"
    finish = json.loads((train / "final-checkpoint.json").read_text())
    assert finish["training_step"] >= 524300
    branch = json.loads((train / "branch-origin.json").read_text())
    inherited = set(branch["source_replay_files"])
    files = [p for p in (train / "replay").glob("*.npz") if p.name not in inherited]
    lookup = {p.stem.split("-")[1]:p for p in files}
    successors = {p.stem.split("-")[2] for p in files}
    audits = {}
    for path in (train / "external-audit").glob("*/*/*episode-00001.bin"):
        tape = path.read_bytes()
        assert len(tape) == 65536
        audits[hashlib.sha256(tape).hexdigest()] = (path, tape)
    assert len(audits) == 4
    rows = []
    total = 0
    for root in sorted(lookup.keys() - successors):
        p = lookup[root]
        acts, rewards, first, last = [], [], [], []
        while p:
            with np.load(p) as data:
                for dest, key in [(acts,"action"),(rewards,"reward"),(first,"is_first"),(last,"is_last")]:
                    dest.append(data[key])
            p = lookup.get(p.stem.split("-")[2])
        acts, rewards, first, last = map(np.concatenate, (acts, rewards, first, last))
        total += len(acts)
        for start in np.flatnonzero(first):
            end = int(start) + 65536
            if end >= len(acts) or not last[end]:
                continue
            assert not np.any(first[start+1:end+1])
            data = acts[start:end].astype(np.uint8).tobytes()
            sha = hashlib.sha256(data).hexdigest()
            path, tape = audits[sha]
            assert tape == data
            events = json.loads(path.with_suffix(".json").read_text())["events"]
            event_rewards = {k:float(rewards[int(start)+i]) for k,i in events.items()}
            if "starter" in event_rewards:
                assert event_rewards["starter"] >= 10 and event_rewards["lab_handoff"] >= 4
            worker = int(path.parent.name.split("-")[-1])
            rows.append(dict(worker=worker, behavior="screen explorer" if worker<2 else "Dreamer actor",
                             matching_buttons=len(data), tape_sha256=sha, rewards=event_rewards, events=events))
    assert sorted(r["worker"] for r in rows) == list(range(4))
    assert total == finish["training_step"] - branch["parent_training_steps"]
    result = dict(training_step=finish["training_step"], parent_step=branch["parent_training_steps"],
                  new_transitions=total, new_replay_chunks=len(files), games=sorted(rows,key=lambda r:r["worker"]))
    (arm / "verified-training-experience.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    run()
