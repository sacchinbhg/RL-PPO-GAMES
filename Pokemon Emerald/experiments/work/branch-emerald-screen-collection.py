"""Preserve the ordinary 262k learner and its training replay before it advances."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

root = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
parent = root / "seed-71299-h15"
variant = sys.argv[1] if len(sys.argv) > 1 else "screen-collection"
if variant not in ("screen-collection", "staggered-screen-collection"):
    raise ValueError(variant)
arm = root / f"seed-71299-h15-{variant}"
train = parent / "train"
step = 262152


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


arm.mkdir(exist_ok=True)
result = arm / "train"
if result.exists():
    current = json.loads((result / "branch-origin.json").read_text())
    assert current["transitions"] == step
else:
    reference = json.loads((parent / "eval-262144/provenance.json").read_text())
    checkpoint = (train / "ckpt/latest").read_text().strip()
    original = train / "ckpt" / checkpoint
    assert checkpoint == Path(reference["source_checkpoint"]).name
    assert reference["training_step"] == step
    assert digest(original / "agent.pkl") == reference["agent_sha256"]
    files = sorted((train / "replay").glob("*.npz"))
    assert sum(int(p.stem.rsplit("-", 1)[1]) for p in files) == step
    with tempfile.TemporaryDirectory(dir=arm) as stage:
        draft = Path(stage) / "train"
        shutil.copytree(original, draft / "ckpt" / checkpoint)
        (draft / "ckpt/latest").write_text(checkpoint)
        replay = draft / "replay"
        replay.mkdir()
        manifest = {}
        for path in files:
            target = replay / path.name
            shutil.copyfile(path, target)
            manifest[path.name] = digest(path)
            assert digest(target) == manifest[path.name]
        record = dict(parent_arm=parent.name, parent_training_steps=step, parent_checkpoint=checkpoint,
                      parent_weight_sha256=reference["agent_sha256"], transitions=step,
                      evaluation_transitions_used=0, pilot_transitions_used=0, source_replay_files=manifest)
        (draft / "branch-origin.json").write_text(json.dumps(record, indent=2))
        draft.rename(result)
print("Verified ordinary learner branch at", step, "training transitions.", flush=True)
