"""Fork the reward-weighted learner at its first checkpoint with only its own replay."""
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import tempfile

base=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
parent=base/"seed-71299-h15-reward-replay"
variant=sys.argv[1] if len(sys.argv)>1 else "retained-rewards"
if variant not in ("retained-rewards","reward-control","retained-entropy"):
    raise ValueError("Unknown fork")
arm=base/f"seed-71299-h15-{variant}"
arm.mkdir(exist_ok=True)
dest=arm/"train"
provenance=json.loads((parent/"eval-65536/provenance.json").read_text())
assert provenance["training_step"]==65544 and provenance["reward_replay_fraction"]==.25
checkpoint=Path(provenance["source_checkpoint"]).name
frozen=parent/"eval-65536/frozen-policy"
with (frozen/"agent.pkl").open("rb") as src:
    digest=hashlib.file_digest(src,"sha256").hexdigest()
assert digest==provenance["agent_sha256"]
if dest.exists():
    record=json.loads((dest/"branch-origin.json").read_text())
    assert record["parent_weight_sha256"]==digest
    print("Verified existing 65,544-step learner fork.",flush=True)
    raise SystemExit

files=list((parent/"train/replay").glob("*.npz"))
by={f.stem.split("-")[1]:f for f in files}
succ={f.stem.split("-")[2] for f in files}
roots=sorted(f for k,f in by.items() if k not in succ and f.name.split("-")[0]<=checkpoint)
assert len(roots)==4
chosen=[]
for path in roots:
    chain=[]
    while path:
        assert path.name.split("-")[0]<=checkpoint
        chain.append(path)
        path=by.get(path.stem.split("-")[2])
    assert sum(int(f.stem.split("-")[3]) for f in chain)==16386
    chosen.extend(chain)
with tempfile.TemporaryDirectory(dir=arm) as stage:
    draft=Path(stage)/"train"
    ck=draft/"ckpt"/checkpoint
    replay=draft/"replay"
    ck.mkdir(parents=True)
    replay.mkdir()
    for name in ("agent.pkl","step.pkl","done"):
        shutil.copyfile(frozen/name,ck/name)
    # The pinned upstream Replay.save() returns None; its experience lives in the chunk files.
    (ck/"replay.pkl").write_bytes(pickle.dumps(None,protocol=4))
    (ck.parent/"latest").write_text(checkpoint)
    manifest={}
    for path in chosen:
        target=replay/path.name
        shutil.copyfile(path,target)
        with path.open("rb") as a,target.open("rb") as b:
            original=hashlib.file_digest(a,"sha256").hexdigest()
            assert hashlib.file_digest(b,"sha256").hexdigest()==original
        manifest[path.name]=original
    record=dict(parent_arm=parent.name,parent_training_steps=65544,parent_checkpoint=checkpoint,
                parent_weight_sha256=digest,training_seed=71299,transitions=65544,
                evaluation_transitions_used=0,source_replay_files=manifest)
    (draft/"branch-origin.json").write_text(json.dumps(record,indent=2))
    draft.rename(dest)
print("Copied and verified",len(chosen),"own training chunks, 65,544 transitions and matching optimizer checkpoint.",flush=True)
