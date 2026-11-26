"""Independent diagnostic of which buttons actually change rewards near held-out events."""
from collections import defaultdict, deque
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from emerald import Emerald
from pixel_env import Score, game_paths

root=project/"runs/pixels-only-01/official-wsl"
out=root/"controls/heldout-action-rewards"
horizon=15


def collect(audit):
    evidence=json.loads(audit.read_text())
    tape=audit.with_suffix(".bin").read_bytes()
    assert len(tape)==evidence["steps"]
    targets=defaultdict(list)
    for event,at in evidence["events"].items():
        for offset in (0,4,12):
            targets[at-offset].append((event,at,offset))
    game=Emerald(*game_paths())
    scorer=Score()
    rows, contexts, past, future, real_rewards=[],[],[],[],[]
    observed_rewards=[0.]
    actual_end_images={}
    try:
        game.load(game.boot)
        images=deque([game.boot_image],maxlen=33)
        for step in range(1,max(targets)+horizon):
            if step in targets:
                assert len(images)==33
                snap=game.save()
                recorded=tape[step-1:step-1+horizon]
                assert len(recorded)==horizon
                sequences=[]; returns=[]; scored=[]; recorded_digest=None
                for first in range(9):
                    game.load(snap)
                    branch=copy.deepcopy(scorer)
                    seq=bytes([first])+recorded[1:]
                    rr=[]
                    for button in seq:
                        pic,key=game.step(button)
                        reward,_=branch.read(game,key)
                        rr.append(reward)
                    if first==recorded[0]:
                        recorded_digest=hashlib.sha256(pic.tobytes()).hexdigest()
                    sequences.append(list(seq))
                    returns.append(rr)
                    scored.append(dict(branch.events))
                game.load(snap)
                for event,at,offset in targets[step]:
                    rows.append(dict(env=audit.parent.name,event=event,event_at=at,offset=offset,button_at=step,
                        original_button=recorded[0],event_decisions=[e.get(event) for e in scored],
                        original_final_image_sha256=recorded_digest))
                    contexts.append(np.stack(images))
                    past.append(list(tape[step-33:step-1]))
                    future.append(sequences)
                    real_rewards.append(returns)
            pic,key=game.step(tape[step-1])
            reward,_=scorer.read(game,key)
            observed_rewards.append(reward)
            images.append(pic)
            if step-horizon+1 in targets:
                actual_end_images[step]=hashlib.sha256(pic.tobytes()).hexdigest()
        assert scorer.events==evidence["events"]
    finally:
        game.close()
    for row, branches in zip(rows,real_rewards):
        s,a=row["button_at"],row["original_button"]
        assert np.array_equal(branches[a],observed_rewards[s:s+horizon])
        assert row["event_decisions"][a]==row["event_at"]
        assert row["original_final_image_sha256"]==actual_end_images[s+horizon-1]
    env=audit.parent.name
    np.savez_compressed(out/f"{env}.npz", context=np.asarray(contexts,np.uint8),
                        past=np.asarray(past,np.int32), future=np.asarray(future,np.int32),
                        real_rewards=np.asarray(real_rewards,np.float32))
    path=out/f"{env}.json"
    path.write_text(json.dumps(dict(source=audit.relative_to(root).as_posix(),
        source_tape_sha256=hashlib.sha256(tape).hexdigest(),windows=rows),indent=2))
    print(env,len(rows),"real action alternatives verified",flush=True)


if __name__=="__main__":
    out.mkdir(parents=True,exist_ok=True)
    audits=sorted((root/"seed-71299-h15/eval-65536-long/external-audit").glob("*/*/*.json"),key=lambda p:p.parent.name)
    assert len(audits)==4
    for audit in audits:
        if not (out/f"{audit.parent.name}.json").exists():
            collect(audit)
    records=[]
    for audit in audits:
        name=audit.parent.name
        rows=json.loads((out/f"{name}.json").read_text())["windows"]
        with np.load(out/f"{name}.npz") as z:
            for row,branches in zip(rows,z["real_rewards"]):
                totals=branches.sum(axis=1)
                times=row["event_decisions"]
                records.append(dict(**row,alternative_actions_change_event_presence=len(set(t is None for t in times))>1,
                    alternative_actions_change_event_timing=len(set(times))>1,
                    alternative_actions_change_total_reward=float(np.ptp(totals))>0.001,
                    total_rewards=[round(float(x),3) for x in totals]))
    assert len(records)==39
    (out/"real-outcomes.json").write_text(json.dumps(dict(horizon=horizon,windows=records,
        validation="Original action matched every real reward, event timing and final RGB image",
        use="Independent read-only diagnostic. All nine actions share a start. The following 14 buttons are fixed.",
        used_in_training=False),indent=2))
