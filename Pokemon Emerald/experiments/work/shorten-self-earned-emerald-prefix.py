"""Reward-only self-practice: shorten a learner's own rewarded button history on fresh boots."""
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

project = Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from pixel_env import PixelPool

root=project/"runs/pixels-only-01/official-wsl"
origin=root/"seed-71299-h15-reward-control/train"
out=root/"controls/self-earned-prefix"


def select_prefix():
    # This manifest was frozen before the continuation. Its files contain only the
    # 65,544 transitions from the original learner's four training games.
    manifest=json.loads((origin/"branch-origin.json").read_text())
    files=[origin/"replay"/name for name in manifest["source_replay_files"]]
    by={p.stem.split("-")[1]:p for p in files}
    following={p.stem.split("-")[2] for p in files}
    candidates=[]
    for start in sorted(p for k,p in by.items() if k not in following):
        actions, rewards=[],[]
        p=start
        while p:
            with p.open("rb") as f:
                assert hashlib.file_digest(f,"sha256").hexdigest()==manifest["source_replay_files"][p.name]
            with np.load(p) as z:
                actions.extend(z["action"])
                rewards.extend(z["reward"])
            p=by.get(p.stem.split("-")[2])
        assert len(actions)==16386
        at=next(i for i,r in enumerate(rewards) if r>=1)
        candidates.append((at,start.name,np.asarray(actions[:at],np.uint8).tobytes(),float(rewards[at])))
    assert len(candidates)==4
    return min(candidates,key=lambda c:c[0]),manifest


def main():
    out.mkdir(parents=True,exist_ok=True)
    if (out/"result.json").exists():
        return
    (n,chunk,source,reward),manifest=select_prefix()
    target=math.floor(reward)
    (out/"source.bin").write_bytes(source)
    (out/"origin.json").write_text(json.dumps(dict(source_chunk=chunk,
        parent_weights=manifest["parent_weight_sha256"], source_steps=n, recorded_scalar_reward=reward,
        target=target,selection="earliest reward >=1 across four original training games",
        source_sha256=hashlib.sha256(source).hexdigest()),indent=2))
    pool=PixelPool(1)
    trials=used=0
    def test(candidate,why):
        nonlocal trials,used
        pool.command(0,"reset")
        attained=False
        for i,a in enumerate(candidate,1):
            _,r,done=pool.command(0,"act",a)
            if r>=target:
                attained=True
                break
            if done:
                break
        used+=i
        trials+=1
        with (out/"trials.jsonl").open("a") as log:
            log.write(json.dumps(dict(trial=trials,reason=why,offered=len(candidate),executed=i,
                                      reached=attained,total_emulator_buttons=used))+"\n")
        if attained:
            pool.command(0,"audit",str(out/f"accepted-{trials:04}"))
            return candidate[:i]
        return None
    try:
        best=test(source,"unaltered training prefix")
        assert best==source
        parts=2
        # Each deletion is judged by the real scalar reward. Every single trial
        # starts from boot, and all retained actions came from the same own tape.
        while len(best)>=2 and used<90000:
            width=math.ceil(len(best)/parts)
            improved=False
            for i in range(0,len(best),width):
                proposal=best[:i]+best[i+width:]
                if not proposal or used+len(proposal)>90000:
                    continue
                candidate=test(proposal,f"remove [{i},{min(i+width,len(best))})")
                if candidate is not None:
                    best=candidate
                    (out/"best.bin").write_bytes(best)
                    print("self-earned prefix",len(best),"buttons, attempts",trials,flush=True)
                    parts=max(2,parts-1)
                    improved=True
                    break
            if not improved:
                if parts>=len(best) or used+max(1,len(best)-width)>90000:
                    break
                parts=min(len(best),parts*2)
        final=test(best,"independent fresh-boot recheck of shortest accepted candidate")
        assert final==best
        audit_dir=out/f"accepted-{trials:04}"
        audit,=audit_dir.glob("*.json")
        assert audit.with_suffix(".bin").read_bytes()==best
        (out/"best.bin").write_bytes(best)
        (out/"result.json").write_text(json.dumps(dict(source_steps=n,shortest_confirmed_steps=len(best),
            trials=trials,emulator_buttons=used,goal_scalar_threshold=target,
            original_training_only=True,restores_during_each_trial=0,
            final_audit=audit.relative_to(out).as_posix(),events=json.loads(audit.read_text())["events"],
            purpose="self-practice search diagnostic; no tape was fed to the official learner"),indent=2))
    finally:
        pool.close()


if __name__=="__main__":
    main()
