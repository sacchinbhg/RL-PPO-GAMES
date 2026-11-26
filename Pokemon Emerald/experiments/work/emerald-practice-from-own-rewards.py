"""Return to a self-earned training frontier by replaying its buttons from boot."""
from pathlib import Path
import fcntl
import hashlib
import importlib.util
import json
import sys
import numpy as np

work=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("screen_exploration",work/"emerald-screen-exploration.py")
explore=importlib.util.module_from_spec(spec)
spec.loader.exec_module(explore)
root=explore.root
out=root/"self-practice-original-rom"


def write_json(path,obj):
    temp=path.with_suffix(".partial.json")
    temp.write_text(json.dumps(obj,indent=2))
    temp.replace(path)


def initial_reference():
    directory=out/"initial"
    directory.mkdir(exist_ok=True)
    marker=directory/"reference.json"
    if marker.exists():
        return marker
    source=root/"seed-71299-h15-reward-control/train"
    manifest=json.loads((source/"branch-origin.json").read_text())
    lookup={p.stem.split("-")[1]:p for p in (source/"replay"/n for n in manifest["source_replay_files"])}
    following={p.stem.split("-")[2] for p in lookup.values()}
    candidates=[]
    for begin in sorted(p for k,p in lookup.items() if k not in following):
        p=begin
        rr=[]; aa=[]
        while p:
            with p.open("rb") as f:
                assert hashlib.file_digest(f,"sha256").hexdigest()==manifest["source_replay_files"][p.name]
            with np.load(p) as data:
                rr.extend(data["reward"]); aa.extend(data["action"])
            p=lookup.get(p.stem.split("-")[2])
        assert len(aa)==16386
        events=[(i,int(r)) for i,r in enumerate(rr[:16385]) if r>=1]
        last=events[-1][0]
        tape=np.asarray(aa[:last],np.uint8).tobytes()
        candidates.append((sum(v for _,v in events),-last,begin.name,events,tape))
    assert len(candidates)==4
    score,_,chunk,signature,tape=max(candidates,key=lambda c:c[:2])
    (directory/"buttons.bin").write_bytes(tape)
    write_json(marker,dict(score=score,steps=len(tape),signature=signature,buttons="buttons.bin",
        sha256=hashlib.sha256(tape).hexdigest(),source_training_chunk=chunk,
        source_checkpoint_sha256=manifest["parent_weight_sha256"],source_training_steps=manifest["parent_training_steps"],
        selection="Max cumulative whole reward units from rewards >=1 among four own training games. Ties favour shorter prefixes."))
    return marker


def choose_reference(base):
    refs=[base,*sorted(out.glob("trial-*/discoveries/*/reference.json"))]
    def key(p):
        row=json.loads(p.read_text())
        return row["score"],-row["steps"]
    return max(refs,key=key)


def trial(index,base):
    directory=out/f"trial-{index:03}"
    directory.mkdir(exist_ok=True)
    if (directory/"result.json").exists():
        return json.loads((directory/"result.json").read_text())
    intent=directory/"origin.json"
    if not intent.exists():
        write_json(intent,dict(reference=choose_reference(base).relative_to(out).as_posix(),seed=[48911,index]))
    origin=json.loads(intent.read_text())
    ref=out/origin["reference"]
    record=json.loads(ref.read_text())
    prefix=(ref.parent/record["buttons"]).read_bytes()
    assert hashlib.sha256(prefix).hexdigest()==record["sha256"] and len(prefix)==record["steps"]
    policy=explore.ScreenExplorer(np.random.SeedSequence(origin["seed"]))
    pool=explore.PixelPool(1)
    signature=[]
    score=0
    last_gain=len(prefix)
    terminal=False
    try:
        state=explore.screen_key(pool.command(0,"reset"))
        for n in range(1,131073):
            if n<=len(prefix):
                action=prefix[n-1]
                policy.visits[state]+=1
                policy.transitions.setdefault(state,{})
            else:
                action=policy.choose(state)
            pic,reward,terminal=pool.command(0,"act",action)
            following=explore.screen_key(pic)
            policy.observe(state,action,following,reward)
            state=following
            if reward>=1:
                signature.append([n,int(reward)])
                score+=int(reward)
            if n==len(prefix):
                assert signature==record["signature"] and score==record["score"]
            if n>len(prefix) and reward>=1:
                last_gain=n
                saved=directory/"discoveries"/f"step-{n:06}"
                pool.command(0,"audit",str(saved))
                tape,=saved.glob("*.bin")
                write_json(saved/"reference.json",dict(score=score,steps=n,signature=list(signature),
                    buttons=tape.name,sha256=hashlib.sha256(tape.read_bytes()).hexdigest(),
                    prefix_origin=origin["reference"],replayed_own_buttons=len(prefix),
                    new_controller_buttons=n-len(prefix)))
                print("New own reward",index,n,score,"after",n-len(prefix),"exploratory buttons",flush=True)
            if n%8192==0:
                write_json(directory/"status.json",dict(step=n,prefix_length=len(prefix),score=score,
                    autonomous_suffix=max(0,n-len(prefix)),latest_reward=signature[-1] if signature else None))
            if terminal or n-last_gain>=16384:
                break
        dest=directory/"full-game"
        pool.command(0,"audit",str(dest))
        tape,=dest.glob("*.bin")
        audit=json.loads(tape.with_suffix(".json").read_text())
        assert tape.read_bytes().startswith(prefix) and audit["steps"]==n
        result=dict(events=audit["events"],steps=n,score=score,replayed_own_buttons=len(prefix),
            exploratory_buttons=n-len(prefix),prefix_signature_verified=True,restores_during_game=0,
            terminal=bool(terminal),proof_type="training exploration, with self-generated action prefix")
        write_json(directory/"result.json",result)
        print("Trial",index,result,flush=True)
        return result
    finally:
        pool.close()


if __name__=="__main__":
    out.mkdir(parents=True,exist_ok=True)
    with (out/"run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base=initial_reference()
        for index in range(16):
            if trial(index,base)["terminal"]:
                break
