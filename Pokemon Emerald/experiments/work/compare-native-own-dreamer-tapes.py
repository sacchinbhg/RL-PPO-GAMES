"""Do original-ROM Dreamer action histories transfer to the faster native game build?"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from native_emerald import NativeEmerald

root=project/"runs/pixels-only-01/official-wsl"
parent=root/"seed-71299-h15-reward-replay/train"
fork=root/"seed-71299-h15-reward-control/train"
out=root/"controls/native-original-tape-transfer.json"
config=json.loads((project/"local.json").read_text(encoding="utf-8-sig"))


def main():
    origin=json.loads((fork/"branch-origin.json").read_text())
    files=[fork/"replay"/name for name in origin["source_replay_files"]]
    lookup={f.stem.split("-")[1]:f for f in files}
    following={f.stem.split("-")[2] for f in files}
    audits={p.with_suffix(".bin").read_bytes():p for p in (parent/"external-audit").glob("*/*/*.json")
            if json.loads(p.read_text())["steps"]==16384}
    assert len(audits)==4
    rows=[]
    for start in sorted(p for k,p in lookup.items() if k not in following):
        path=start
        stream=bytearray()
        original={}
        cutoffs=[0,32,128,512,1024,2048,4096,8192,16384]
        while path:
            with np.load(path) as chunk:
                a=chunk["action"]
                capture=[k for k in cutoffs if len(stream)<=k<len(stream)+len(a)]
                if capture:
                    imgs=chunk["image"]
                    for k in capture:
                        original[k]=imgs[k-len(stream),12:52,2:62].copy()
                stream.extend(a.astype(np.uint8).tobytes())
            path=lookup.get(path.stem.split("-")[2])
        tape=bytes(stream[:16384])
        audit=audits[tape]
        observed={}
        game=NativeEmerald(config["native_core"],config["rom"])
        compare={}
        try:
            for n in range(len(tape)+1):
                if n==0:
                    pic=game.boot_image
                else:
                    pic,key=game.step(tape[n-1])
                    story=game.verified_story(key)
                    if story is not None:
                        rival,party=story
                        for name,condition in (("bedroom",key[5]>=4),("clock",bool(key[6])),
                            ("sent_to_meet_rival",key[5]>=6 and bool(key[6])),("met_rival",rival),
                            ("route101",rival and game._values[10]>=1),("starter",party)):
                            if condition:
                                observed.setdefault(name,n)
                if n in original:
                    delta=np.abs(pic.astype(np.int16)-original[n].astype(np.int16))
                    compare[n]=dict(exact=not bool(delta.any()),mean_absolute_channel_error=float(delta.mean()))
        finally:
            game.close()
        rows.append(dict(env=audit.parent.name,source=audit.relative_to(root).as_posix(),
            tape_sha256=hashlib.sha256(tape).hexdigest(),native_events=observed,
            original_events=json.loads(audit.read_text())["events"],sampled_rgb=compare))
        print(audit.parent.name,observed,flush=True)
    out.write_text(json.dumps(dict(steps_per_game=16384,engine_sha256=hashlib.sha256(
        Path(config["native_core"]).read_bytes()).hexdigest(),input="four actual Dreamer training action histories",
        rows=rows),indent=2))


if __name__=="__main__":
    main()
