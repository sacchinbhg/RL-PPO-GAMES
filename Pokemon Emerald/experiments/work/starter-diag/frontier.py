"""Bounded search from an agent-discovered, post-rival town state using real mGBA."""
from collections import Counter
import json, random, struct
from pathlib import Path
import sys
from PIL import Image
project=Path(__file__).resolve().parents[2]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from emerald import Emerald
from experiment import location

cfg=json.loads((project/"local.json").read_text(encoding="utf-8-sig"))
env=Emerald(cfg["core"],cfg["rom"])
out=project/"starter-diagnostics"
prefix=Path(__file__).with_name("mgba-town_after_rival.bin").read_bytes()
def rd(a,n):
    return int.from_bytes(env.ram(a,n),"little")
def facts():
    k=env.descriptor()
    name=location(k)
    if not name or env._header(*k[1:3])!=(rd(0x0203731c,4),rd(0x0203732a,2)):
        return None
    sb=rd(0x03005d8c,4)
    rc=rd(sb+0x139c+2*0x60,2)
    rival=rd(sb+0x139c+2*0x8d,2)>=3 and rd(sb+0x139c+2*0x50,2)>=1
    party=rd(0x020244e9,1)
    x,y=struct.unpack("<hh",env.ram(sb,4))
    oid=rd(0x02037590+5,1)
    facing=rd(0x02037350+36*oid+0x18,1)&15 if oid<16 else 0
    bag=bool(k[7])
    tier=1+int(name=="Route101")+int(name=="Route101" and rc>=2)+int(name=="Route101" and bag) if name in ("Route101","LittlerootTown") else 0
    sig=(name,x,y,facing,rc,bag,k[0],rd(0x030022c4,4))
    return sig,tier,dict(map=name,route_state=rc,bag=bag,party=party,met_rival=rival,phase=k[0])

def run():
    for a in prefix: env.step(a)
    sig,tier,start=facts()
    assert start["map"]=="LittlerootTown" and start["met_rival"] and start["party"]==0
    cells={sig:(env.save(),b"",tier)}
    starts=Counter()
    rng=random.Random(72503)
    events={}
    attempts=0
    path=b""
    def record(event,story):
        if event in events: return
        events[event]=dict(search_decisions=attempts,route_length=len(prefix)+len(path),story=story)
        (out/("frontier-"+event+".bin")).write_bytes(prefix+path)
        Image.fromarray(env.screen).save(out/("frontier-"+event+".png"))
        print(event,events[event],flush=True)
    while attempts<45000 and "party" not in events:
        best=max(c[2] for c in cells.values())
        keys=[k for k,c in cells.items() if c[2]==best]
        key=rng.choices(keys,weights=[1/(1+starts[k])**1.5 for k in keys])[0]
        state,path,_=cells[key]
        starts[key]+=1
        env.load(state)
        for macro in range(7):
            a=rng.choices([0,1,2,4,5,6,7],weights=[1,6,1,4,4,4,4])[0]
            for _ in range(rng.choice([1,2,3,4,5])):
                env.step(a)
                path+=bytes((a,))
                attempts+=1
                info=facts()
                if info:
                    sig,tier,story=info
                    if story["map"]=="Route101" and story["phase"] in (5,6):
                        record("route101",story)
                        if story["bag"]:record("bag",story)
                        if 1<=story["party"]<=6:record("party",story)
                    if (sig not in cells or len(cells[sig][1])>len(path)+8) and tier>=1:
                        cells[sig]=(env.save(),path,tier)
                if attempts>=45000 or "party" in events:break
            if attempts>=45000 or "party" in events:break
    proof_key=next((k for k in ("party","bag","route101") if k in events),None)
    proof=None
    if proof_key:
        tape=(out/("frontier-"+proof_key+".bin")).read_bytes()
        env.load(env.boot)
        route=False
        for button in tape:
            env.step(button)
            info=facts()
            if info and info[2]["map"]=="Route101" and info[2]["phase"] in (5,6):route=True
        proof=facts()[2]
        assert route and proof==events[proof_key]["story"]
        if proof_key=="party":assert 1<=proof["party"]<=6
    (out/"frontier.json").write_text(json.dumps(dict(seed=72503,budget=45000,decisions=attempts,
         start_buttons=len(prefix),start=start,cells=len(cells),events=events,independent_replay=proof,
         verified_from_boot=bool(proof_key)),indent=2)+"\n")
try:
    run()
finally:
    env.close()
