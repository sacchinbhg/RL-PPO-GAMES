"""Replay the same training prefix before and after a fork and compare every delivered observation."""
import hashlib
import json
import os
from pathlib import Path
import sys

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from reward_branch_env import ForkEnv,VisitReward
from reward_branch_search import write_json

def run():
    root=project/"runs/pixels-only-01/official-wsl"
    out=root/"controls/shared-visit-reward"
    out.mkdir(parents=True,exist_ok=True)
    source=root/"paired-self-earned-frontiers/visual-slots-0/discoveries/work-0000139234/game.bin"
    tape=source.read_bytes()[:1536]
    reward=VisitReward()
    assert [reward.adjust(x,(1,2,3,4),fresh) for x,fresh in [(102,True),(0,False),(102,True)]]==[102,0,100]
    reports={}
    traces={}
    for shared in (False,True):
        env=ForkEnv(shared_visit_reward=shared)
        try:
            env.command("save",0)
            for attempt in (1,2):
                env.command("restore",0)
                name=f"{'shared' if shared else 'ordinary'}-{attempt}"
                digest=hashlib.sha256()
                rewards=[]
                for button in tape:
                    image,cents,done,n=env.command("step",(button,1))
                    assert not done and n==1
                    digest.update(image)
                    rewards.append(cents)
                env.command("audit",str(out/name))
                audit=json.loads((out/name/"game.json").read_text())
                assert (out/name/"game.bin").read_bytes()==tape
                reports[name]=dict(pixels_sha256=digest.hexdigest(),delivered_cents=sum(rewards),
                                   original_cents=audit["return_cents"],cells=audit["first_visit_cells"],events=audit["events"])
                traces[name]=rewards
        finally:
            env.close()
    baseline=reports["ordinary-1"]
    assert baseline["cells"]>0 and set(r["pixels_sha256"] for r in reports.values())=={baseline["pixels_sha256"]}
    assert all(r["original_cents"]==baseline["original_cents"] and r["events"]==baseline["events"] for r in reports.values())
    assert traces["ordinary-1"]==traces["ordinary-2"]==traces["shared-1"]
    removed=[old-new for old,new in zip(traces["shared-1"],traces["shared-2"])]
    assert set(removed)<={0,2} and sum(removed)==2*baseline["cells"]
    write_json(out/"verified.json",dict(buttons_per_pass=len(tape),passes=reports,
               duplicated_location_payments_removed=sum(n>0 for n in removed),all_policy_frames_identical=True))

if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),
        EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
    run()
