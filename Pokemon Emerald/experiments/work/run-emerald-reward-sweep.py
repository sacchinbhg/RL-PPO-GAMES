"""Test systematic single-button coverage at rewarded training states with the same seed and budget."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from train_earned_dreamer import write_json
root=project/"runs/pixels-only-01/official-wsl"
out=root/"controls/reward-sweep"
comparison=root/"paired-self-earned-frontiers/systematic-reward-frontiers"
dest=root/"paired-self-earned-frontiers/sweep-reward-frontiers"
registration=root/"controls/reward-frontier-expansion"
motion=root/"controls/visual-motion/route-training.json"
read=lambda p:json.loads(p.read_text())


def call(script,*args):
    with (out/"run.log").open("a") as log:
        subprocess.run([sys.executable,"-u",str(project/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True)


def run():
    plan=read(registration/"plan.json")
    write_json(out/"registration.json",dict(source=plan["selected"]["tape_sha256"],seed=plan["seed"],
        new_emulated_buttons=plan["steps"],reward_slots=plan["capacity"],matched_control=str(comparison),
        intervention="store rewarded states and try all nine first buttons before longer options; 1/16 short learned continuations",
        signal="RGB, scalar reward and done; exclude external progress review",fresh_evaluation_data_used=False,
        outcomes="earned new scalar rewards, verified new story events and first new maps"))
    smoke=root/"controls/outcome-collection-smoke"
    small=out/"smoke"
    marker=out/"smoke-verified.json"
    if not marker.exists():
        call("explore_reward_frontiers.py","--source",smoke,"--record",smoke/"best.json","--motion",motion,"--output",small,
            "--steps",512,"--seed",plan["seed"],"--capacity",16,"--reward-sweep")
        result=read(small/"completed.json")
        best=result["best"]
        proof=out/"smoke-replay"
        call("pixel_replay.py",small/best["audit"]/"game.json",proof,"--decision",best["length"])
        replay=read(proof/"verification.json")
        assert result["status"]["new_buttons"]==512 and result["status"]["branches"]>1
        assert replay["input_sha256"]==best["tape_sha256"] and replay["restores_during_game"]==0
        assert result["status"]["admissions"].get("appearance",0)==0 and read(small/"settings.json")["reward_sweep"]
        write_json(marker,dict(buttons=512,branches=result["status"]["branches"],reproduced_training_decisions=replay["buttons"],
                               original_rom_sha1=replay["rom_sha1"],input_sha256=replay["input_sha256"]))
    while int(next(s.split()[1] for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:")))<1550000:
        time.sleep(20)
    call("explore_reward_frontiers.py","--source",plan["source"],"--record",registration/"selected-record.json",
        "--motion",motion,"--output",dest,"--steps",plan["steps"],"--capacity",plan["capacity"],"--seed",plan["seed"],"--reward-sweep")
    write_json(out/"result.json",read(dest/"completed.json"))


if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so",
                      OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    out.mkdir(exist_ok=True)
    with (out/"job.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
