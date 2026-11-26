"""Verify search accounting on an actual self-earned game with a later work high-water mark."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from reward_branch_search import train,write_json

if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),
                      EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
    root=project/"runs/pixels-only-01/official-wsl"
    prior=root/"controls/outcome-collection-smoke"
    out=root/"controls/morning-host-recovery"
    read=lambda p:json.loads(p.read_text())
    best=read(prior/"best.json")
    with tempfile.TemporaryDirectory() as tmp:
        run=Path(tmp)
        shutil.copytree(prior/best["audit"],run/best["audit"])
        write_json(run/"best.json",best)
        previous=best["work"]+2000
        branches=best["branches"]+25
        peak=best["return_cents"]+20
        write_json(run/"status.json",dict(work=previous,branches=branches,best_return_cents=peak))
        train(run,69731,previous+32)
        result=read(run/"status.json")
        assert result["work"]==previous+32 and result["branches"]==branches+1
        assert result["historical_peak_cents"]>=peak and read(run/"best.json")["tape_sha256"]==best["tape_sha256"]
        write_json(out/"verified-search-accounting.json",dict(actual_cold_reconstruction=len((run/best["audit"]/"game.bin").read_bytes()),
                   fixture_recorded_work=previous,fixture_recorded_branches=branches,actual_new_decisions=32,
                   reported_work=result["work"],reported_branches=result["branches"],prior_peak_preserved=True))
