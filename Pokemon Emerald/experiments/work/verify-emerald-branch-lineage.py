"""Seal the generating training history for the two live searches using the identical saved states."""
import json
import os
from pathlib import Path
import sys

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from earned_starts import EarnedStarts,verified_reset_lineage
from train_earned_dreamer import write_json
root=project/"runs/pixels-only-01/official-wsl"
read=lambda p:json.loads(p.read_text())


if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
    bank=EarnedStarts(root/"controls/dreamer-earned-resets/starts")
    accepted=[]
    for name in ("ordinary","learned"):
        arm=root/"paired-self-earned-frontiers"/("bank-movement-"+name)
        cfg=read(arm/"settings.json")
        assert cfg["bank_snapshot_sha256"]==[s["sha256"] for s in bank.records]
        dest=arm/"earned-start-provenance.json"
        if dest.exists():
            assert read(dest)==bank.info
        else:
            write_json(dest,bank.info)
        best=read(arm/"best.json")
        tape=(arm/best["audit"]/"game.bin").read_bytes()
        assert verified_reset_lineage(arm,cfg,tape)
        assert not verified_reset_lineage(arm,cfg,tape[::-1])
        accepted.append(dict(arm=name,tape_sha256=best["tape_sha256"],decisions=len(tape),reversed_history_rejected=True))
    write_json(root/"controls/local-screen-motion/identical-starts/verified-lineage.json",dict(arms=accepted,
               source=bank.info["source_sha256"],cold_export_required_for_future_policy_training=True))
