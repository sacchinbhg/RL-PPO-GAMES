"""Measure native-port restore timing and verify all subsequent original-resolution frames on real action tapes."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import numpy as np

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from native_emerald import NativeEmerald
from train_earned_dreamer import write_json
root=project/"runs/pixels-only-01/official-wsl"
out=root/"controls/native-restore-rendering"
read=lambda path:json.loads(path.read_text(encoding="utf-8-sig"))


def test(game,tape,cut,alternate):
    prefix=game.prefix+struct.pack("<I",cut)+tape[:cut]
    suffix=tape[cut:cut+256]+np.random.default_rng(19401+cut).integers(0,9,128,dtype=np.uint8).tobytes()
    rows={}
    for render in ((False,True) if alternate else (True,False)):
        start=time.process_time()
        game.load(prefix,render_intermediate=render)
        seconds=time.process_time()-start
        stream=hashlib.sha256()
        for i in range(len(suffix)+1):
            if i:
                game.step(suffix[i-1])
            stream.update(game.screen.tobytes())
            stream.update(np.asarray(game.inspect(),dtype="<i4").tobytes())
        rows["stock" if render else "final_frame_only"]=dict(cpu_seconds=seconds,full_frames=len(suffix)+1,sha256=stream.hexdigest())
    rows["equal"]=rows["stock"]["sha256"]==rows["final_frame_only"]["sha256"]
    return rows


if __name__=="__main__":
    out.mkdir(exist_ok=True)
    settings=read(project/"local.json")
    game=NativeEmerald(settings["native_core"],settings["rom"])
    pilot=root/"controls/native-visual-transfer/raw/audit"
    second=root/"controls/native-visual-transfer/quantized/audit"
    rows=[]
    try:
        for directory,cuts in ((pilot,[4096,16384,44393]),(second,[8192,32768])):
            tape=(directory/"game.bin").read_bytes()
            assert hashlib.sha256(tape).hexdigest()==read(directory/"game.json")["tape_sha256"]
            for cut in cuts:
                rows.append(dict(cut=cut,tape=hashlib.sha256(tape).hexdigest(),**test(game,tape,cut,len(rows)%2)))
                write_json(out/"status.json",dict(cases_completed=len(rows),cases=rows))
    finally:
        game.close()
    ratio=sum(r["stock"]["cpu_seconds"] for r in rows)/sum(r["final_frame_only"]["cpu_seconds"] for r in rows)
    write_json(out/"result.json",dict(cases=rows,native_engine_sha256=hashlib.sha256(Path(settings["native_core"]).read_bytes()).hexdigest(),
        all_rgb_frames_and_observed_states_match=all(r["equal"] for r in rows),cpu_speedup=ratio,
        hardware_frames_per_button=24,eligible_for_native_training=all(r["equal"] for r in rows) and ratio>=1.10))
