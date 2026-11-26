"""Restore the complete selected official replay and compare its arrays with the saved originals."""
from pathlib import Path
import json
import resource
import subprocess
import sys
import tempfile
import numpy as np

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path[:0]=[str(project),"/opt/emerald-rl/source"]
import elements
from embodied.core.replay import Replay
from embodied.core.chunk import Chunk
from file_backed_replay import install
from train_earned_dreamer import write_json
root=project/"runs/pixels-only-01/official-wsl"
out=root/"controls/file-backed-replay"
cache=Path("/opt/emerald-rl/replay-cache")


def run():
    cache.mkdir(exist_ok=True)
    assert subprocess.check_output(["findmnt","-T",str(cache),"-n","-o","FSTYPE"],text=True).strip()=="ext4"
    out.mkdir(exist_ok=True)
    replay_dir=root/"seed-71299-h15-staggered-screen-collection/train/replay"
    length,capacity=33,131072
    replay=Replay(length,capacity,str(replay_dir))
    disk=sorted((p.name for p in replay_dir.glob("*.npz")),reverse=True)
    nitems=replay._numitems(disk)
    selected=[]
    s=0
    for name in disk:
        selected.append(name)
        s+=nitems[elements.UUID(name.split("-")[1])]
        if s>=capacity:
            break
    install(cache,out/"loaded.json")
    replay.load()
    rows={c.filename:c for c in replay.chunks.values()}
    assert set(rows)<=set(selected) and len(replay)==capacity
    # Upstream may evict the earliest items on insertion to enforce capacity.
    assert set(json.loads((out/"loaded.json").read_text())["filenames"])==set(rows)
    compared=mapped=0
    for i,name in enumerate(sorted(rows)):
        chunk=rows[name]
        with np.load(replay_dir/name) as original:
            assert set(original.files)==set(chunk.data)
            for k in original.files:
                a=original[k]
                b=chunk.data[k]
                assert a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b)
                compared+=a.nbytes
                if isinstance(b,np.memmap):
                    mapped+=b.nbytes
        if i%8==0:
            write_json(out/"status.json",dict(arrays_verified_through_chunk=i+1,total_chunks=len(rows),compared_bytes=compared))
    with tempfile.TemporaryDirectory(dir=cache) as tmp:
        field=np.arange(16384,dtype=np.float32).reshape(128,128)
        child=Chunk(2)
        child.append(dict(image=field,action=np.asarray(3,np.int32)))
        child.append(dict(image=field+2,action=np.asarray(4,np.int32)))
        assert isinstance(child.data["image"],np.memmap)
        child.update(0,1,dict(image=field[None]+4))
        child.save(elements.Path(tmp))
        restored=Chunk.load(elements.Path(tmp)/child.filename)
        assert np.array_equal(restored.slice(0,2)["image"],np.stack((field+4,field+2)))
        damaged=Path(tmp)/child.filename
        damaged.write_bytes(b"invalid test fixture")
        try:
            Chunk.load(elements.Path(damaged),error="none")
        except (ValueError,OSError):
            pass
        else:
            raise ValueError("Damaged replay was silently accepted")
    write_json(out/"verified.json",dict(selected_historical_files=len(selected),retained_historical_files=len(rows),
        retained_sequences=len(replay),compared_bytes=compared,file_backed_bytes=mapped,all_retained_arrays_identical=True,
        append_update_save_reload=True,unreadable_chunk_stops_restore=True,peak_process_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=="__main__":
    run()
