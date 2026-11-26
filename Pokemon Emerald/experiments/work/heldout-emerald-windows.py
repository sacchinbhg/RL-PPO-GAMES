"""Build prediction-only windows from the frozen policy's own evaluation tapes."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

project = Path(__file__).resolve().parents[1] / "outputs" / "EmeraldRL"
sys.path.insert(0,str(project))
from emerald import Emerald
from pixel_env import Score, game_paths

official = project / "runs/pixels-only-01/official-wsl"
starts = (2048, 6144)
length = 33


def extract(path):
    tape = path.with_suffix(".bin").read_bytes()
    evidence = json.loads(path.read_text())
    assert len(tape) == evidence["steps"] and len(tape) > max(starts) + length
    g = Emerald(*game_paths())
    score = Score()
    samples = [{k:[] for k in ("image","act","rew","first","last")} for _ in starts]
    try:
        g.load(g.boot)
        for n, action in enumerate(tape):
            if n == 0:
                pic, rew, done = g.boot_image, 0., False
            else:
                pic, key = g.step(tape[n-1])
                rew, done = score.read(g,key)
            for start, sample in zip(starts,samples):
                if start <= n < start+length:
                    image = np.zeros((64,64,3),np.uint8)
                    image[12:52,2:62]=pic
                    for k,v in zip(sample,(image, action, rew, n==0, done)):
                        sample[k].append(v)
            if n == starts[-1]+length-1:
                break
        assert score.events == {k:v for k,v in evidence["events"].items() if v <= n}
    finally:
        g.close()
    return samples, dict(source=path.relative_to(official).as_posix(),
                         input_sha256=hashlib.sha256(tape).hexdigest(), starts=starts)


if __name__ == "__main__":
    files = sorted((official/"seed-71299-h15/eval-65536/external-audit").glob("*/*/*.json"),
                   key=lambda p:p.parent.name)
    assert len(files)==4
    with ProcessPoolExecutor(max_workers=2) as executor:
        results=list(executor.map(extract,files))
    windows=[sample for samples,_ in results for sample in samples]
    arrays={k:np.array([w[k] for w in windows]) for k in windows[0]}
    out=official/"controls/heldout-world-model"
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"evaluation-windows.npz", image=arrays["image"],
                        action=arrays["act"].astype(np.int32), reward=arrays["rew"].astype(np.float32),
                        is_first=arrays["first"], is_last=arrays["last"], is_terminal=arrays["last"])
    (out/"sources.json").write_text(json.dumps([info for _,info in results],indent=2))
    print("Verified 8 evaluation windows of 33 observations, four fresh games.", flush=True)
