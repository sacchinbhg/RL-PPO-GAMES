"""Compare two fresh games under identical buttons while one writes partial audits."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import numpy as np

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from pixel_env import PixelPool
root=project/"runs/pixels-only-01/official-wsl"


if __name__=="__main__":
    source,=(root/"seed-71299-h15-reward-replay/eval-65536/external-audit").glob("*/env-0/*.bin")
    actions=source.read_bytes()[:2600]
    expected=json.loads(source.with_suffix(".json").read_text())["events"]
    pools=PixelPool(2)
    hashes=[hashlib.sha256(),hashlib.sha256()]
    with tempfile.TemporaryDirectory(dir=project.parents[1]/"work") as scratch:
        folder=Path(scratch)
        try:
            resets=[pools.command(i,"reset") for i in range(2)]
            assert np.array_equal(*resets)
            for n,a in enumerate(actions,1):
                pics,rews,ends=pools.step([a,a])
                assert np.array_equal(*pics) and rews[0]==rews[1] and ends[0]==ends[1]
                for i in range(2):
                    hashes[i].update(pics[i].tobytes()+rews[i].tobytes()+ends[i].tobytes())
                if n in (100,1592,2473):
                    dest=folder/str(n)
                    pools.command(0,"audit",str(dest))
                    tape,=dest.glob("*.bin")
                    assert tape.read_bytes()==actions[:n]
            for i in range(2):
                dest=folder/f"final-{i}"
                pools.command(i,"audit",str(dest))
                tape,=dest.glob("*.bin")
                assert tape.read_bytes()==actions
                events=json.loads(tape.with_suffix(".json").read_text())["events"]
                assert events=={k:v for k,v in expected.items() if v<=len(actions)}
        finally:
            pools.close()
    result=dict(decisions=len(actions),extra_audits=[100,1592,2473],events=events,
                rgb_reward_terminal_sha256=[h.hexdigest() for h in hashes])
    (root/"controls/passive-audit-equivalence.json").write_text(json.dumps(result,indent=2))
    print(result,flush=True)
