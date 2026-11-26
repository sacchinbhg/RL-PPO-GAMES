import hashlib
import json
from pathlib import Path
import sys
import tempfile

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from option_explorer import initialize

source=project/"runs/pixels-only-01/official-wsl/paired-self-earned-frontiers/visual-slots-0"
with tempfile.TemporaryDirectory(dir=base/"work") as tmp:
    target=Path(tmp)/"experiment"
    initialize(source,target,42)
    raw=(target/"branch-origin.json").read_bytes()
    record=json.loads((target/"best.json").read_text())
    assert hashlib.sha256((target/record["audit"]/"game.bin").read_bytes()).hexdigest()==record["tape_sha256"]
    origin=json.loads(raw)
    assert origin["inherited_training_buttons"]==json.loads((source/"status.json").read_text())["work"]
    initialize(source,target,42)
    assert (target/"branch-origin.json").read_bytes()==raw
    try:
        initialize(source,target,43)
    except ValueError:
        pass
    else:
        raise AssertionError("Accepted an incompatible restart")
    print(json.dumps(dict(verified=record["tape_sha256"],work=origin["inherited_training_buttons"],seed=42)))
