"""Confirm the fast frozen player records and stops before a deliberately discordant reference action."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

project = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL"
root = project / "runs/pixels-only-01/official-wsl"
source = root / "seed-71299-h15/eval-65536"
test = root / "controls/direct-reference-stop"
test.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=test) as scratch:
    directory = Path(scratch)
    reference = directory / "reference"
    reference.mkdir()
    shutil.copyfile(source/"provenance.json", reference/"provenance.json")
    for p in (source/"external-audit").glob("*/*/*episode-00001.bin"):
        worker = p.parent.name
        tape = bytearray(p.read_bytes()[:8])
        if worker == "env-2":
            tape[2] = (tape[2] + 1) % 9
        target = reference / "external-audit/synthetic" / worker / p.name
        target.parent.mkdir(parents=True)
        target.write_bytes(tape)
    out = directory/"play"
    result = subprocess.run([sys.executable, str(project/"play_frozen_dreamer.py"), str(source/"frozen-policy"),
        str(source), str(out), "--limit","8","--reference",str(reference),"--upstream","/opt/emerald-rl/source"],
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    if not (out/"prefix-mismatch.json").exists():
        raise RuntimeError(result.stdout[-3000:])
    failure = json.loads((out/"prefix-mismatch.json").read_text())
    assert result.returncode and failure["decision"] == 3 and failure["workers"] == [2]
    tape, = (out/"mismatch/env-2").glob("*.bin")
    assert len(tape.read_bytes()) == 2 and not (out/"result.json").exists()
    (test/"test-result.json").write_text(json.dumps({"synthetic_reference_change":True, **failure},indent=2))
