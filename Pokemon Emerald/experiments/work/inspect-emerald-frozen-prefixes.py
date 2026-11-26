"""Compare recorded, real original-ROM inputs at matched cutoffs while a long frozen game runs."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL/runs/pixels-only-01/official-wsl/seed-71299-h15"
short, long = root/"eval-65536-game-65536", root/"eval-65536-game-262144"
p, q = (json.loads((d/"provenance.json").read_text()) for d in (short,long))
assert p["agent_sha256"] == q["agent_sha256"] and p["seed"] == q["seed"]
records = []
for old in sorted((short/"external-audit").glob("*/*/*episode-00001.bin")):
    worker = old.parent.name
    candidates = list((long/"in-progress").glob(f"*/{worker}/step-*/*.json"))
    candidates += list((long/"external-audit").glob(f"*/{worker}/*episode-00001.json"))
    if not candidates:
        continue
    new = max(candidates, key=lambda p: json.loads(p.read_text())["steps"])
    info = json.loads(new.read_text())
    earlier, later = old.read_bytes(), new.with_suffix(".bin").read_bytes()
    n = min(len(earlier), len(later))
    mismatch = next((i for i,(a,b) in enumerate(zip(earlier[:n],later[:n]),1) if a != b), None)
    old_events = json.loads(old.with_suffix(".json").read_text())["events"]
    align_events = {k:v for k,v in old_events.items() if v<=n} == {k:v for k,v in info["events"].items() if v<=n}
    records.append(dict(worker=worker, observed=len(later), checked=n, first_mismatch=mismatch,
                        matching_events_through_cutoff=align_events, events=info["events"]))
if records:
    result = dict(agent_sha256=p["agent_sha256"], seed=p["seed"], games=records)
    scratch = long / "checked-input-prefixes.tmp"
    scratch.write_text(json.dumps(result, indent=2))
    scratch.replace(long/"checked-input-prefixes.json")
    print(json.dumps(records), flush=True)
