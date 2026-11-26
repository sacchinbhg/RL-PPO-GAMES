from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path

work = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("audit", work/"audit-frozen-exploration.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
root = audit.official
dest = root / "seed-71299-h15-screen-collection/eval-524288/diagnostics-32768"


def investigate(worker):
    out = dest / f"env-{worker}.json"
    if out.exists():
        return json.loads(out.read_text())
    def files(arm):
        directory = root / arm
        partial = list((directory/"in-progress").glob(f"*/env-{worker}/step-032768/*.json"))
        return partial[0] if partial else next((directory/"external-audit").glob(f"*/env-{worker}/*episode-00001.json"))
    candidate = files("seed-71299-h15-screen-collection/eval-524288")
    parent = files("seed-71299-h15/eval-262144")
    data = audit.audit(candidate)
    for name, p in (("new",candidate), ("parent",parent)):
        tape = p.with_suffix(".bin").read_bytes()
        assert len(tape) == 32768
        data[f"{name}_button_counts"] = [tape.count(i) for i in range(9)]
        data[f"{name}_story"] = json.loads(p.read_text())["events"]
    data["worker"] = worker
    out.write_text(json.dumps(data, indent=2))
    return data


if __name__ == "__main__":
    dest.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(investigate,[1,0,2,3]))
    (dest/"complete.json").write_text(json.dumps(results, indent=2))
