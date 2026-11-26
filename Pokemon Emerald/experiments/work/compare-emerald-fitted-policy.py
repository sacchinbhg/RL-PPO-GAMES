"""Compare fresh learned actors at the same button limit and the same four-game evaluation seed."""
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL/runs/pixels-only-01/official-wsl"
actors = root / "reward-only-training-forks/actor-fits"
source = root / "seed-71299-h15/eval-65536-game-65536"


def run(tape, limit):
    arm = actors / tape
    fit = json.loads((arm / "result.json").read_text())
    evaluation = arm / f"fresh-{limit}"
    fresh = json.loads((evaluation / "result.json").read_text())
    parent = json.loads((source / "provenance.json").read_text())
    assert fit["parent_sha256"] == parent["agent_sha256"]
    assert fit["fitted_sha256"] == fresh["weight_sha256"] and fit["training_tape_sha256"] == tape
    assert fresh["seed"] == parent["seed"] and fresh["in_game_restores"] == 0
    assert fresh["buttons_per_game"] == limit <= parent["truncation"]
    calibration = root / "seed-71299-h15/frozen-play-262144/prefix-mismatch.json"
    matched_prefix = json.loads(calibration.read_text())["matched_all_workers_through"] if calibration.exists() else 0
    if limit <= matched_prefix:
        control_method = "official reference within independently matched inference prefix"
        logs = {a.parent.name:a for a in (source/"external-audit").glob("*/*/*episode-00001.json")}
    else:
        control_method = "same inference runner, unchanged parent"
        control = root / f"controls/direct-parent-{limit}"
        identity = json.loads((control/"result.json").read_text())
        assert identity["weight_sha256"] == fit["parent_sha256"] and identity["seed"] == fresh["seed"]
        assert identity["decisions_per_game"] >= fresh["decisions_per_game"]
        logs = {a.parent.parent.name:a for a in (control/"games").glob(f"*/step-{identity['decisions_per_game']:07d}/*.json")}
    assert len(logs) == 4
    budget = fresh["decisions_per_game"]
    rows = []
    for worker in sorted(logs):
        stock = json.loads(logs[worker].read_text())["events"]
        audit, = (evaluation/"games"/worker/f"step-{budget:07d}").glob("*.json")
        candidate = json.loads(audit.read_text())["events"]
        assert len(audit.with_suffix(".bin").read_bytes()) == budget
        rows.append(dict(worker=worker, parent={k:v for k,v in stock.items() if v<=budget}, fitted=candidate))
    names = sorted({e for row in rows for which in ("parent","fitted") for e in row[which]})
    counts = {which:{name:sum(name in row[which] for row in rows) for name in names} for which in ("parent","fitted")}
    comparison = dict(training_tape_sha256=tape, fitting_steps=fit["selected_updates"], parent_sha256=fit["parent_sha256"],
                      control_method=control_method, matched_seed=fresh["seed"], buttons_per_game=budget,
                      fresh_games=4, counts=counts, games=rows)
    (evaluation/"compared-with-parent.json").write_text(json.dumps(comparison, indent=2))
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]))
