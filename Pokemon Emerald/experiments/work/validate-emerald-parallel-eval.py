"""Repeat a frozen serial evaluation using the official GPU parallel driver."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

project = Path(__file__).resolve().parents[1] / "outputs" / "EmeraldRL"
official = project / "runs/pixels-only-01/official-wsl"


def episodes(log):
    records = {}
    for path in sorted((log / "external-audit").glob("*/*/*.json")):
        key = (path.parent.name, path.name)
        data = json.loads(path.read_text())
        value = dict(events=data["events"], steps=data["steps"],
                     tape_sha256=hashlib.sha256(path.with_suffix(".bin").read_bytes()).hexdigest())
        if key in records and records[key] != value:
            raise ValueError(f"Conflicting episodes for {key}")
        records[key] = value
    return records


def median_fps(log):
    values = [json.loads(line).get("fps/policy") for line in (log / "metrics.jsonl").read_text().splitlines()]
    return statistics.median(x for x in values if x and x > 0)


def main():
    serial = official / "seed-71299-h15/eval-65536"
    parallel = official / "controls/parallel-frozen-evaluation"
    parallel.mkdir(exist_ok=True)
    marker = parallel / "verified-equivalence.json"
    if marker.exists():
        return
    command = [sys.executable, "-u", str(Path(__file__).with_name("eval-with-model-check.py")), "--upstream",
        "/opt/emerald-rl/source", "--platform", "cuda", "--parallel-envs", "--envs", "4",
        "--batch", "8", "--length", "32", "--ratio", "16", "--imag-length", "15",
        "--seed", "271340", "--steps", "32780", "--episode", "8192",
        "--evaluate", str(serial / "frozen-policy"), "--logdir", str(parallel)]
    with (parallel / "eval.log").open("a") as logfile:
        subprocess.run(command, cwd=project, stdout=logfile, stderr=subprocess.STDOUT, check=True)
    reference, actual = episodes(serial), episodes(parallel)
    assert reference == actual and len(actual) == 4
    provenance = json.loads((parallel / "provenance.json").read_text())
    assert provenance["agent_sha256"] == json.loads((serial / "provenance.json").read_text())["agent_sha256"]
    a, b = median_fps(serial), median_fps(parallel)
    result = dict(frozen_policy_sha256=provenance["agent_sha256"], checkpoint_step=provenance["training_step"],
                  games=list(actual.values()), serial_median_fps=a, parallel_median_fps=b,
                  evaluation_speed_ratio=b/a)
    marker.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
