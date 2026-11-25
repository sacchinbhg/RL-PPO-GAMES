"""Run the validated native pilot, its timed continuation, and final on-ROM transfer test."""
import argparse
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def phase(folder, name, **details):
    tmp = folder / "pipeline.tmp"
    tmp.write_text(json.dumps(dict(phase=name, updated=datetime.now(timezone.utc).isoformat(),
                                   **details), indent=2) + "\n", encoding="utf-8")
    tmp.replace(folder / "pipeline.json")


def run(log, command):
    with log.open("ab") as f:
        subprocess.run([sys.executable, "-u", *map(str, command)], cwd=ROOT,
                       stdout=f, stderr=subprocess.STDOUT, check=True)


def main(args):
    directory = args.out.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    cfg = load(ROOT / "local.json")
    benchmark = load(directory / "verification/benchmark.json")
    checksum = hashlib.sha256(Path(cfg["native_core"]).read_bytes()).hexdigest()
    if (benchmark.get("native_core_sha256") != checksum or not benchmark["scripted_reference"].get("matches_reference")
            or len(benchmark["samples"]) != 4 or any(not r["cross_worker_equal_after_eight_steps"] for r in benchmark["samples"])):
        raise ValueError("Finish the native engine validation before starting a long run")
    if args.mirror:
        shutil.copytree(directory / "verification", args.mirror / "verification", dirs_exist_ok=True)
    pilot = directory / "pilot"
    phase(directory, "Training native pilot", pilot_steps=16384)
    if not (pilot / "training.json").exists() or not load(pilot / "training.json").get("finished"):
        run(directory / "pilot.log", [
            "experiment.py", "train", "--backend", "native", "--core", cfg["native_core"], "--rom", cfg["rom"],
            "--mode", "world", "--device", "cuda", "--weights", args.weights.resolve(), "--steps", "16384",
            "--envs", "8", "--episode", "512", "--horizon", "64", "--seed", str(args.seed), "--out", pilot,
        ])
    info = load(pilot / "training.json")
    if not (info["finished"] and info["backend"] == "native" and info["dynamics_updates"] > 0
            and info["imagined_steps"] > 0 and info.get("archive_resets", 0) > 0):
        raise RuntimeError("Native pilot did not exercise world updates and archive restores")
    if args.mirror:
        (args.mirror / "pilot").mkdir(parents=True, exist_ok=True)
        shutil.copy2(pilot / "training.json", args.mirror / "pilot/training.json")
    long = directory / "long"
    phase(directory, "Training and evaluating native checkpoints", until=args.until.isoformat())
    if not (long / "status.json").exists():
        seconds = (args.until - datetime.now().astimezone()).total_seconds()
        if seconds <= 0:
            raise ValueError("The requested experiment deadline has passed")
        command = ["overnight.py", "--out", long, "--restart-from", pilot / "checkpoint.pt",
                   "--hours", str(seconds / 3600), "--seed", str(args.seed + 1)]
        if args.mirror:
            command += ["--mirror", args.mirror / "long"]
        run(directory / "supervisor.log", command)
    status = load(long / "status.json")
    if not status["phase"].startswith("Completed"):
        raise RuntimeError(f"Native continuation phase: {status['phase']}")
    final = Path(status["final_checkpoint"]).with_name("policy.pt")
    if not (directory / "transfer/evaluation.json").is_file():
        phase(directory, "Evaluating final native policy on the original ROM")
        run(directory / "transfer.log", ["experiment.py", "eval", "--backend", "mgba", "--core", cfg["core"],
                                        "--rom", cfg["rom"], "--mode", "world", "--device", "cuda",
                                        "--weights", final, "--envs", "8", "--episode", "32768",
                                        "--capture-moments", "--seed", "246821", "--out", directory / "transfer"])
    if args.mirror:
        shutil.copytree(directory / "transfer", args.mirror / "transfer", dirs_exist_ok=True)
    phase(directory, "Completed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--mirror", type=Path, help="Optional second copy of lightweight reports and gameplay moments")
    p.add_argument("--until", type=datetime.fromisoformat, required=True, help="ISO time with UTC offset")
    p.add_argument("--seed", type=int, default=7251124)
    args = p.parse_args()
    if args.until.tzinfo is None:
        p.error("--until requires a timezone offset")
    try:
        main(args)
    except BaseException as exc:
        if args.out.is_dir():
            phase(args.out.resolve(), "Interrupted", error=str(exc))
        raise
