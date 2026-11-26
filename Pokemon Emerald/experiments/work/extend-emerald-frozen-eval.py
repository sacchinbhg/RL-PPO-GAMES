"""Evaluate a frozen policy and compare archived inputs independently of the game outcome."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

project = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL"
official = project / "runs/pixels-only-01/official-wsl"
extensions = {
    16384: ("eval-65536", "eval-65536-long"),
    65536: ("eval-65536-long", "eval-65536-game-65536"),
    131072: ("eval-65536-game-65536", "eval-65536-game-131072"),
    262144: ("eval-65536-game-65536", "eval-65536-game-262144"),
}
workers = {f"env-{i}" for i in range(4)}


def load(path):
    return json.loads(path.read_text())


def store(path, result):
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(path)


def active(directory):
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = path.read_bytes().decode(errors="replace").split("\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "--logdir" in args:
            i = args.index("--logdir")
            if args[i + 1:i + 2] == [str(directory)]:
                return True
    return False


def gather(directory, cutoff, intermediate=False):
    paths = {}
    def add(worker, audit):
        if worker in paths and audit != paths[worker]:
            raise ValueError(f"Multiple first episodes for {worker} at {cutoff}: {paths[worker]}, {audit}")
        paths[worker] = audit
    for audit in (directory / "external-audit").glob("*/*/*episode-00001.json"):
        data = load(audit)
        count = data["steps"]
        if count == cutoff or (count < cutoff and data["events"].get("stone_badge") == count):
            add(audit.parent.name, audit)
    if intermediate:
        for audit in (directory / "in-progress").glob(f"*/*/step-{cutoff:06}/*episode-00001.json"):
            add(audit.parent.parent.name, audit)
    return paths


def read_game(audit, cutoff):
    info = load(audit)
    tape = audit.with_suffix(".bin").read_bytes()
    if not tape or len(tape) != info["steps"] or max(tape) > 8:
        raise ValueError(f"Invalid action tape at {audit}")
    if len(tape) != cutoff and not (len(tape) < cutoff and info["events"].get("stone_badge") == len(tape)):
        raise ValueError(f"Unexpected game length at {audit}")
    if any(type(n) != int or not 0 < n <= len(tape) for n in info["events"].values()):
        raise ValueError(f"Out-of-range milestone at {audit}")
    return tape, info["events"]


def compare(audit_before, audit_now, lower, limit):
    prior, old_events = read_game(audit_before, lower)
    actual, events = read_game(audit_now, limit)
    overlap = min(len(prior), len(actual))
    difference = next((i for i, (a, b) in enumerate(zip(prior, actual), 1) if a != b), None)
    common = overlap if difference is None else difference - 1
    old = {k: v for k, v in old_events.items() if v <= overlap}
    new = {k: v for k, v in events.items() if v <= overlap}
    return dict(previous_decisions=len(prior), compared_decisions=overlap, identical_first_decisions=common,
                first_different_button=difference, all_previous_decisions_identical=(common == len(prior)),
                matching_milestones_through_overlap=(old == new), previous_events=old_events, events=events,
                events_after_previous_game={k:v for k,v in events.items() if v > len(prior)},
                total_decisions=len(actual), previous_tape_sha256=hashlib.sha256(prior).hexdigest(),
                tape_sha256=hashlib.sha256(actual).hexdigest(), previous_audit=str(audit_before),
                original_rom_audit=str(audit_now))


def publish(short, target, actual, limit, intermediate):
    provenance, current = (load(p / "provenance.json") for p in (short, actual))
    keys = ("agent_sha256", "training_step", "seed", "envs", "imagination_horizon", "upstream_revision",
            "adapter", "platform", "discount_horizon", "uncapped_exploration")
    if any(current[k] != provenance[k] for k in keys) or current["envs"] != len(workers):
        raise ValueError("The checkpoint, evaluation configuration or number of games changed")
    if current["truncation"] != (262144 if intermediate else limit):
        raise ValueError("Unexpected evaluation horizon")
    lower = provenance["truncation"]
    before = gather(short, lower)
    after = gather(actual, limit, intermediate)
    if set(before) != workers or set(after) != workers:
        return False
    rows = [dict(env=env, **compare(before[env], after[env], lower, limit)) for env in sorted(workers)]
    exact = all(r["all_previous_decisions_identical"] and r["matching_milestones_through_overlap"] for r in rows)
    result = dict(agent_sha256=current["agent_sha256"], training_step=current["training_step"],
                  evaluation_seed=current["seed"], buttons_per_game=limit, lower_cutoff=lower,
                  games=rows, all_prefixes_match=exact, source_official_evaluation=str(actual),
                  fresh_boot=True, emulator_restores_during_game=0,
                  comparison_kind="exact extensions of earlier games" if exact else
                  "new fresh games with matching configuration and partial input prefixes")
    store(target / "prefix-comparison.json", result)
    if exact:
        store(target / "prefix-verification.json", result)
    print(f"Compared {len(rows)} frozen games through {limit} buttons. All earlier prefixes matched: {exact}", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm")
    parser.add_argument("limit", nargs="?", default=16384, type=int, choices=extensions)
    parser.add_argument("--watch-existing", action="store_true", help="Only monitor and compare an existing evaluation")
    opts = parser.parse_args()
    limit = opts.limit
    arm = official / opts.arm
    source_name, target_name = extensions[limit]
    short, target = arm / source_name, arm / target_name
    target.mkdir(exist_ok=True)
    if any((target / name).exists() for name in ("prefix-comparison.json", "prefix-verification.json")):
        return
    p = load(short / "provenance.json")
    longer = arm / "eval-65536-game-262144"
    for actual in ([longer, target] if limit == 131072 else [target]):
        if not (actual / "provenance.json").exists():
            continue
        while True:
            if publish(short, target, actual, limit, actual != target):
                return
            if not active(actual):
                if publish(short, target, actual, limit, actual != target):
                    return
                break
            time.sleep(15)
    if opts.watch_existing:
        raise RuntimeError("The existing evaluation has stopped and its four first game audits are incomplete")
    command = [sys.executable, "-u", str(project / "upstream_emerald.py"), "--upstream", "/opt/emerald-rl/source",
               "--platform", "cuda", "--envs", "4", "--batch", "8", "--length", "32", "--ratio", "16",
               "--imag-length", str(p["imagination_horizon"]), "--seed", str(p["seed"]),
               "--discount-horizon", str(p.get("discount_horizon", 333)), "--evaluate", str(short / "frozen-policy"),
               "--episode", str(limit), "--steps", str(4*limit+8), "--logdir", str(target),
               "--reward-replay-fraction", str(p.get("reward_replay_fraction", 0.))]
    if (official / "controls/parallel-frozen-evaluation/verified-equivalence.json").exists():
        command.append("--parallel-envs")
    if p.get("uncapped_exploration"):
        command.append("--uncapped-exploration")
    if p.get("retain_reward_windows"):
        command.append("--retain-reward-windows")
    with (target / "evaluation.log").open("a") as log:
        subprocess.run(command, cwd=project, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not publish(short, target, target, limit, False):
        raise RuntimeError("The completed evaluation is missing one or more first-game audits")


if __name__ == "__main__":
    main()
