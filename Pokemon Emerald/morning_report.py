"""Publish an atomic, read-only morning report from the two local supervisors."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time


def read(path):
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8-sig"))
    return {}


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def clock(iso):
    return datetime.fromisoformat(iso).astimezone().strftime("%b %d %I:%M %p %Z")


def summary(root, native, verify, pilot, transfer):
    lines = ["# Emerald overnight report", "", "Updated " + clock(datetime.now(timezone.utc).isoformat()) + ".", ""]
    statuses = []
    for name, folder in (("Original mGBA run", root), ("Native Emerald run", native)):
        s = read(folder / "status.json")
        statuses.append(s)
        lines += [f"## {name}", ""]
        if not s:
            p = read(pilot / "training.json") if name.startswith("Native") else {}
            if p:
                lines += [f"The pilot reached {p['total_training_steps']:,} real decisions and {p.get('dynamics_updates',0):,} dynamics updates. The long run has not published its first status yet.", ""]
            else:
                lines += ["Waiting for the first training report.", ""]
            continue
        row = s.get("latest", {})
        total = row.get("total_training_steps", s["starting_steps"])
        lines += [f"**{s['phase']}** as of {clock(s['updated_utc'])}. "
                  f"{total:,} total real decisions, including {total - s['starting_steps']:,} in this continuation. "
                  f"The latest logged rate is {row.get('active_steps_per_second', 0):.1f} real decisions/s. "
                  f"The world model records {row.get('imagined_steps', 0):,} imagined transitions and "
                  f"{row.get('dynamics_updates', 0):,} dynamics updates, including inherited work.", ""]
        if s.get("error"):
            lines += [f"Supervisor error: {s['error']}", ""]
        lines += ["Frozen policies played from a fresh boot. Counts are independent games that reached each event at least once.", "",
                  "| Checkpoint | Decisions | Games | Clock | Met rival | Route 101 | Actual party |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for entry in s.get("evaluations", []):
            report = read(folder / entry["report"])
            games = report.get("trials", [])
            if not games:
                continue
            counts = ["—"] * 4
            if all("verified_event_decision" in game for game in games):
                counts = [str(sum(metric in game["verified_event_decision"] for game in games))
                          for metric in ("clock", "met_rival", "route101", "party")]
            else:
                counts[0] = str(sum(game["clock"] for game in games))
                audited = read((folder / entry["report"]).parent / "rival-audit/report.json")
                if audited and all(t["first_map_decisions_match_original"] for t in audited["trials"]):
                    counts[1:] = [str(sum(metric in t["first_event_decisions"] for t in audited["trials"]))
                                  for metric in ("met_rival", "route101", "party")]
            lines.append(f"| {entry['name']} | {entry['steps']:,} | {len(games)} | " + " | ".join(counts) + " |")
        lines += ["", "A dash means that checkpoint used an older evaluator and the event lacks a verified count.", ""]
    lines += ["## Backend and transfer checks", ""]
    b = read(verify / "benchmark.json")
    if b:
        for label in ("native", "mgba"):
            samples = [s for s in b["samples"] if s["backend"] == label]
            if samples:
                lines.append(f"- {label}: {statistics.median(s['decisions_per_second'] for s in samples):,.1f} decisions/s "
                             f"across {len(samples)} timed sample(s), eight workers, with RGB and state observations. "
                             f"Median route restore {statistics.median(s['restore_seconds'] for s in samples):.4f} s.")
        lines += ["", "The upstream scripted starter sequence checks compatibility. Its inputs are excluded from policy training. "
                  "Both backends share the laptop with the original training run during these measurements.", ""]
    else:
        lines += ["Backend verification results are pending.", ""]
    result = read(transfer / "evaluation.json")
    if result:
        games = result["trials"]
        counts = [sum(k in g["verified_event_decision"] for g in games)
                  for k in ("clock", "met_rival", "route101", "party")]
        lines += [f"The frozen native-trained policy tested in the original ROM reached the clock in {counts[0]}/{len(games)}, "
                  f"the rival in {counts[1]}/{len(games)}, Route 101 in {counts[2]}/{len(games)}, "
                  f"and an occupied party in {counts[3]}/{len(games)} fresh games.", ""]
    lines += ["## What to do next", "",
              "Compare fresh-game milestones at matched real-decision budgets. Use the faster verified backend for algorithm tests, "
              "and evaluate its best frozen policy from boot on the original mGBA ROM. If reaching the rival stays rare across seeds, "
              "test one general doorway or NPC interaction bonus against the same archive/world learner.", "",
              "The policy can discover routes during archive-assisted training before it can repeat them independently. "
              "Use confirmed fresh-game dialogue, Route 101, and occupied-party events for video milestones. "
              "The original raw room log includes transient map IDs during loading.", ""]
    return "\n".join(lines), statuses


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "native", "verification", "pilot", "transfer", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--mirror", type=Path)
    p.add_argument("--follow", action="store_true")
    args = p.parse_args()
    while True:
        content, states = summary(args.original, args.native, args.verification, args.pilot, args.transfer)
        write(args.out, content)
        if args.mirror:
            try:
                write(args.mirror, content)
            except OSError as exc:
                print(f"Desktop mirror: {exc}", flush=True)
        finished = all(s.get("phase", "").startswith(("Completed", "Interrupted")) for s in states)
        if not args.follow or (finished and (args.transfer / "evaluation.json").is_file()):
            break
        time.sleep(60)
