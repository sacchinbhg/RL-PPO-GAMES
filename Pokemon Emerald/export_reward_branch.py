"""Reproduce an earned training branch from boot and export aligned visual transitions."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from pixel_env import PixelPool


def export(run, out, record_path=None):
    settings = json.loads((run / "settings.json").read_text())
    record_path = record_path or run / "best.json"
    if not record_path.resolve().is_relative_to(run.resolve()):
        raise ValueError("Discovery record outside the training run")
    record = json.loads(record_path.read_text())
    if not (settings["training_forks"] and record["training_only"] and not settings["receives_saves_from_other_runs"]):
        raise ValueError("Expected a self-generated training branch")
    source = (run / record["audit"]).resolve()
    if not source.is_relative_to((run / "discoveries").resolve()):
        raise ValueError("Training tape outside the branch archive")
    tape = (source / "game.bin").read_bytes()
    if hashlib.sha256(tape).hexdigest() != record["tape_sha256"] or len(tape) != record["length"]:
        raise ValueError("Training tape checksum or length mismatch")
    audit = json.loads((source / "game.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    dest = out / record["tape_sha256"]
    if dest.exists():
        if "whiteouts" in audit and json.loads((dest/"manifest.json").read_text()).get("verified_whiteouts")!=audit["whiteouts"]:
            raise ValueError("The existing export has different battle-outcome evidence")
        return dest
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        draft = Path(tmp) / "verified"
        draft.mkdir()
        env = PixelPool(1, uncapped_exploration=True,log_whiteouts="whiteouts" in audit)
        total = 0
        try:
            picture = env.command(0, "reset")
            for offset in range(0, len(tape), 256):
                buttons = np.frombuffer(tape[offset:offset + 256], dtype=np.uint8)
                frames = [picture]
                rewards, ends = [], []
                for button in buttons:
                    picture, reward, end = env.command(0, "act", int(button))
                    frames.append(picture)
                    rewards.append(reward)
                    ends.append(end)
                    total += round(reward * 100)
                np.savez_compressed(draft / f"chunk-{offset:09d}.npz",
                    image=np.stack(frames), action=buttons, reward=np.array(rewards, np.float32),
                    terminal=np.array(ends, bool))
            env.command(0, "audit", str(draft / "audit"))
        finally:
            env.close()
        replay, = (draft / "audit").glob("*.json")
        checked = json.loads(replay.read_text())
        if (checked["events"] != audit["events"] or checked["first_visit_cells"] != audit["first_visit_cells"]
                or total != record["return_cents"] or replay.with_suffix(".bin").read_bytes() != tape):
            raise ValueError("Cold-boot playback disagreed with the training branch")
        if "whiteouts" in audit and checked["whiteouts"]!=audit["whiteouts"]:
            raise ValueError("Cold-boot playback disagreed with the training battle outcomes")
        manifest = dict(source_sha256=record["tape_sha256"], source_seed=settings["seed"],
                        search_buttons_at_discovery=record["work"], decisions=len(tape),
                        return_cents=total, source_training_forks=record["branches"],
                        replay_restores_during_game=0, frozen_evaluation_data_used=False,
                        alignment="Action i takes image i to image i+1 and then earns reward i.",
                        chunk_length=256)
        if "whiteouts" in audit:
            manifest["verified_whiteouts"]=checked["whiteouts"]
        (draft / "manifest.json").write_text(json.dumps(manifest, indent=2))
        draft.rename(dest)
    return dest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("training_run", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--record", type=Path, help="Export a particular preserved discovery from this training run")
    args = p.parse_args()
    print(export(args.training_run.resolve(), args.output.resolve(), args.record))
