"""Measure independent original-ROM outcomes for a self-generated native action history."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

base = Path(__file__).resolve().parents[1]
project = base / "outputs/EmeraldRL"
sys.path.insert(0, str(project))
from emerald import Emerald, REPEAT, ROM_SHA1, verify_rom
from pixel_env import Score, game_paths
from pixel_replay import verify_starter
from train_earned_dreamer import write_json

root = project / "runs/pixels-only-01/official-wsl"
source = root / "native-screen-exploration/seed-173744/first-maps/Route103"
out = root / "controls/native-to-original-route"


def run():
    record = json.loads((source / "game.json").read_text())
    tape = (source / "game.bin").read_bytes()
    sha = hashlib.sha256(tape).hexdigest()
    assert record["backend"] == "native-port" and record["training_only"] and record["from_blank"]
    assert record["steps"] == len(tape) and record["tape_sha256"] == sha
    core, rom = game_paths()
    verify_rom(rom)
    meta = dict(tape_sha256=sha, buttons=len(tape), native_source=str(source),
                rom_sha1=ROM_SHA1, emulator_sha256=hashlib.sha256(core.read_bytes()).hexdigest(),
                frames_per_decision=REPEAT, fresh_boot=True, in_game_restores=0,
                native_events=record["events"], native_maps=record["maps_first_reached"])
    write_json(out / "registration.json", meta)
    score = Score(True)
    game = Emerald(str(core), str(rom))
    start = time.monotonic()
    starter = None
    try:
        game.load(game.boot)
        for decision, button in enumerate(tape, 1):
            _, state = game.step(button)
            _, terminal = score.read(game, state)
            if score.events.get("starter") == decision:
                starter = verify_starter(game, decision)
            if decision % 4096 == 0:
                write_json(out / "status.json", dict(real_buttons=decision, expected=len(tape),
                           original_events=score.events, original_maps=score.first_maps))
            if terminal:
                break
        write_json(out / "result.json", dict(**meta, real_buttons=decision, original_events=score.events,
                   original_maps=score.first_maps, verified_starter=starter, seconds=time.monotonic()-start))
    finally:
        game.close()


if __name__ == "__main__":
    os.environ.update(EMERALD_ROM=str(base / "work/pokeemerald/pokeemerald.gba"),
                      EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (out / "result.json").exists():
            run()
