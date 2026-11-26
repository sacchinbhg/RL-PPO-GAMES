"""Read-only original-ROM replay for geography and independently checked party gains."""
import hashlib
import json
from pathlib import Path
import struct
import sys

from PIL import Image
project = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL"
sys.path.insert(0, str(project))
from emerald import Emerald, MAPS, verify_rom
from pixel_env import Score, game_paths
from pixel_replay import GROWTH_BLOCK, STARTERS, verify_starter


def first_member(game):
    mon = game.ram(0x020244ec, 100)
    personality, trainer = struct.unpack_from("<II", mon)
    decrypted = struct.pack("<12I", *(w ^ personality ^ trainer for w in struct.unpack_from("<12I", mon, 32)))
    checksum, = struct.unpack_from("<H", mon, 28)
    if sum(struct.unpack("<24H", decrypted)) % 65536 != checksum:
        return None
    off = 12 * GROWTH_BLOCK[personality % 24]
    species, = struct.unpack_from("<H", decrypted, off)
    experience, = struct.unpack_from("<I", decrypted, off + 4)
    return personality, trainer, species, mon[84], experience, checksum


def run(source, out):
    original = json.loads(source.read_text())
    tape = source.with_suffix(".bin").read_bytes()
    assert len(tape) == original["steps"]
    out.mkdir(parents=True, exist_ok=True)
    result = out / "verified.json"
    sha = hashlib.sha256(tape).hexdigest()
    if result.exists():
        assert json.loads(result.read_text())["tape_sha256"] == sha
        return
    core, rom = game_paths()
    verify_rom(rom)
    game = Emerald(str(core),str(rom))
    score = Score(original.get("uncapped_exploration",False))
    maps, levels, party_max = {}, [], {}
    all_phases, maximum_by_party = [], {}
    total = 0
    starter = None
    snapshots = set()
    try:
        game.load(game.boot)
        for n, button in enumerate(tape, 1):
            _, key = game.step(button)
            amount, terminal = score.read(game, key)
            total += round(100 * amount)
            if n == original["events"].get("starter"):
                starter = verify_starter(game, n)
            verified = game.verified_story(key)
            if verified and verified[1]:
                member=first_member(game)
                if member:
                    pid,trainer,species,level,exp,checksum=member
                    old=maximum_by_party.get((pid,trainer))
                    if 1<=level<=100 and (old is None or level>old[0] and exp>old[1]):
                        maximum_by_party[pid,trainer]=(level,exp)
                        all_phases.append(dict(decision=n,species=species,species_name=STARTERS.get(species),level=level,
                                        experience=exp,checksum=checksum,phase="battle" if key[0]==6 else "overworld",
                                        map=MAPS[key[1]][key[2]]))
            if verified and key[0] == 5:
                group, number = key[1:3]
                name = MAPS[group][number]
                maps.setdefault(name, n)
                if verified[1]:
                    member = first_member(game)
                    if member:
                        pid, trainer, species, level, exp, checksum = member
                        prior = party_max.get((pid, trainer))
                        if 1 <= level <= 100 and (prior is None or (level > prior[0] and exp > prior[1])):
                            party_max[pid, trainer] = level, exp
                            levels.append(dict(decision=n, species=species, species_name=STARTERS.get(species),
                                level=level, experience=exp, checksum=checksum, map=name))
                            if prior and level not in snapshots:
                                Image.fromarray(game.screen).save(out/f"level-{level:02}.png")
                                snapshots.add(level)
        assert original["events"] == score.events
        assert original.get("return_cents",total) == total and original["first_visit_cells"] == len(score.cells)
        record = dict(source=str(source), tape_sha256=sha, buttons=len(tape), restores_during_replay=0,
                      source_type=original.get("source", "recorded gameplay"), events=score.events,
                      maps_first_reached=maps, first_party_level_history=levels,
                      first_party_level_history_all_phases=all_phases,starter=starter, return_cents=total)
        result.write_text(json.dumps(record, indent=2))
        print(json.dumps(dict(maps=maps, levels=levels)), flush=True)
    finally:
        game.close()


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
