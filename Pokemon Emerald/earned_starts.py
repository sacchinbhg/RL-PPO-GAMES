"""Prepare and verify self-earned original-game starts for Dreamer training workers."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from emerald import Emerald, REPEAT, ROM_SHA1
from pixel_env import Score, game_paths
from train_earned_dreamer import write_json


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def score_record(score):
    return dict(paid=sorted(score.paid), cells=sorted(score.cells), high_level=score.high_level,
                events=dict(score.events), first_maps=dict(score.first_maps), decisions=score.decisions)


def restore_score(record):
    score = Score(True)
    score.paid = set(record["paid"])
    score.cells = set(map(tuple, record["cells"]))
    score.high_level = record["high_level"]
    score.events = dict(record["events"])
    score.first_maps = dict(record["first_maps"])
    score.decisions = record["decisions"]
    return score


def observation_hash(frame, state, reward, terminal):
    return [checksum(frame.tobytes()), list(state), round(100*reward), bool(terminal)]


def verified_reset_lineage(run,settings,tape=None):
    """Check that imported training states came from the recorded parent's own original-game play."""
    if not settings.get("receives_saves_from_other_runs"):
        return True
    if not settings.get("uses_verified_own_training_snapshots") or settings.get("external_playthrough_or_human_save") is not False:
        return False
    origin=json.loads((run/"branch-origin.json").read_text())
    info=json.loads((run/"earned-start-provenance.json").read_text())
    parent=Path(origin["source_run"]).resolve()
    policy=json.loads((parent/"settings.json").read_text())
    record=origin["parent_record"]
    history=(parent/record["audit"]/"game.bin").resolve()
    if (not policy.get("training_forks") or policy.get("receives_saves_from_other_runs") or not record["training_only"]
            or not history.is_relative_to(parent/"discoveries")):
        return False
    source=history.read_bytes()
    starts=info["starts"]
    return (info["all_starts_verified"] and info["training_only"] and info["rom_sha1"]==ROM_SHA1
            and checksum(source)==record["tape_sha256"]==settings["source_sha256"]==info["source_sha256"]
            and [s["sha256"] for s in starts]==settings["bank_snapshot_sha256"]
            and all(0<s["prefix"]<=len(source) for s in starts) and bool(starts)
            and (tape is None or any(tape.startswith(source[:s["prefix"]]) for s in starts)))

class EarnedStarts:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.info = json.loads((self.directory/"ready.json").read_text())
        self.tape = (self.directory/"source.bin").read_bytes()
        core, _ = game_paths()
        if (self.info["source_sha256"] != checksum(self.tape) or self.info["core_sha256"] != checksum(core.read_bytes())
                or self.info["rom_sha1"] != ROM_SHA1 or self.info["frames_per_decision"] != REPEAT
                or not self.info["training_only"] or not self.info["all_starts_verified"]):
            raise ValueError("Earned starts require the verified source tape, ROM and emulator")
        self.records = self.info["starts"]

    def restore(self, game, index):
        record = self.records[index % len(self.records)]
        data = self.directory/record["filename"]
        if checksum(data.read_bytes()) != record["sha256"]:
            raise ValueError("An earned training save changed")
        with np.load(data, allow_pickle=False) as arrays:
            game.load(arrays["state"].tobytes())
            image = arrays["image"]
            score = restore_score(json.loads(arrays["score"].tobytes()))
        if score.decisions != record["prefix"] or list(game.descriptor()) != record["descriptor"]:
            raise ValueError("Saved emulator state and earned prefix disagree")
        return image, score, bytearray(self.tape[:score.decisions])


def prepare(arm, record_path, output, count=8):
    arm = arm.resolve()
    record = json.loads(record_path.read_text())
    cfg = json.loads((arm/"settings.json").read_text())
    audit_dir = (arm/record["audit"]).resolve()
    if (not cfg.get("training_forks") or cfg.get("receives_saves_from_other_runs")
            or not record["training_only"] or not audit_dir.is_relative_to(arm/"discoveries") or count < 2):
        raise ValueError("Use an original self-earned training record")
    source = (audit_dir/"game.bin").read_bytes()
    audit = json.loads((audit_dir/"game.json").read_text())
    if checksum(source) != record["tape_sha256"] or len(source) != record["length"] or not audit["uncapped_exploration"]:
        raise ValueError("Expected a matching training game with uncapped visit rewards")
    output.mkdir(parents=True, exist_ok=True)
    core, rom = game_paths()
    plan = dict(source_sha256=checksum(source), source_return_cents=record["return_cents"], source_buttons=len(source),
                core_sha256=checksum(core.read_bytes()), rom_sha1=ROM_SHA1, frames_per_decision=REPEAT, training_only=True,
                count=count, rule="evenly spaced after first earning 75% of final scalar return; reserve 32 real successor buttons")
    register = output/"registration.json"
    if register.exists() and json.loads(register.read_text()) != plan:
        raise ValueError("Another training-start selection owns this directory")
    write_json(register,plan)
    if (output/"ready.json").exists():
        EarnedStarts(output)
        return
    (output/"source.bin").write_bytes(source)
    game = Emerald(core,rom)
    score = Score(True)
    targets, records, follow = set(), [], {}
    total = 0
    try:
        game.load(game.boot)
        for n,button in enumerate(source,1):
            frame,state = game.step(button)
            reward,done = score.read(game,state)
            total += round(100*reward)
            if done:
                raise ValueError("Select a nonterminal training game")
            if not targets and total >= .75 * record["return_cents"]:
                if len(source)-32-n < count-1:
                    raise ValueError("Too few later positions to validate the requested starts")
                targets = set(map(int,np.linspace(n,len(source)-32,count,dtype=int)))
            for saved in reversed(records):
                if n-saved["prefix"] > 32:
                    break
                follow[saved["prefix"]].append(observation_hash(frame,state,reward,done))
            if n in targets:
                name = f"prefix-{n:08d}.npz"
                file = output/name
                np.savez(file,state=np.frombuffer(game.save(),np.uint8),image=frame,
                         score=np.frombuffer(json.dumps(score_record(score)).encode(),np.uint8))
                records.append(dict(prefix=n,filename=name,sha256=checksum(file.read_bytes()),descriptor=list(state),
                                    scalar_return_cents=total))
                follow[n] = []
            if n%4096==0:
                write_json(output/"status.json",dict(cold_buttons=n,expected=len(source),saved_starts=len(records)))
        if (total != record["return_cents"] or score.events != audit["events"]
                or len(score.cells) != audit["first_visit_cells"] or len(records) != count):
            raise ValueError("The cold game disagrees with its earned training record")
        for saved in records:
            with np.load(output/saved["filename"],allow_pickle=False) as arrays:
                game.load(arrays["state"].tobytes())
                restored=restore_score(json.loads(arrays["score"].tobytes()))
            start=saved["prefix"]
            assert len(follow[start])==32
            for offset,button in enumerate(source[start:start+32]):
                frame,state=game.step(button)
                reward,done=restored.read(game,state)
                if observation_hash(frame,state,reward,done)!=follow[start][offset]:
                    raise ValueError(f"Earned start {start} diverged at successor {offset+1}")
    finally:
        game.close()
    write_json(output/"ready.json",dict(**plan,starts=records,all_starts_verified=True,real_successors_compared=32*count))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("arm","record","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--count",type=int,default=8)
    args=parser.parse_args()
    prepare(args.arm,args.record,args.output,args.count)
