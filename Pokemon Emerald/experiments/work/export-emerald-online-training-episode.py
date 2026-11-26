"""Recover the highest scalar-return episode from verified official training replay and check the original ROM."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from emerald import Emerald,verify_rom
from pixel_env import Score,game_paths
from reward_branch_search import write_json

root=project/"runs/pixels-only-01/official-wsl"
arm=root/"seed-71299-h15-screen-collection"
output=root/"earned-online-datasets"


def choose():
    train=arm/"train"
    origin=json.loads((train/"branch-origin.json").read_text())
    verified=json.loads((arm/"verified-training-experience.json").read_text())
    audit={row["tape_sha256"]:row for row in verified["games"]}
    inherited=origin["source_replay_files"]
    paths=[p for p in (train/"replay").glob("*.npz") if p.name not in inherited]
    lookup={p.stem.split("-")[1]:p for p in paths}
    successors={p.stem.split("-")[2] for p in paths}
    candidates=[]
    for first in sorted(lookup.keys()-successors):
        chain=[]
        entries={k:[] for k in ("action","reward","is_first","is_last")}
        p=lookup[first]
        while p is not None:
            chain.append(p)
            with np.load(p) as d:
                for k in entries:
                    entries[k].append(d[k])
            p=lookup.get(p.stem.split("-")[2])
        acts,rew,starts,last=(np.concatenate(v) for v in entries.values())
        for n in np.flatnonzero(starts):
            ends=np.flatnonzero(last[n:])
            if not len(ends):
                continue
            finish=int(n+ends[0])
            sha=hashlib.sha256(acts[n:finish].astype(np.uint8).tobytes()).hexdigest()
            if sha in audit:
                total=int(np.rint(rew[n:finish+1]*100).sum())
                candidates.append((total,sha,int(n),finish,chain,audit[sha]))
    assert len(candidates)==4 and verified["training_step"]==524304
    return max(candidates,key=lambda x:(x[0],-(x[3]-x[2]))),[(r[0],r[1]) for r in candidates]


def run():
    (total,sha,start,end,chain,record),candidates=choose()
    output.mkdir(exist_ok=True)
    dest=output/sha
    if (dest/"manifest.json").exists():
        return
    official,=(arm/"train/external-audit").glob(f"*/env-{record['worker']}/*episode-00001.json")
    tape=official.with_suffix(".bin").read_bytes()
    assert hashlib.sha256(tape).hexdigest()==sha
    expected=json.loads(official.read_text())
    core,rom=game_paths()
    verify_rom(rom)
    game=Emerald(str(core),str(rom))
    score=Score(expected["uncapped_exploration"])
    try:
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            draft=Path(tmp)/"dataset"
            draft.mkdir()
            game.load(game.boot)
            frame=game.boot_image
            offset=at=matched=0
            visual_first=None
            frames=[]
            rewards,ends=[],[]
            for path in chain:
                with np.load(path,allow_pickle=False) as d:
                    n=len(d["action"])
                    lo,hi=max(0,start-offset),min(n,end+1-offset)
                    if hi>lo:
                        image=d["image"][lo:hi,12:52,2:62]
                        actual=d["action"][lo:hi]
                        training_reward=d["reward"][lo:hi]
                        term=d["is_terminal"][lo:hi]
                        for j in range(hi-lo):
                            if at:
                                frame,key=game.step(tape[at-1])
                                reward,done=score.read(game,key)
                                if np.float32(reward)!=training_reward[j] or done!=term[j]:
                                    raise ValueError(f"Original-ROM reward or termination differs at {at}")
                                rewards.append(training_reward[j])
                                ends.append(term[j])
                            elif training_reward[j]!=0 or term[j]:
                                raise ValueError("Opening training observation is not a genuine reset")
                            if at<len(tape) and actual[j]!=tape[at]:
                                raise ValueError("Stored action differs from the immutable training tape")
                            if np.array_equal(frame,image[j]):
                                matched+=1
                            elif visual_first is None:
                                visual_first=at
                            frames.append(image[j].copy())
                            if len(rewards)==256 or at==len(tape):
                                k=at-len(rewards)
                                np.savez_compressed(draft/f"chunk-{k:09d}.npz",image=np.stack(frames),
                                    action=np.frombuffer(tape[k:at],dtype=np.uint8),reward=np.asarray(rewards,np.float32),
                                    terminal=np.asarray(ends,bool))
                                frames=[frames[-1]]
                                rewards,ends=[],[]
                            at+=1
                offset+=n
                if at and at%2048<1024:
                    write_json(output/"export-status.json",dict(source_sha256=sha,checked_observations=at,matching_images=matched))
            assert at==len(tape)+1 and score.events==expected["events"] and len(score.cells)==expected["first_visit_cells"]
            (draft/"audit").mkdir()
            shutil.copyfile(official,draft/"audit/game.json")
            shutil.copyfile(official.with_suffix(".bin"),draft/"audit/game.bin")
            write_json(draft/"manifest.json",dict(source_sha256=sha,decisions=len(tape),source_seed=71299,
                search_buttons_at_discovery=None,source_training_forks=0,source_kind="official_training_worker",
                behavior=record["behavior"],source_audit=str(official),stored_replay_actions_verified=True,
                cold_replay_rewards_verified=len(tape),cold_replay_identical_images=matched,first_different_image=visual_first,
                replay_restores_during_game=0,frozen_evaluation_data_used=False,return_cents=total,chunk_length=256,
                candidate_scalar_returns=candidates,selection="highest recorded training return among four complete workers",
                image_source="original images in stock learner replay",alignment="button i takes image i to i+1"))
            draft.rename(dest)
    finally:
        game.close()


if __name__=="__main__":
    run()
