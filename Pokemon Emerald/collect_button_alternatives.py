"""Measure all nine first-button alternatives around scalar rewards in an agent's own training games."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from reward_branch_env import ForkEnv
from train_earned_dreamer import Trajectory, digest, write_json

HORIZON = 15
HISTORY = 33


def effective(cents, visits):
    new_cell = cents % 25 == 2
    visits += int(new_cell)
    return cents - (2 if new_cell and visits > 50 else 0), visits


def collect(datasets, output):
    output.mkdir(parents=True, exist_ok=True)
    games = [Trajectory(p) for p in datasets]
    sources = [g.source["source_sha256"] for g in games]
    plan = dict(training_tapes=sources, history=HISTORY, horizon=HORIZON, first_actions=9,
                origins="all scalar rewards of at least one with offsets 0, 4, 12, and up to 16 evenly spaced small visit rewards in the second half",
                targets="15 real scalar rewards for each alternative, subsequent recorded buttons fixed",
                frozen_evaluation_used=False)
    setup=output/"plan.json"
    if setup.exists() and json.loads(setup.read_text()) != plan:
        raise ValueError("The requested collection differs from the registered experiment")
    write_json(setup, plan)
    seen = set()
    summaries = []
    for folder, game in zip(datasets, games):
        sha = game.source["source_sha256"]
        dest = output / sha
        if dest.exists():
            saved = json.loads((dest/"manifest.json").read_text())
            if digest(dest/"windows.npz") != saved["data_sha256"]:
                raise ValueError("A completed training export changed")
            seen.update(saved["prefix_hashes"])
            summaries.append(saved)
            continue
        raw = []
        for chunk in sorted(folder.glob("chunk-*.npz")):
            with np.load(chunk) as segment:
                raw.extend(np.rint(100*segment["reward"]).astype(np.int32).tolist())
        if len(raw) != len(game.tape) or sum(raw) != game.source["return_cents"] or game.source.get("frozen_evaluation_data_used"):
            raise ValueError("Expected an intact self-generated trajectory with its original scoring")
        raw = np.asarray(raw)
        targets = set()
        for index in np.flatnonzero(raw >= 100):
            targets.update(int(index - delay) for delay in (0,4,12))
        small = np.flatnonzero((raw==2) & (np.arange(len(raw))>=len(raw)//2))
        if len(small):
            targets.update(map(int,small[np.linspace(0,len(small)-1,min(16,len(small)),dtype=int)]))
        targets = sorted(t for t in targets if t>=HISTORY-1 and t+HORIZON<=len(raw))
        selected = {}
        for t in targets:
            # Identical action histories ending before the intervention are one training state.
            prefix = hashlib.sha256(game.tape[:t]).hexdigest()
            if prefix not in seen:
                selected[t] = prefix
                seen.add(prefix)
        if not selected:
            continue
        env = ForkEnv(output/"first-maps",trainer_audits=output/"first-trainers")
        images = deque([env.initial], maxlen=HISTORY)
        contexts, past, futures, outcomes, full_outcomes, indices, hashes = [], [], [], [], [], [], []
        checks, matched, visits, new_steps = {}, 0, 0, 0
        last = max(selected)+HORIZON
        try:
            for t, original_button in enumerate(game.tape[:last]):
                if t in selected:
                    env.command("save",0)
                    ctx = np.stack(images)
                    assert len(ctx)==HISTORY
                    recorded = game.tape[t:t+HORIZON]
                    buttons, real, uncapped = [], [], []
                    for first in range(9):
                        restored = env.command("restore",0)
                        if not np.array_equal(restored,ctx[-1]):
                            raise RuntimeError("A training fork returned a different observation")
                        sequence = bytes([first])+recorded[1:]
                        row, raw_row = [], []
                        branch_visits = visits
                        for action in sequence:
                            im, cents, done, used = env.command("step",(action,1))
                            if used!=1:
                                raise ValueError("A decision must advance one emulator action")
                            value, branch_visits = effective(cents,branch_visits)
                            row.append(value)
                            raw_row.append(cents)
                            new_steps += 1
                            if done:
                                env.command("audit",str(output/"terminal"/f"{sha[:12]}-{t}-{first}"))
                                row.extend([0]*(HORIZON-len(row)))
                                raw_row.extend([0]*(HORIZON-len(raw_row)))
                                break
                        if first==original_button:
                            original = raw_row if game.audit["uncapped_exploration"] else row
                            if original != raw[t:t+HORIZON].tolist():
                                raise ValueError("The original action failed to reproduce the training rewards")
                            checks[t+HORIZON-1]=hashlib.sha256(im).hexdigest()
                        buttons.append(list(sequence))
                        real.append(row)
                        uncapped.append(raw_row)
                    env.command("restore",0)
                    contexts.append(ctx)
                    past.append(list(game.tape[t-HISTORY+1:t]))
                    futures.append(buttons)
                    outcomes.append(real)
                    full_outcomes.append(uncapped)
                    indices.append(t+1)
                    hashes.append(selected[t])
                frame,cents,done,used=env.command("step",(original_button,1))
                capped,visits=effective(cents,visits)
                original = cents if game.audit["uncapped_exploration"] else capped
                if original!=raw[t] or (done and t+1<len(raw)) or used!=1:
                    raise ValueError("Cold replay disagrees with the self-earned training episode")
                images.append(frame)
                if t in checks:
                    if hashlib.sha256(frame).hexdigest()!=checks[t]:
                        raise ValueError("The matched action did not reproduce the later original screen")
                    matched+=1
                if (t+1)%4096==0:
                    write_json(output/"status.json",dict(source=sha,cold_buttons=t+1,expected=last,
                               training_states=len(indices),branch_buttons=new_steps))
        finally:
            env.close()
        assert matched==len(selected) and 0 < new_steps <= 9*HORIZON*len(selected)
        with tempfile.TemporaryDirectory(dir=output) as temp:
            draft=Path(temp)/"game"
            draft.mkdir()
            scores=np.asarray(outcomes,dtype=np.float32)/100
            np.savez_compressed(draft/"windows.npz",context=np.asarray(contexts),past=np.asarray(past,np.int32),
                future=np.asarray(futures,np.int32),real_rewards=scores,uncapped_rewards=np.asarray(full_outcomes,np.float32)/100,
                decision=np.asarray(indices))
            with (draft/"windows.npz").open("rb") as f:
                digest=hashlib.file_digest(f,"sha256").hexdigest()
            summary=dict(training_tape=sha,distinct_training_states=len(indices),prefix_hashes=hashes,emulated_alternative_buttons=new_steps,
                         original_prefix_buttons=last,original_action_matched=matched,data_sha256=digest,
                         action_sensitive_states=int((np.ptp(scores.sum(-1),axis=1)>.001).sum()))
            write_json(draft/"manifest.json",summary)
            draft.rename(dest)
        summaries.append(summary)
    write_json(output/"completed.json",dict(games=summaries,training_states=sum(x["distinct_training_states"] for x in summaries),
               additional_alternative_buttons=sum(x["emulated_alternative_buttons"] for x in summaries),
               cold_replay_buttons=sum(x["original_prefix_buttons"] for x in summaries)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,action="append",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    collect(args.dataset,args.output)
