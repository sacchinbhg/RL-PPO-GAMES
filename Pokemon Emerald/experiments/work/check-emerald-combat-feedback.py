"""Check live scalar feedback against the independently verified HP and outcome traces."""
import json
import os
from pathlib import Path
import sys

base=Path(__file__).resolve().parents[1]
project=base/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from battle_feedback import BattleFeedback
from earned_starts import EarnedStarts,checksum
from emerald import Emerald
from pixel_env import game_paths
from train_earned_dreamer import write_json

root=project/"runs/pixels-only-01/official-wsl"
out=root/"outcome-curriculum"
raw=root/"controls/endpoint-button-sweep/first-encounter-followup"
read=lambda path:json.loads(path.read_text())


if __name__=="__main__":
    os.environ.update(EMERALD_ROM=str(base/"work/pokeemerald/pokeemerald.gba"),EMERALD_CORE="/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so")
    tape=Path(read(raw/"measured.json")["source"]).with_suffix(".bin").read_bytes()
    proof=read(raw/"original-from-boot.json")
    assert checksum(tape)==proof["input_sha256"] and proof["in_game_restores"]==0
    battle=proof["recorded_battles"][-1]
    end,loss=battle["first_nonzero_outcome"]
    hp=read(raw/"battle-health.json")["hp_changes"][-1]
    bank=EarnedStarts(root/"controls/dreamer-earned-resets/starts")
    n,i=max((r["prefix"],i) for i,r in enumerate(bank.records) if tape.startswith(bank.tape[:r["prefix"]]) and r["prefix"]<battle["first_battle_frame"])
    game=Emerald(*game_paths())
    try:
        bank.restore(game,i)
        signal=BattleFeedback()
        pulses={}
        for step,b in enumerate(tape[n:],n+1):
            _,key=game.step(b)
            value=signal.read(game,key)
            if value:
                pulses[step]=value
        assert loss==2 and signal.wins==0 and signal.losses==1
        assert set(pulses)=={end,hp["decision"]} and pulses[end]==-1
        entry=read(raw/"battle-health.json")["entry_party"]
        assert abs(pulses[hp["decision"]]+.5*entry["hp"]/entry["maximum"])<1e-6
        loss_audit=signal.audit()
        wins=[b["first_nonzero_outcome"][0] for b in proof["recorded_battles"]
              if b["first_nonzero_outcome"] and b["first_nonzero_outcome"][1]==1]
        _,decision,idx=min((win-r["prefix"],win,idx) for idx,r in enumerate(bank.records) for win in wins
                           if r["prefix"]<win and tape.startswith(bank.tape[:r["prefix"]]))
        begin=bank.records[idx]["prefix"]
        bank.restore(game,idx)
        positive=BattleFeedback()
        positive_steps=[]
        health_rewards=[]
        for step,b in enumerate(tape[begin:decision+8],begin+1):
            _,key=game.step(b)
            before=positive.wins
            value=positive.read(game,key)
            if positive.wins>before:
                positive_steps.append(step)
            elif value:
                health_rewards.append((step,value))
        assert positive_steps==[decision] and positive.losses==0
        out.mkdir(exist_ok=True)
        write_json(out/"feedback-check.json",dict(checked_tape=checksum(tape),verified_prefix=n,replayed_buttons=len(tape)-n,
                   actual_nonzero_rewards=pulses,checked_against_original_game=True,feedback=loss_audit,
                   positive_test=dict(prefix=begin,verified_win_decision=decision,observed_win_decisions=positive_steps,
                                      health_rewards=health_rewards,feedback=positive.audit())))
    finally:
        game.close()
