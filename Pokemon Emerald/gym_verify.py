"""Independently replay a frozen policy's original-ROM button log and certify the Stone Badge."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image
from gym_data import ROOT, RUN
from gym_explore import Game
from experiment import write_json


def verify(directory):
    record=json.loads((directory/"evaluation.json").read_text())
    tape=(directory/"buttons.bin").read_bytes()
    if "stone_badge" not in record["events"] or record["resets"]!=0:
        raise ValueError("Evaluation is incomplete")
    digest=hashlib.sha256(tape).hexdigest()
    assert digest==record["tape_sha256"]
    out=directory/"verification"
    out.mkdir(exist_ok=True)
    g=Game()
    trainer_step=first_badge=None
    battle_outcomes=set()
    first_both=None
    try:
        for n,button in enumerate(tape,1):
            g.step(button)
            s=g.state()
            if not s or s["name"]!="RustboroCity_Gym":continue
            if s["phase"]==6 and g.num(0x02038bca,2)==265 and g.num(0x02022fec,4)&8:
                if trainer_step is None:
                    trainer_step=n
                if n==trainer_step+35:
                    Image.fromarray(g.env.screen).save(out/"roxanne-battle.png")
                battle_outcomes.add(g.num(0x0202433a))
            if s["phase"]==5 and s["badge"] and s["defeated"] and s["party"]>0:
                if first_badge is None:
                    first_badge=n
                    Image.fromarray(g.env.screen).save(out/"stone-badge.png")
                    first_both=s.copy()
        assert trainer_step and first_badge and trainer_step<first_badge
        assert first_badge==record["initial_waits"]+record["events"]["stone_badge"]
        cfg=json.loads((ROOT/"local.json").read_text(encoding="utf-8-sig"))
        evidence=dict(verified=True,boot="new mGBA instance",resets=0,ram_writes=0,
                      trainer_id=265,trainer_battle_decision=trainer_step,stone_badge_decision=first_badge,
                      defeated_rustboro_gym_flag="0x4f0",stone_badge_flag="0x867",
                      state_when_both_flags_first_verified=first_both,battle_outcomes_seen=sorted(battle_outcomes),
                      rom_sha1=hashlib.sha1(Path(cfg["rom"]).read_bytes()).hexdigest(),
                      core_sha256=hashlib.sha256(Path(cfg["core"]).read_bytes()).hexdigest(),buttons_sha256=digest)
        write_json(out/"verification.json",evidence)
        print(json.dumps(evidence),flush=True)
    finally:g.env.close()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("evaluation")
    verify(RUN/p.parse_args().evaluation)
