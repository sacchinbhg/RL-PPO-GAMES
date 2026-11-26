"""Measure distinct screens after real camera movements on an existing training game."""
from collections import Counter
import json
from pathlib import Path
import sys

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from fit_visual_motion import translation
from reward_branch_search import visual_code
from screen_explorer import screen_key
from train_earned_dreamer import Trajectory,write_json

root=project/"runs/pixels-only-01/official-wsl"
out=root/"controls/unrewarded-motion-landmarks"


def measure():
    probe=json.loads((root/"controls/visual-motion/route-training.json").read_text())
    game=Trajectory(next((root/"paired-self-earned-frontiers/visual-slots-0/verified-training-tapes").glob("e30822*")))
    assert probe["informative"] and probe["training_tape_sha256"]==game.source["source_sha256"]
    shifts={int(b):tuple(row["most_common"]) for b,row in probe["learned_actions"].items()}
    write_json(out/"registration.json",dict(training_source=game.source["source_sha256"],scope="after the last scalar payment of at least 1",
        requirement="at least sixteen distinct camera-scrolled RGB observations unavailable in paid observations",
        candidate="receive zero scalar reward while the preceding button produced its learned camera shift",
        model_inputs="gameplay pixels and actions",reward="uncapped per-episode rewards for training exploration"))
    counts=Counter()
    neutral=[set(),set()]
    paid=[set(),set()]
    functions=(visual_code,screen_key)
    for i,(frame,button,reward,first,last,terminal) in enumerate(game.episode(True)):
        if reward>=1:
            counts.clear()
            neutral=[set(),set()]
            paid=[set(),set()]
            counts["stage_start"]=i
        if reward>0:
            counts["paid_positions"]+=1
            for store,fn in zip(paid,functions):
                store.add(fn(frame))
        elif not first and prior in shifts and translation(previous,frame)==shifts[prior]:
            counts["neutral_scrolling_positions"]+=1
            for store,fn in zip(neutral,functions):
                store.add(fn(frame))
        previous,prior=frame,button
    stats={}
    for label,a,b in zip(("coarse","fine"),neutral,paid):
        stats[label]=dict(distinct_neutral_scroll_codes=len(a),distinct_paid_codes=len(b),
                          neutral_scroll_codes_absent_from_paid=len(a-b))
    write_json(out/"measured.json",dict(training_source=game.source["source_sha256"],steps=len(game.tape),**counts,
               distinct=stats,qualifies_for_comparison=stats["fine"]["neutral_scroll_codes_absent_from_paid"]>=16))


if __name__=="__main__":
    out.mkdir(parents=True,exist_ok=True)
    measure()
