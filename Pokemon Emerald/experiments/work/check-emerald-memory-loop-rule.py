"""Check the recovery rule on known advancing frames and a deliberately repeated real training image."""
import json
from pathlib import Path
import sys

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from earned_memory import Memory
from train_earned_dreamer import Trajectory,write_json

root=project/"runs/pixels-only-01/official-wsl"
model=root/"earned-screen-memory/model"
data=next((root/"reward-only-training-forks/verified-training-tapes").glob("24479561ff0b*"))
out=root/"controls/retrieval-loops"


def run():
    game=Trajectory(data)
    controls=[Memory(model,197461,True,guard) for guard in (False,True)]
    changing_frame=None
    seen=0
    for index,(frame,button,reward,first,last,terminal) in enumerate(game.episode()):
        choices=[a.choose(frame,reward) for a in controls]
        assert choices[0]==choices[1]
        for a in controls:
            a.executed(button)
        seen+=1
        trace=controls[0].traces[0]
        if 2<index<256 and changing_frame is None and trace["keys"][index]!=trace["keys"][index+1]:
            changing_frame=(index,frame.copy())
        if seen>=512:
            break
    index,frame=changing_frame
    agents=[Memory(model,197461,True,guard) for guard in (False,True)]
    for agent in agents:
        agent.expected=(0,index)
        agent.phase=int(agent.traces[0]["phase"][index])
    first_changed=None
    histories=[[],[]]
    for decision in range(1,65):
        choices=[a.choose(frame,0.) for a in agents]
        if choices[0]!=choices[1] and first_changed is None:
            first_changed=decision
        for i,a in enumerate(agents):
            histories[i].append(choices[i])
            a.executed(choices[0])
    assert agents[1].first_recovery is not None and first_changed>=agents[1].first_recovery
    assert agents[1].usage["loop_recovery"]>=16 and len(set(histories[1]))>1
    write_json(out/"mechanism-verified.json",dict(model_sha256=agents[0].identity["model_sha256"],training_tape=game.source["source_sha256"],
        original_advancing_frames_with_identical_choices=seen,real_training_image_index=index,deliberate_fixed_image_repeats=64,
        first_recovery=agents[1].first_recovery,first_changed_choice=first_changed,recovery_actions=agents[1].usage["loop_recovery"],
        heldout_game_progress_measured=False))


if __name__=="__main__":
    run()
