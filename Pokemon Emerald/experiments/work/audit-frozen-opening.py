"""Replay the first failed frozen game and measure time on each opening screen."""
from collections import Counter
import json
from pathlib import Path
import sys
from PIL import Image

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from emerald import Emerald, PHASES, BUTTONS
from pixel_env import Score, game_paths

arm=project/"runs/pixels-only-01/official-wsl/seed-71299-h15-reward-replay"
audit,=(arm/"eval-262144/external-audit").glob("*/env-0/*.json")
out=arm/"highlights/frozen-opening-262144"
out.mkdir(parents=True,exist_ok=True)
tape=audit.with_suffix(".bin").read_bytes()
scorer=Score()
phases, last_quarter = Counter(),Counter()
game=Emerald(*game_paths())
try:
    game.load(game.boot)
    for n,button in enumerate(tape,1):
        _,key=game.step(button)
        scorer.read(game,key)
        phases[PHASES[key[0]]]+=1
        if n>24576:
            last_quarter[PHASES[key[0]]]+=1
        if n in (2048,8192,16384,32768):
            Image.fromarray(game.screen).save(out/f"decision-{n}.png")
            print(n,PHASES[key[0]],flush=True)
    assert scorer.events == json.loads(audit.read_text())["events"]
finally:
    game.close()
result=dict(source=audit.relative_to(arm).as_posix(),buttons=len(tape),events=scorer.events,
            phase_counts=phases,final_quarter_phase_counts=last_quarter,
            action_counts=Counter(BUTTONS[x] for x in tape))
(out/"verification.json").write_text(json.dumps(result,indent=2))
