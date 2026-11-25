"""Recreate an overnight discovery from boot and capture its first visible post-warp frame."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from emerald import BUTTONS, Emerald
from experiment import location, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    folder=parser.parse_args().directory
    cfg=json.loads(Path(__file__).with_name("local.json").read_text(encoding="utf-8-sig"))
    event=json.loads((folder/"event.json").read_text())
    actions=(folder/"buttons.bin").read_bytes()
    if len(actions)!=event["recorded_buttons"]:
        raise ValueError("Incomplete recorded route")
    env=Emerald(cfg["core"],cfg["rom"])
    try:
        key=env.descriptor()
        for button in actions:
            _, key=env.step(button)
        with Image.open(folder/"frame.png") as original:
            same_frame=np.array_equal(env.screen, np.asarray(original))
        result=dict(source="fresh boot, recorded training buttons", decisions=len(actions),
                    expected=event["descriptor"], observed=key, map=location(key),
                    matched=list(key)==event["descriptor"], matched_frame=bool(same_frame))
        if result["matched"]:
            # The game changes its RAM map during the fade. Keep the actual arrival
            # image and capture a visible follow-up separately, with explicit inputs.
            for waits in range(17):
                if location(key)==event["map"] and env.screen.std()>30 and env.screen.mean()>40:
                    Image.fromarray(env.screen).save(folder/"replay-visible.png")
                    result.update(visible_frame="replay-visible.png", extra_wait_decisions=waits)
                    break
                if waits < 16:
                    _,key=env.step(BUTTONS.index("wait"))
        write_json(folder/"replay.json",result)
        print(json.dumps(result))
        if not result["matched"]:
            raise SystemExit(1)
    finally:
        env.close()


if __name__=="__main__":
    main()
