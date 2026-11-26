"""Check every action, reward and pixel in the continuous stock Dreamer input stream."""
import json
from pathlib import Path
import sys
import numpy as np

project = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL"
sys.path.insert(0, str(project))
from train_earned_dreamer import Trajectory, populate, write_json


def check(directory):
    game = Trajectory(directory)
    audit = game.audit
    n = len(game.tape)
    cycle = n + 1
    width = 32
    previous = np.zeros(1, np.int32)
    lane = [game.repeat(False)]
    # Reference stream comes directly from the independent chunk export:
    # each recorded button has a before-image, an after-image and the reward it caused.
    frame_shape = (40, 60, 3)
    def reference():
        while True:
            yield None
            for file in sorted(directory.glob("chunk-*.npz")):
                with np.load(file) as ds:
                    images, buttons, rewards = ds["image"], ds["action"], ds["reward"]
                yield from zip(images[1:], buttons, rewards)
    raw = reference()
    cells = raw_cents = adjusted_cents = 0
    file = sorted(directory.glob("chunk-*.npz"))[0]
    with np.load(file) as ds:
        initial = ds["image"][0]
    assert initial.shape == frame_shape
    for start in range(0, n+width+1, width):
        shape = (1,width+1)
        data = {k:np.zeros(shape, dtype) for k,dtype in
                (("consec", np.int32), ("action",np.int32), ("reward",np.float32),
                 ("is_first",bool), ("is_last",bool), ("is_terminal",bool))}
        data["image"] = np.zeros((*shape,64,64,3),np.uint8)
        populate(data,lane,previous)
        assert np.all(data["consec"]==1)
        for col in range(1, width+1):
            step = start + col - 1
            t = step % cycle
            ref = next(raw)
            assert bool(data["is_first"][0,col]) == (t==0)
            assert bool(data["is_last"][0,col]) == (t==n)
            assert bool(data["is_terminal"][0,col]) == (t==n and "stone_badge" in audit["events"])
            assert data["action"][0,col] == (game.tape[t] if t<n else 0)
            if t==0:
                assert ref is None and np.array_equal(data["image"][0,col,12:52,2:62],initial)
                assert data["reward"][0,col]==0
                continue
            pic, button, reward = ref
            assert np.array_equal(data["image"][0,col,12:52,2:62],pic)
            assert game.tape[t-1] == button == data["action"][0,col-1]
            value = round(float(reward)*100)
            if step<cycle:
                raw_cents += value
                cells += int(value%25==2)
                adjusted_cents += round(float(data["reward"][0,col])*100)
                expected = value - (2 if cells>50 and value%25==2 else 0)
                assert round(float(data["reward"][0,col])*100) == expected
    assert raw_cents==game.source["return_cents"] and cells==audit["first_visit_cells"]
    assert adjusted_cents == raw_cents-2*max(cells-50,0)
    return dict(tape=game.source["source_sha256"], actions=n, matching_images=n+1,
                incoming_actions_and_rewards_matched=n, raw_return_cents=raw_cents,
                stock_capped_return_cents=adjusted_cents, matched_wrap_to_fresh_boot=True)


if __name__ == "__main__":
    result = check(Path(sys.argv[1]))
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out,result)
    print(json.dumps(result),flush=True)
