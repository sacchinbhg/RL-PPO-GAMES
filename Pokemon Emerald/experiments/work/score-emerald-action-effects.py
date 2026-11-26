"""Compare predicted and real effects of reversing the same future button list."""
import json
from pathlib import Path
import numpy as np

out = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL/runs/pixels-only-01/official-wsl/controls/heldout-world-model"


def predicted(name):
    with np.load(out/f"{name}.npz") as d:
        grid=d["frames"]
    red=np.all(grid[:,0,0]==[255,0,0],axis=-1)
    return grid[red].reshape(16,196,8,68,3).transpose(2,0,1,3,4)[:,:,78:118,4:64].astype(np.int16)


with np.load(out/"alternative-futures.npz") as data:
    actual, alternative = (data[k].astype(np.int16) for k in ("recorded","reversed"))
wanted, scrambled = predicted("recorded"), predicted("reversed")
changed=np.abs(actual-alternative).max(-1)>=8
assert changed.any()


def score(mask):
    if not mask.any():
        return None
    correct = (np.abs(wanted-actual)+np.abs(scrambled-alternative))/2
    swapped = (np.abs(wanted-alternative)+np.abs(scrambled-actual))/2
    return dict(paired_mae=float(correct[mask].mean()), swapped_mae=float(swapped[mask].mean()),
                real_action_effect_magnitude=float(np.abs(actual-alternative)[mask].mean()),
                model_action_effect_magnitude=float(np.abs(wanted-scrambled)[mask].mean()),
                action_effect_error=float(np.abs((wanted-scrambled)-(actual-alternative))[mask].mean()))


result=dict(changed_pixel_fraction=float(changed.mean()), changed_pixels=score(changed),
            all_pixels=score(np.ones_like(changed)), windows_with_changed_pixels=int(changed.any((1,2,3)).sum()),
            origin="Eight fixed fresh-evaluation prefixes; two new button orders played from the same original-ROM state.")
(out/"button-effects.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result), flush=True)
