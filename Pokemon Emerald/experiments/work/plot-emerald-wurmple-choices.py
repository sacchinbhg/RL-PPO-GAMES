"""Put the original emulator frames side by side for the experiment diary."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

root=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL/runs/pixels-only-01/official-wsl/controls/retrieval-loops"
folder=root/"heldout-button-effects"
proof=json.loads((folder/"result.json").read_text())
assert proof["full_rgb_start_matched"] and proof["original_buttons"]==50000
assert {k for k,v in proof["outcomes"].items() if v["persistent"]}=={"A","B"}
images=[np.array(Image.open(folder/(name+".png"))) for name in ("wait","Select","A","B")]
fig,axes=plt.subplots(1,3,figsize=(11,3.8),layout="constrained")
for ax,pic,label in zip(axes,(images[0],images[1],images[2]),("Wait","Select","A")):
    ax.imshow(pic,interpolation="nearest")
    ax.set_title(label,loc="left",fontsize=14,weight="bold")
    ax.axis("off")
fig.suptitle("What happened when we changed one button?",x=.01,ha="left",weight="bold",fontsize=19)
fig.supxlabel("Same Wurmple encounter after 50,000 recorded decisions. Each button is followed by three waits. B also advanced the encounter.",ha="left",x=.01,fontsize=10)
fig.savefig(folder/"same-encounter.png",dpi=150)
plt.close(fig)
