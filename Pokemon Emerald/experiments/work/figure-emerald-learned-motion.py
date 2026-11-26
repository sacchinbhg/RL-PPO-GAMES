"""Plot the aggregate held-out result and the first held-out choice where directions matter."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

project = Path(__file__).resolve().parents[1] / "outputs/EmeraldRL"
sys.path.insert(0, str(project))
from emerald import BUTTONS
from fit_local_motion import hashes, load, ScrollModel

out = project / "runs/pixels-only-01/official-wsl/controls/local-screen-motion"
train, _, _ = load(out / "train")
held, truth, _ = load(out / "test")
model = ScrollModel(out / "fitted")
report = json.loads((out / "decision.json").read_text())["measured"]
eligible = ~np.isin(hashes(held), hashes(train)) & (truth.min(1) != truth.max(1))
index = np.flatnonzero(eligible)[0]
with np.load(out / "test/examples.npz") as data:
    decision = int(data["decision"][index])
frame = held[index]
predictions = model.predictions(frame)
assert eligible.sum() == report["action_sensitive_positions"]
fig = plt.figure(figsize=(11.2, 6.9), layout="constrained")
grid = fig.add_gridspec(3, 4, height_ratios=[.95, 1.32, .22])
chart = fig.add_subplot(grid[0, :])
labels = ["Uniform direction", "Button frequency", "Whole-screen model", "Local image model"]
values = [report["uniform"]] + [report["results"][k]["realized_scroll"] for k in
                                 ("button_frequency", "whole_screen", "local_screen")]
rows = np.arange(4)
chart.barh(rows, values, color=["#849199", "#849199", "#597c96", "#167852"], height=.57)
for i, value in enumerate(values):
    chart.text(value+.012, i, f"{value:.1%}", va="center", fontsize=10, weight="bold")
chart.set(xlim=(0,1), yticks=rows, yticklabels=labels, xticks=[0,.25,.5,.75,1],
          xticklabels=["0%", "25%", "50%", "75%", "100%"])
chart.set_title("How often did the chosen direction move the camera?", fontsize=17, weight="bold", loc="left", pad=12)
chart.invert_yaxis()
chart.spines[["right", "top", "left"]].set_visible(False)
chart.tick_params(axis="y", length=0)
for j, (button, (top, left)) in enumerate(zip(model.buttons, model.patches)):
    ax = fig.add_subplot(grid[1, j])
    ax.imshow(frame, interpolation="nearest")
    ax.add_patch(Rectangle((left-.5, top-.5), 12, 12, fill=False, edgecolor="white", linewidth=3.5))
    ax.add_patch(Rectangle((left-.5, top-.5), 12, 12, fill=False, edgecolor="#d94f12", linewidth=1.6))
    ax.set_title(BUTTONS[button], weight="bold", fontsize=13)
    ax.set(xticks=[], yticks=[])
    ax.set_xlabel(f"Model score {predictions[j]:.0%}\nCamera {'scrolled' if truth[index,j] else 'stayed put'}",
                  fontsize=10, linespacing=1.5)
foot = fig.add_subplot(grid[2, :])
foot.axis("off")
foot.text(0, .9, "Top: 78 held-out screens where the four directions had different outcomes. Ties share credit.\n"
          f"Below: first such screen, decision {decision:,}. Boxes show the 12 × 12 pixels learned for each button. "
          "Every direction was tried for three decisions.", va="top", fontsize=9, linespacing=1.5)
fig.savefig(out / "heldout-motion.png", dpi=150, bbox_inches="tight", pad_inches=.18)
plt.close(fig)
