"""Measure the official Dreamer report on the game's visible RGB region."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def measure(truth, model, last):
    model_error = np.abs(truth.astype(np.int16) - model.astype(np.int16))
    copy_error = np.abs(truth.astype(np.int16) - last.astype(np.int16))
    changed = copy_error.max(-1) >= 8
    return dict(model_rgb_mae=float(model_error.mean()), last_frame_rgb_mae=float(copy_error.mean()),
                changed_pixel_fraction=float(changed.mean()),
                model_changed_rgb_mae=float(model_error[changed].mean()) if changed.any() else None,
                last_changed_rgb_mae=float(copy_error[changed].mean()) if changed.any() else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("report", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    with np.load(args.report) as data:
        frames = data["frames"]
    count, h, w, colors = frames.shape
    assert h == 196 and w % 68 == 0 and colors == 3 and frames.dtype == np.uint8
    green = np.flatnonzero(np.all(frames[:,0,0] == [0,255,0], axis=-1))
    red = np.flatnonzero(np.all(frames[:,0,0] == [255,0,0], axis=-1))
    assert len(green) and len(red) and green[-1]+1 == red[0] and red[-1] < count
    grid = frames.reshape(count, h, w // 68, 68, 3).transpose(2,0,1,3,4)
    truth = grid[:,:,14:54,4:64]
    pred = grid[:,:,78:118,4:64]
    target = truth[:,red]
    last = truth[:,green[-1]:green[-1]+1]
    metrics = dict(source=args.report.name, sequences=w//68, conditioned_decisions=len(green),
                   predicted_decisions=len(red), rgb_scale="0 through 255; game content only",
                   overall=measure(target, pred[:,red], last),
                   by_horizon={str(k+1):measure(target[:,k:k+1], pred[:,red[k:k+1]], last)
                               for k in range(len(red))})
    # Preselect the middle future time and the median-changing sequence for the illustration.
    k = (len(red)-1)//2
    change = (np.abs(target[:,k].astype(float)-last[:,0]) >= 8).any(-1).mean((1,2))
    order = np.argsort(change, kind="stable")
    i = int(order[(len(order)-1)//2])
    metrics.update(illustrated_sequence=i, illustration_horizon=k+1)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"prediction-errors.json").write_text(json.dumps(metrics, indent=2))
    canvas = Image.new("RGB", (960, 284), "#15181d")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=19)
    draw.text((14,12), f"Emerald, {k+1} button decisions after the last observed frame",
              font=font, fill="white")
    panels = [(target[i,k], "Recorded game"), (pred[i,red[k]], "Dreamer prediction"),
              (last[i,0], "Last observed frame")]
    for j,(frame,label) in enumerate(panels):
        x = 12+j*316
        canvas.paste(Image.fromarray(frame).resize((300,200), Image.Resampling.NEAREST), (x,48))
        draw.text((x,254),label,font=font,fill="white")
    canvas.save(args.out/"prediction-comparison.png")
    print(json.dumps({k:v for k,v in metrics.items() if k!="by_horizon"}), flush=True)


if __name__ == "__main__":
    main()
